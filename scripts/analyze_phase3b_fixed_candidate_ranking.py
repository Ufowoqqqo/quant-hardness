#!/usr/bin/env python3
"""Regenerate fixed-candidate ranking measurements from packed raw scores."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from phase3b_metrics import measure
from analyze_phase3a_sift1m import read_conf, spearman, write_csv

DTYPE = np.dtype([("id", "<i4"), ("exact", "<f4"), ("pq", "<f4")])
FEATURES = ["candidate_mae", "relative_boundary_margin", "boundary_local_mae",
            "error_scale_over_boundary_margin", "cross_boundary_inversion_count",
            "cross_boundary_inversion_fraction", "maximum_violation",
            "topk_agreement", "random_inversion_rate", "candidate_count"]
CLASSES = ["Stable", "Benign", "Harmful", "Lucky"]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def derive(config_path, raw_root, output_root, ef):
    cfg = read_conf(config_path)
    raw = raw_root / f"ef{ef}"
    manifest = json.loads((raw / "manifest.json").read_text())
    for key in ("graph_fingerprint", "codebook_sha256", "codes_sha256"):
        assert manifest[key] == cfg[key], key
    out = output_root / f"ef{ef}"
    out.mkdir(parents=True, exist_ok=False)
    maps = {p.name: np.memmap(p, mode="r", dtype=DTYPE)
            for p in sorted(raw.glob("candidate_scores_*.bin"))}
    prior_path = Path(cfg["phase3a_run"]) / f"decomposition_ef{ef}.jsonl"
    count, candidates, mismatches = 0, 0, 0
    non_tied_mismatches = []
    with (raw/"queries.jsonl").open() as queries, prior_path.open() as previous, \
            (out/"queries.jsonl").open("x") as scalar, \
            (out/"harmful_pairs.jsonl").open("x") as harmful, \
            (out/"random_pair_indices.bin").open("xb") as samples:
        for qid, line in enumerate(queries):
            meta, old = json.loads(line), json.loads(next(previous))
            assert meta["query_id"] == old["query_id"] == qid
            assert meta["ef_search"] == ef
            n = meta["candidate_count"]
            start = meta["byte_offset"] // DTYPE.itemsize
            a = maps[meta["candidate_file"]][start:start+n]
            assert len(a) == n
            assert meta["native_ids"] == old["pq_native_result_ids"]
            assert meta["ground_truth_ids"] == old["ground_truth_ids"]
            assert n == old["pq_L0_distance_evaluations"]
            if old["pq_L0_evaluated_ids"] is not None:
                assert sorted(a["id"].tolist()) == old["pq_L0_evaluated_ids"]
            row, pairs, indices = measure(
                a["id"], a["exact"], a["pq"], meta["native_ids"],
                meta["ground_truth_ids"], int(cfg["k"]),
                int(cfg["random_pair_seed"])+qid, int(cfg["random_pairs_per_query"]),
                float(cfg["epsilon"]))
            assert row["exact_topk_ids"] == old["pq_L0_oracle_result_ids"]
            assert abs(row["oracle_recall"]-old["recall_pq_L0_oracle"]) < 1e-12
            lookup = dict(zip(a["id"].tolist(), a["pq"].tolist()))
            assert [lookup[i] for i in meta["native_ids"]] == meta["native_pq_distances"]
            row.update(query_id=qid, ef_search=ef,
                       graph_fingerprint=manifest["graph_fingerprint"],
                       ground_truth_ids=meta["ground_truth_ids"],
                       phase3a_exact_native_recall=old["recall_exact_native"],
                       pq_candidate_coverage=old["pq_L0_candidate_coverage"],
                       random_pair_byte_offset=qid*int(cfg["random_pairs_per_query"])*8)
            scalar.write(json.dumps(row, allow_nan=False)+"\n")
            for pair in pairs:
                pair.update(query_id=qid, ef_search=ef)
                harmful.write(json.dumps(pair, allow_nan=False)+"\n")
            samples.write(indices.tobytes())
            mismatches += not row["native_set_equal"]
            if not row["native_order_equal_modulo_pq_ties"]:
                non_tied_mismatches.append(qid)
            count += 1
            candidates += n
            if count % 2000 == 0:
                print(f"ef={ef} measured_queries={count}", flush=True)
        assert next(previous, None) is None
    assert count == 10000 and candidates == manifest["total_candidates"]
    input_paths = [raw/"queries.jsonl", raw/"manifest.json", prior_path,
                   Path(config_path)] + sorted(raw.glob("candidate_scores_*.bin"))
    dump(out/"validation.json", {
        "query_count": count, "total_candidates": candidates,
        "phase3a_native_and_oracle_ids_match": True,
        "phase3a_candidate_counts_and_stored_samples_match": True,
        "native_set_mismatch_count": mismatches,
        "non_tied_native_mismatch_query_ids": non_tied_mismatches,
        "input_sha256": {str(p): digest(p) for p in input_paths},
        "analysis_source_sha256": digest(__file__),
        "metrics_source_sha256": digest(Path(__file__).with_name("phase3b_metrics.py")),
        "numpy_version": np.__version__,
        "random_pairs_schema": "little-endian int32 candidate-position pairs, query-major, 512 per query",
    })
    print(json.dumps({"ef": ef, "native_set_mismatches": mismatches,
                      "non_tied_mismatches": len(non_tied_mismatches)}), flush=True)
    if non_tied_mismatches:
        raise RuntimeError("H1 non-tied discrepancies require investigation before interpretation")


def summary(rows, ef):
    mean = lambda key: float(np.mean([r[key] for r in rows]))
    out = {"ef_search": ef, "queries": len(rows)}
    for key in ["oracle_recall", "pq_fixed_recall", "native_recall",
                "ranking_recall_loss", "ranking_disagreement", "topk_agreement",
                "candidate_count", "candidate_mae", "boundary_local_mae",
                "cross_boundary_inversion_count", "random_inversion_rate",
                "pq_candidate_coverage", "phase3a_exact_native_recall"]:
        out["mean_"+key] = mean(key)
    for key in ["native_set_equal", "native_list_equal", "native_order_equal_modulo_pq_ties",
                "pq_topk_score_tie", "exact_boundary_tie", "pq_boundary_tie"]:
        out[key+"_count"] = sum(r[key] for r in rows)
        out[key+"_fraction"] = mean(key)
    for category in CLASSES:
        out[category+"_count"] = sum(r["class"] == category for r in rows)
        out[category+"_fraction"] = out[category+"_count"] / len(rows)
    for key in ("ranking_recall_loss", "ranking_disagreement"):
        for name, q in (("median",.5),("p90",.9),("p95",.95),("p99",.99)):
            out[name+"_"+key] = float(np.quantile([r[key] for r in rows], q))
    clean = [r for r in rows if not r["exact_boundary_tie"]]
    out["no_exact_boundary_ties_queries"] = len(clean)
    out["no_exact_boundary_ties_mean_loss"] = float(np.mean([r["ranking_recall_loss"] for r in clean]))
    out["benign_given_disagreement"] = out["Benign_count"] / max(1,len(rows)-out["Stable_count"])
    benign = [r for r in rows if r["class"]=="Benign"]
    harmful = [r for r in rows if r["class"]=="Harmful"]
    exchanges = [r for r in benign if set(r["displaced_ids"]) & set(r["ground_truth_ids"])]
    out["benign_queries_with_gt_exchange"] = len(exchanges)
    out["benign_gt_exchanges_at_exact_boundary_ties"] = sum(r["exact_boundary_tie"] for r in exchanges)
    out["harmful_queries_with_adjacent_boundary_overtake"] = sum(r["boundary_overtaken"] for r in harmful)
    for label in ("displaced", "intruder"):
        out["harmful_total_gt_"+label] = sum(len(set(r[label+"_ids"]) & set(r["ground_truth_ids"])) for r in harmful)
    return out


def binned(rows, feature, bins):
    finite = [r for r in rows if r[feature] is not None]
    missing = [r for r in rows if r[feature] is None]
    xs = np.array([r[feature] for r in finite])
    edges = np.unique(np.quantile(xs, np.linspace(0,1,bins+1)))
    labels = np.searchsorted(edges[1:-1],xs,side="right")
    result = []
    for label in np.unique(labels):
        members = [r for r,l in zip(finite,labels) if l == label]
        result.append(bin_row(members,feature,int(label)))
    if missing:
        result.append(bin_row(missing,feature,"zero_margin"))
    return result


def bin_row(members, feature, label):
    xs = [r[feature] for r in members if r[feature] is not None]
    losses = [r["ranking_recall_loss"] for r in members]
    return {"feature": feature, "bin": label, "count": len(members),
            "min_x": min(xs) if xs else None, "max_x": max(xs) if xs else None,
            "mean_x": float(np.mean(xs)) if xs else None,
            "mean_loss": float(np.mean(losses)),
            "query_standard_error": float(np.std(losses,ddof=1)/math.sqrt(len(losses))) if len(losses)>1 else 0.,
            "mean_disagreement": float(np.mean([r["ranking_disagreement"] for r in members])),
            "harmful_fraction": sum(r["class"]=="Harmful" for r in members)/len(members)}


def analyze(config, analysis_root, tables, figures):
    cfg = read_conf(config)
    tables.mkdir(parents=True,exist_ok=True)
    figures.mkdir(parents=True,exist_ok=True)
    collections = {}
    for ef in (int(cfg["primary_ef"]), int(cfg["replication_ef"])):
        path = analysis_root/f"ef{ef}"/"queries.jsonl"
        if path.exists():
            validation = json.loads((path.parent/"validation.json").read_text())
            assert not validation["non_tied_native_mismatch_query_ids"]
            collections[ef] = [json.loads(line) for line in path.open()]
    assert int(cfg["primary_ef"]) in collections
    summaries = [summary(rows,ef) for ef,rows in collections.items()]
    correlations, bins, thresholds, classes, pair_summaries = [], [], [], [], []
    for ef, rows in collections.items():
        for category in CLASSES:
            members = [r for r in rows if r["class"] == category]
            record = {"ef_search":ef, "class":category, "queries":len(members)}
            for feature in FEATURES + ["oracle_recall", "ranking_recall_loss"]:
                values = [r[feature] for r in members if r[feature] is not None]
                record["mean_"+feature] = float(np.mean(values)) if values else None
            classes.append(record)
        pairs = [json.loads(line) for line in
                 (analysis_root/f"ef{ef}"/"harmful_pairs.jsonl").open()]
        pair_record = {"ef_search":ef,"pair_count":len(pairs)}
        for feature in ("exact_margin", "error_difference", "violation"):
            values = [r[feature] for r in pairs]
            for label,quantile in (("min",0),("median",.5),("p95",.95),("max",1)):
                pair_record[label+"_"+feature] = float(np.quantile(values,quantile)) if values else None
        pair_record["strict_inversion_count"] = sum(r["strict_inversion"] for r in pairs)
        pair_record["nonpositive_violation_count"] = sum(r["violation"]<=0 for r in pairs)
        pair_summaries.append(pair_record)
        for subset in ("all", "no_exact_boundary_ties"):
            source = rows if subset=="all" else [r for r in rows if not r["exact_boundary_tie"]]
            for feature in FEATURES:
                finite = [r for r in source if r[feature] is not None]
                for outcome in ("ranking_recall_loss", "ranking_disagreement"):
                    value = spearman([r[feature] for r in finite],[r[outcome] for r in finite])
                    correlations.append({"ef_search":ef,"subset":subset,"feature":feature,
                                         "outcome":outcome,"queries":len(finite),"spearman":value})
        for feature in FEATURES:
            br = binned(rows,feature,int(cfg["quantile_bins"]))
            bins.extend(dict(ef_search=ef,**r) for r in br)
            xs = [r[feature] for r in rows if r[feature] is not None]
            for cutoff in np.unique(np.quantile(xs,np.linspace(.1,.9,9))):
                direction = "le" if feature in ("relative_boundary_margin","topk_agreement") else "ge"
                selected = [r for r in rows if r[feature] is not None and
                            (r[feature] <= cutoff if direction=="le" else r[feature] >= cutoff)]
                thresholds.append({"ef_search":ef,"feature":feature,"direction":direction,
                                   "threshold":float(cutoff),"query_fraction":len(selected)/len(rows),
                                   **{k:v for k,v in bin_row(selected,feature,0).items()
                                      if k in ("count","mean_loss","harmful_fraction")}})
    for name, records in (("summary",summaries),("spearman",correlations),
                           ("bins",bins),("thresholds",thresholds),
                           ("classes",classes),("harmful_pairs",pair_summaries)):
        dump(tables/f"phase3b_{name}.json",records)
        write_csv(tables/f"phase3b_{name}.csv",records)
    plot(collections,summaries,bins,thresholds,figures)
    print(json.dumps(summaries,indent=2))


def plot(collections, summaries, bins, thresholds, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"svg.hashsalt":"phase3b", "font.size":10})
    def save(fig,name):
        fig.tight_layout()
        fig.savefig(destination/f"phase3b_{name}.svg",metadata={"Date":None})
        plt.close(fig)
    labels = {"relative_boundary_margin":"Relative exact top-10 boundary margin",
              "error_scale_over_boundary_margin":"Boundary-local MAE / exact boundary margin",
              "cross_boundary_inversion_count":"Strict cross-boundary inversion count",
              "candidate_mae":"Mean absolute PQ error over all candidates (squared L2)",
              "boundary_local_mae":"Boundary-local mean absolute PQ error (squared L2)",
              "candidate_count":"Number of L0 candidates"}
    for feature,xlabel in labels.items():
        fig, ax = plt.subplots(figsize=(6.5,4.3))
        for ef in collections:
            records = [r for r in bins if r["ef_search"]==ef and r["feature"]==feature and r["mean_x"] is not None]
            ax.errorbar([r["mean_x"] for r in records],[r["mean_loss"] for r in records],
                        yerr=[r["query_standard_error"] for r in records], marker="o",label=f"efSearch={ef}")
        ax.set(xlabel=xlabel,ylabel="Mean ranking recall loss",title="Quantile bins; error bars: query SE")
        if feature=="error_scale_over_boundary_margin":
            ax.set_xscale("log")
            ax.text(.02,.97,"Zero-margin bin reported separately in tables",transform=ax.transAxes,va="top",fontsize=8)
        ax.legend(); ax.grid(alpha=.2); save(fig,feature)
    fig,ax=plt.subplots(figsize=(6.5,4.3))
    for ef,rows in collections.items():
        x=np.sort([r["topk_agreement"] for r in rows])
        values,counts=np.unique(x,return_counts=True)
        ax.step(values,np.cumsum(counts)/len(x),where="post",label=f"efSearch={ef}")
    ax.set(xlabel="Exact/PQ top-10 membership agreement",ylabel="Empirical CDF",ylim=(0,1.02)); ax.legend(); save(fig,"topk_agreement_cdf")
    fig,ax=plt.subplots(figsize=(6.5,4.3))
    width=.8/len(summaries)
    for j,row in enumerate(summaries):
        ax.bar(np.arange(4)+j*width,[row[c+"_fraction"] for c in CLASSES],width,label=f"efSearch={row['ef_search']}")
    ax.set_xticks(np.arange(4)+width*(len(summaries)-1)/2,CLASSES)
    ax.set(ylabel="Fraction of queries"); ax.legend(); save(fig,"classes")
    fig,axs=plt.subplots(1,3,figsize=(12,3.8))
    efs=[r["ef_search"] for r in summaries]
    for field,label in (("mean_oracle_recall","Exact candidate oracle"),("mean_pq_fixed_recall","Global PQ top-10"),("mean_phase3a_exact_native_recall","Exact HNSW (Phase 3A)")):
        axs[0].plot(efs,[r[field] for r in summaries],"o-",label=label)
    axs[0].set(ylabel="Mean Recall@10"); axs[0].legend(fontsize=7)
    for field,label in (("mean_ranking_recall_loss","Recall loss"),("mean_ranking_disagreement","Membership disagreement")):
        axs[1].plot(efs,[r[field] for r in summaries],"o-",label=label)
    axs[1].legend(fontsize=7)
    axs[2].plot(efs,[r["mean_candidate_count"] for r in summaries],"o-")
    axs[2].set(ylabel="Mean L0 candidate count")
    for ax in axs: ax.set(xlabel="efSearch",xticks=efs); ax.grid(alpha=.2)
    save(fig,"efsearch_comparison")
    fig,ax=plt.subplots(figsize=(6.5,4.3))
    for ef in collections:
        records=[r for r in bins if r["ef_search"]==ef and r["feature"]=="candidate_count"]
        ax.plot([r["mean_x"] for r in records],[r["mean_disagreement"] for r in records],"o-",label=f"efSearch={ef}")
    ax.set(xlabel="L0 candidate count",ylabel="Mean membership disagreement"); ax.legend(); save(fig,"candidate_count_disagreement")
    fig,ax=plt.subplots(figsize=(6.5,4.3))
    primary=min(collections)
    for feature,label in (("relative_boundary_margin","Small boundary margin"),
                           ("error_scale_over_boundary_margin","Large local error/margin"),
                           ("cross_boundary_inversion_count","Many cross-boundary inversions"),
                           ("candidate_mae","Large global candidate MAE")):
        records=sorted([r for r in thresholds if r["ef_search"]==primary and r["feature"]==feature],key=lambda r:r["query_fraction"])
        ax.plot([r["query_fraction"] for r in records],[r["harmful_fraction"] for r in records],"o-",label=label)
    ax.set(xlabel="Fraction of queries selected by threshold",ylabel="Harmful fraction among selected queries",title=f"Descriptive threshold curves, efSearch={primary}")
    ax.legend(fontsize=8);ax.grid(alpha=.2);save(fig,"threshold_curves")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("derive")
    for arg in ("config","raw_root","output_root"): p.add_argument(arg,type=Path)
    p.add_argument("ef",type=int)
    p=sub.add_parser("summarize")
    for arg in ("config","analysis_root","tables","figures"): p.add_argument(arg,type=Path)
    args=parser.parse_args()
    if args.command=="derive": derive(args.config,args.raw_root,args.output_root,args.ef)
    else: analyze(args.config,args.analysis_root,args.tables,args.figures)
