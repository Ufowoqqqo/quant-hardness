#!/usr/bin/env python3
"""Score-only seed stability: preparation, raw derivation, summary, diagnostic."""
import argparse
import hashlib
import itertools
import json
import platform
import shutil
import subprocess
from pathlib import Path
import numpy as np
from analyze_phase3a_sift1m import read_conf, spearman, write_csv
from analyze_phase3b_fixed_candidate_ranking import digest, dump
from analyze_phase3c_precision_transition import jsonl, read_candidates, gz_writer, stats
from phase3d_metrics import measure_seeds, single, severity_overlap, GEOMETRY, CATEGORIES, PAIR_CLASSES


def hashes(paths):
    return {str(p): digest(p) for p in paths}


def verify_hashes(values):
    for path, expected in values.items():
        assert digest(path) == expected, f"input hash changed: {path}"


def prepare(config, run):
    cfg = read_conf(config); source = Path(cfg["candidate_source"])
    assert (run/"preregistration.md").exists()
    assert not (run/"input_provenance.json").exists()
    # Check every source against the previously validated Phase 3C derivation.
    validation = json.loads((Path(cfg["phase3c_analysis"])/"validation.json").read_text())
    verify_hashes(validation["input_sha256"])
    prior = read_conf(Path(cfg["phase3a_config"]))
    graph = Path(prior["cache_directory"])/"hnsw_fp32.index"
    paths = sorted(source.glob("*"))+[config, run/"preregistration.md", graph,
        Path(cfg["phase3c_analysis"])/"queries.jsonl",
        Path(cfg["phase3c_analysis"])/"random_pair_indices.bin"]
    paths = [p for p in paths if p.is_file()]
    total = 0
    with (run/"candidate_requests.tsv").open("x") as out:
        for q, a in read_candidates(source, jsonl(source/"queries.jsonl")):
            assert q["query_id"] == total; total += 1
            assert hashlib.sha256(a["id"].astype("<i8").tobytes()).hexdigest() == q["candidate_order_sha256"]
            out.write(f'{q["query_id"]}\t{q["candidate_file"]}\t{q["byte_offset"]}\t{q["candidate_count"]}\t{q["candidate_order_sha256"]}\n')
    assert total == 10000
    paths.append(run/"candidate_requests.tsv")
    shutil.copyfile(config, run/"resolved_config.conf")
    dump(run/"input_provenance.json", {"input_sha256": hashes(paths),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], text=True),
        "machine": platform.uname()._asdict(), "numpy": np.__version__,
        "candidate_source": str(source), "queries": total})
    print("prepare source identity=PASS", flush=True)


def derive(config, run, out):
    cfg = read_conf(config); source = Path(cfg["candidate_source"])
    verify_hashes(json.loads((run/"input_provenance.json").read_text())["input_sha256"])
    manifests = [json.loads((run/f"seed_{s}"/"manifest.json").read_text()) for s in range(5)]
    for key in ("training_ids_sha256", "training_vectors_sha256", "requests_sha256", "graph_fingerprint"):
        assert len({m[key] for m in manifests}) == 1, key
    assert len({m["codebook_sha256"] for m in manifests}) == 5
    score_maps = [np.memmap(run/f"seed_{s}"/"scores.f32", dtype="<f4", mode="r") for s in range(5)]
    for s, m in enumerate(manifests):
        assert digest(run/f"seed_{s}"/"scores.f32") == m["scores_sha256"]
        assert digest(run/f"seed_{s}"/"pq64.index") == m["model_file_sha256"]
        assert digest(run/f"seed_{s}"/"training_ids.i32") == m["training_ids_sha256"]
    out.mkdir(parents=True, exist_ok=False)
    previous = iter(jsonl(Path(cfg["phase3c_analysis"])/"queries.jsonl"))
    old_random = np.memmap(Path(cfg["phase3c_analysis"])/"random_pair_indices.bin", dtype="<i4", mode="r").reshape(10000, -1, 2)
    offset = 0; pair_hist = np.zeros(6, dtype=np.int64)
    pair_inter = np.zeros((5,5), dtype=np.int64); pair_union = pair_inter.copy()
    with gz_writer(out/"queries.jsonl.gz") as scalar, gz_writer(out/"replacement_pairs.jsonl.gz") as repl, \
            gz_writer(out/"inversion_pairs.jsonl.gz") as pairs:
        for meta, a in read_candidates(source, jsonl(source/"queries.jsonl")):
            qid = meta["query_id"]; n = len(a); prior = next(previous)
            assert prior["query_id"] == qid
            score = np.array([p[offset:offset+n] for p in score_maps]); offset += n
            assert np.array_equal(score[0], a["pq64"])
            row, replacements, union, random = measure_seeds(a["id"], a["exact"], score, meta["ground_truth_ids"],
                int(cfg["k"]), int(cfg["random_pair_seed"])+qid, int(cfg["random_pairs_per_query"]), float(cfg["epsilon"]))
            assert np.array_equal(random, old_random[qid])
            for key, value in row["models"][0].items():
                if key+"_64" in prior: assert value == prior[key+"_64"], (qid, key)
            for key in GEOMETRY+["exact_topk_ids", "exact_topk_set", "exact_boundary_tie"]:
                assert row[key] == prior[key], (qid, key)
            row.update(query_id=qid, candidate_order_sha256=meta["candidate_order_sha256"],
                       ground_truth_ids=meta["ground_truth_ids"], score_offset=offset-n)
            scalar.write(json.dumps(row, allow_nan=False)+"\n")
            for p in replacements:
                p["query_id"] = qid; repl.write(json.dumps(p, allow_nan=False)+"\n")
            for p in union:
                p["query_id"] = qid; pairs.write(json.dumps(p, allow_nan=False)+"\n")
                pair_hist[p["frequency_count"]] += 1
                mask = np.array(p["mask"])
                pair_inter += mask[:,None] & mask[None,:]
                pair_union += mask[:,None] | mask[None,:]
            if (qid+1)%1000 == 0: print(f"derive queries={qid+1}", flush=True)
    assert next(previous, None) is None and all(len(p) == offset for p in score_maps)
    dump(out/"pair_summary.json", {"frequency_counts": pair_hist.tolist(),
        "intersection": pair_inter.tolist(), "union": pair_union.tolist()})
    verify_hashes(json.loads((run/"input_provenance.json").read_text())["input_sha256"])
    dump(out/"validation.json", {"queries": 10000, "candidates": offset, "seed0_scores_metrics_identity": True,
        "candidate_exact_gt_identity": True, "random_pairs_identity": True,
        "training_sample_identity": True, "distinct_codebooks": True, "source_unchanged": True,
        "analysis_sha256": digest(__file__), "metrics_sha256": digest(Path(__file__).with_name("phase3d_metrics.py")),
        "numpy": np.__version__})
    print("derive status=PASS", flush=True)


def variability(v):
    v = np.asarray(v, dtype=float)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v)>1 else 0.0,
            "min": float(v.min()), "max": float(v.max()),
            "coefficient_of_variation": float(v.std(ddof=1)/abs(v.mean())) if len(v)>1 and v.mean()!=0 else None}


def summarize(config, run, derived, tables, figures):
    cfg = read_conf(config); rows = list(jsonl(derived/"queries.jsonl.gz")); n = len(rows)
    tables.mkdir(parents=True, exist_ok=True); figures.mkdir(parents=True, exist_ok=True)
    def table(name, records):
        write_csv(tables/f"phase3d_{name}.csv", records); dump(tables/f"phase3d_{name}.json", records)
    fields = ["fixed_candidate_recall", "ranking_recall_loss", "cross_boundary_inversion_count",
              "global_mae", "random_pair_inversion_rate", "topk_agreement", "boundary_window_mae"]
    aggregate = [{"model":s, **{f:float(np.mean([r["models"][s][f] for r in rows])) for f in fields},
                  "candidate_weighted_mae":float(sum(r["models"][s]["global_mae"]*r["candidate_count"] for r in rows)/sum(r["candidate_count"] for r in rows))} for s in range(5)]
    table("models", aggregate)
    table("aggregate_variability", [{"metric":f, **variability([r[f] for r in aggregate])} for f in fields+["candidate_weighted_mae"]])
    losses = np.array([[r["models"][s]["ranking_recall_loss"] for r in rows] for s in range(5)])
    inv = np.array([[r["models"][s]["cross_boundary_inversion_count"] for r in rows] for s in range(5)])
    harmful = losses>0; prevalence = harmful.mean(axis=1)
    pair_data = json.loads((derived/"pair_summary.json").read_text())
    jaccard = np.eye(5); correlations = np.eye(5); pair_rows = []; overlaps = []
    for a,b in itertools.combinations(range(5),2):
        intersection = int((harmful[a]&harmful[b]).sum()); union = int((harmful[a]|harmful[b]).sum())
        jaccard[a,b] = jaccard[b,a] = intersection/union if union else 1
        correlations[a,b] = correlations[b,a] = spearman(losses[a].tolist(),losses[b].tolist())
        expected = prevalence[a]*prevalence[b]
        pair_rows.append({"model_a":a,"model_b":b,"harmful_jaccard":jaccard[a,b],
            "loss_spearman":correlations[a,b], "inversion_spearman":spearman(inv[a].tolist(),inv[b].tolist()),
            "independent_harmful_jaccard":float(expected/(prevalence[a]+prevalence[b]-expected)),
            "pair_inversion_jaccard":pair_data["intersection"][a][b]/pair_data["union"][a][b]})
        for fraction in (.05,.1,.2):
            overlaps.append({"model_a":a,"model_b":b, **severity_overlap(losses[a],losses[b],fraction)})
    table("pairwise",pair_rows); table("severity_overlap",overlaps)
    signed_total = float(losses.sum()); positive_total = float(np.maximum(losses,0).sum())
    categories = []
    for name in CATEGORIES:
        members = [r for r in rows if r["category"]==name]
        loss_sum = sum(sum(m["ranking_recall_loss"] for m in r["models"]) for r in members)
        categories.append({"category":name,"count":len(members),"fraction":len(members)/n,
                           "signed_loss_sum":loss_sum,"fraction_total_signed_loss":loss_sum/signed_total,
                           "mean_oracle_recall":float(np.mean([r["oracle_recall"] for r in members])) if members else None})
    table("query_categories",categories)
    freq = [{"harmful_count":i,"frequency":i/5,"queries":sum(r["harmful_count"]==i for r in rows)} for i in range(6)]
    table("harmful_frequency",freq)
    pair_freq = [{"seed_count":i,"frequency":i/5,"pairs":pair_data["frequency_counts"][i],
                  "fraction_union_pairs":pair_data["frequency_counts"][i]/sum(pair_data["frequency_counts"])} for i in range(1,6)]
    table("inversion_frequency",pair_freq)
    pair_query=[]
    for name in CATEGORIES:
        members=[r for r in rows if r["category"]==name]
        for f in range(1,6):
            pair_query.append({"query_category":name,"frequency_count":f,"queries":len(members),
                "queries_with_pairs":sum(r["inversion_pair_frequency_counts"][f-1]>0 for r in members),
                "pair_count":sum(r["inversion_pair_frequency_counts"][f-1] for r in members)})
    table("pair_frequency_by_query_category",pair_query)
    attribution = [{"bucket":key,"signed_loss_sum":sum(m["loss_attribution"][key] for r in rows for m in r["models"])}
                   for key in PAIR_CLASSES+["unassigned_tie_or_no_strict_overtake","gt_intruder_credit"]]
    for r in attribution:r["fraction_total_signed_loss"] = r["signed_loss_sum"]/signed_total
    assert abs(sum(r["signed_loss_sum"] for r in attribution)-signed_total)<1e-7
    table("descriptive_pair_loss_allocation",attribution)
    strata = []
    for flag in (False,True):
        members = [m for r in rows for m in r["models"] if m["has_high_frequency_inversion"]==flag]
        strata.append({"has_high_frequency_inversion":flag,"query_models":len(members),
            "mean_loss":float(np.mean([m["ranking_recall_loss"] for m in members])) if members else None,
            "fraction_signed_loss":sum(m["ranking_recall_loss"] for m in members)/signed_total})
    table("high_frequency_pair_presence",strata)
    geometry, geo_corr, bins = [], [], []
    for stratum in ("all","oracle1"):
        subset = rows if stratum=="all" else [r for r in rows if r["oracle_recall"]==1]
        for descriptor in GEOMETRY:
            geo_corr.append({"stratum":stratum,"descriptor":descriptor,"queries":len(subset),
                "spearman_harmful_frequency":spearman([r[descriptor] for r in subset],[r["harmful_frequency"] for r in subset])})
            for name in CATEGORIES:
                values=[r[descriptor] for r in subset if r["category"]==name]
                geometry.append({"stratum":stratum,"category":name,"descriptor":descriptor,"queries":len(values),**stats(values)})
            # Quantile edges, with exact ties kept together (possibly <10 bins).
            xs=np.array([r[descriptor] for r in subset])
            edges=np.unique(np.quantile(xs,np.linspace(0,1,int(cfg["quantile_bins"])+1)))
            labels=np.searchsorted(edges[1:-1],xs,side="right")
            for b in sorted(set(labels.tolist())):
                members=[r for r,label in zip(subset,labels) if label==b]
                bins.append({"stratum":stratum,"descriptor":descriptor,"bin":b,"queries":len(members),
                    "descriptor_mean":float(np.mean([r[descriptor] for r in members])),
                    "harmful_frequency_mean":float(np.mean([r["harmful_frequency"] for r in members]))})
    table("geometry",geometry); table("geometry_correlations",geo_corr); table("geometry_bins",bins)
    median_j=float(np.median([r["harmful_jaccard"] for r in pair_rows]))
    median_r=float(np.median([r["loss_spearman"] for r in pair_rows]))
    summary={"queries":n,"models":5,"oracle_recall":float(np.mean([r["oracle_recall"] for r in rows])),
        "aggregate":aggregate,"categories":categories,"pairwise":pair_rows,
        "median_pairwise_jaccard":median_j,"median_pairwise_loss_spearman":median_r,
        "mean_pairwise_loss_spearman":float(np.mean([r["loss_spearman"] for r in pair_rows])),
        "h2_operational_pass":median_j>=float(cfg["h2_min_median_jaccard"]) and median_r>=float(cfg["h2_min_median_spearman"]),
        "harmful_prevalence_by_model":prevalence.tolist(),
        "independent_expected_all_harmful_queries":float(n*np.prod(prevalence)),
        "independent_expected_robust_queries":float(n*np.prod(1-prevalence)),
        "signed_loss_sum":signed_total,"positive_loss_sum":positive_total,
        "negative_query_model_losses":int((losses<0).sum()),
        "pair_frequency":pair_freq,"pair_loss_allocation":attribution,
        "mean_query_std_ranking_loss":float(np.mean([r["std_ranking_loss"] for r in rows])),
        "oracle1_queries":sum(r["oracle_recall"]==1 for r in rows),
        "exact_boundary_tie_queries":sum(r["exact_boundary_tie"] for r in rows),
        "pq_boundary_ties_by_model":[sum(r["models"][s]["pq_boundary_tie"] for r in rows) for s in range(5)]}
    # Predeclared oracle1 comparison; exact-tie exclusion sensitivity is labeled.
    sensitivity=[]
    for stratum,subset in (("oracle1",[r for r in rows if r["oracle_recall"]==1]),
                           ("no_exact_boundary_ties",[r for r in rows if not r["exact_boundary_tie"]])):
        for a,b in itertools.combinations(range(5),2):
            x=[r["models"][a]["ranking_recall_loss"] for r in subset];y=[r["models"][b]["ranking_recall_loss"] for r in subset]
            aa=np.array(x)>0;bb=np.array(y)>0
            sensitivity.append({"stratum":stratum,"queries":len(subset),"a":a,"b":b,
                "harmful_jaccard":float((aa&bb).sum()/(aa|bb).sum()),"loss_spearman":spearman(x,y)})
    table("stability_sensitivity",sensitivity)
    dump(tables/"phase3d_summary.json",summary)
    plots(rows,freq,pair_freq,jaccard,correlations,bins,figures)
    print(json.dumps({k:summary[k] for k in ("median_pairwise_jaccard","median_pairwise_loss_spearman","h2_operational_pass")}))


def savefig(fig,path):
    fig.savefig(path,metadata={"Date":None})
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines())+"\n")


def plots(rows,freq,pair_freq,jaccard,correlations,bins,out):
    import matplotlib
    matplotlib.use("Agg");matplotlib.rcParams["svg.hashsalt"]="phase3d"
    import matplotlib.pyplot as plt
    for name,values,xlabel in (("harmful_frequency",freq,"Fraction of models harmful"),("pair_frequency",pair_freq,"Fraction of models inverting pair")):
        fig,ax=plt.subplots(figsize=(6,4));ax.bar([r["frequency"] for r in values],[r.get("queries",r.get("pairs")) for r in values],width=.13)
        ax.set(xlabel=xlabel,ylabel="Query count" if name=="harmful_frequency" else "Union pair count")
        fig.tight_layout();savefig(fig,out/f"phase3d_{name}.svg");plt.close(fig)
    for name,matrix in (("harmful_jaccard",jaccard),("loss_spearman",correlations)):
        fig,ax=plt.subplots(figsize=(5,4));im=ax.imshow(matrix,vmin=0,vmax=1);fig.colorbar(im,ax=ax)
        for a in range(5):
            for b in range(5):ax.text(b,a,f"{matrix[a,b]:.2f}",ha="center",va="center",color="white" if matrix[a,b]<.6 else "black")
        ax.set(xlabel="PQ model",ylabel="PQ model");fig.tight_layout();savefig(fig,out/f"phase3d_{name}.svg");plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,field in zip(axes,("relative_boundary_margin","boundary_gap_r5","local_distance_concentration")):
        for stratum in ("all","oracle1"):
            points=[r for r in bins if r["stratum"]==stratum and r["descriptor"]==field]
            ax.plot([r["descriptor_mean"] for r in points],[r["harmful_frequency_mean"] for r in points],"o-",label=stratum)
        ax.set(xlabel=field,ylabel="Mean harmful frequency");ax.legend()
    fig.tight_layout();savefig(fig,out/"phase3d_geometry.svg");plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4));h=ax.hexbin([r["mean_ranking_loss"] for r in rows],[r["std_ranking_loss"] for r in rows],gridsize=25,mincnt=1)
    fig.colorbar(h,ax=ax,label="Queries");ax.set(xlabel="Mean ranking loss across seeds",ylabel="Sample SD across seeds")
    fig.tight_layout();savefig(fig,out/"phase3d_loss_mean_std.svg");plt.close(fig)


def ensemble(config,run,out,tables,figures):
    cfg=read_conf(config); source=Path(cfg["candidate_source"])
    checkpoint=run/"primary_checkpoint"/"phase3d_summary.json"
    assert checkpoint.exists(),"complete primary analysis before diagnostic"
    maps=[np.memmap(run/f"seed_{s}"/"scores.f32",dtype="<f4",mode="r") for s in range(5)]
    out.mkdir(parents=True,exist_ok=False);offset=0;rows=[]
    with (out/"queries.jsonl").open("x") as target:
        for meta,a in read_candidates(source,jsonl(source/"queries.jsonl")):
            qid=meta["query_id"];n=len(a)
            score=np.array([p[offset:offset+n] for p in maps],dtype=np.float64).mean(axis=0);offset+=n
            m,g,_,_=single(a["id"],a["exact"],score,meta["ground_truth_ids"],10,
                          int(cfg["random_pair_seed"])+qid,int(cfg["random_pairs_per_query"]),float(cfg["epsilon"]))
            row={"query_id":qid,"candidate_order_sha256":meta["candidate_order_sha256"],**g,**m}
            target.write(json.dumps(row,allow_nan=False)+"\n");rows.append(row)
    summary={f:float(np.mean([r[f] for r in rows])) for f in ("fixed_candidate_recall","ranking_recall_loss","cross_boundary_inversion_count")}
    summary.update(queries=len(rows),uniform_weights=[.2]*5,accumulation="float64",primary_checkpoint_sha256=digest(checkpoint))
    dump(out/"summary.json",summary);dump(tables/"phase3d_ensemble.json",summary)
    primary=json.loads(checkpoint.read_text());records=[{"condition":f"seed_{r['model']}",**{f:r[f] for f in ("fixed_candidate_recall","ranking_recall_loss","cross_boundary_inversion_count")}} for r in primary["aggregate"]]
    records.append({"condition":"ensemble diagnostic",**{f:summary[f] for f in ("fixed_candidate_recall","ranking_recall_loss","cross_boundary_inversion_count")}})
    write_csv(tables/"phase3d_ensemble.csv",records)
    import matplotlib
    matplotlib.use("Agg");matplotlib.rcParams["svg.hashsalt"]="phase3d"
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,metric in zip(axes,("fixed_candidate_recall","ranking_recall_loss","cross_boundary_inversion_count")):
        ax.bar(range(6),[r[metric] for r in records]);ax.set_xticks(range(6),["0","1","2","3","4","Avg"]);ax.set(xlabel="PQ model / diagnostic",ylabel=metric)
    fig.tight_layout();savefig(fig,figures/"phase3d_ensemble.svg");plt.close(fig)
    print(json.dumps(summary),flush=True)


def checkpoint(run,tables):
    summary=json.loads((tables/"phase3d_summary.json").read_text())
    assert summary["queries"]==10000 and not (run/"ensemble").exists()
    out=run/"primary_checkpoint";out.mkdir(exist_ok=False)
    for file in sorted(tables.glob("phase3d_*")):shutil.copyfile(file,out/file.name)
    dump(out/"checkpoint.json",{"stage":"primary complete before ensemble diagnostic",
        "summary_sha256":digest(out/"phase3d_summary.json"),
        "raw_validation_sha256":digest(run/"analysis"/"validation.json")})


def main():
    p=argparse.ArgumentParser();p.add_argument("mode",choices=["prepare","derive","summarize","checkpoint","ensemble"])
    p.add_argument("--config",type=Path,default=Path("configs/indexes/phase3d_quantizer_seed_stability.conf"))
    p.add_argument("--run",type=Path,default=Path("runs/phase3d_quantizer_seed_stability_v1"))
    p.add_argument("--derived",type=Path);p.add_argument("--tables",type=Path,default=Path("results/tables"));p.add_argument("--figures",type=Path,default=Path("results/figures"))
    a=p.parse_args();derived=a.derived or a.run/"analysis"
    if a.mode=="prepare":prepare(a.config,a.run)
    elif a.mode=="derive":derive(a.config,a.run,derived)
    elif a.mode=="summarize":summarize(a.config,a.run,derived,a.tables,a.figures)
    elif a.mode=="checkpoint":checkpoint(a.run,a.tables)
    else:ensemble(a.config,a.run,derived if a.derived else a.run/"ensemble",a.tables,a.figures)


if __name__=="__main__":main()
