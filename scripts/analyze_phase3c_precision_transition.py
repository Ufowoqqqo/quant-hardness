#!/usr/bin/env python3
"""Primary exact-pool and separately labeled recorded-PQ-pool precision analysis."""
import argparse
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import numpy as np
from phase3c_metrics import measure
from analyze_phase3a_sift1m import read_conf, spearman, write_csv
from analyze_phase3b_fixed_candidate_ranking import digest, dump

DTYPE=np.dtype([("id","<i4"),("exact","<f4"),("pq32","<f4"),("pq64","<f4")])
OLD_DTYPE=np.dtype([("id","<i4"),("exact","<f4"),("pq","<f4")])
FLAGS=["stable_both","rescued_improved","persistent_harmful","regressed",
       "fully_rescued","partially_improved_persistent","unchanged_harmful"]
CHANGES=["global_mae","relative_mae","random_pair_inversion_rate",
         "boundary_window_mae","cross_boundary_inversion_count",
         "cross_boundary_inversion_fraction","maximum_violation","displaced_count"]
GEOMETRY=["boundary_margin","relative_boundary_margin","boundary_gap_r1",
          "boundary_gap_r2","boundary_gap_r5","d1","dk","dk_plus_1","d20",
          "local_distance_concentration","candidate_count"]


def jsonl(path):
    opener=gzip.open if str(path).endswith(".gz") else open
    with opener(path,"rt") as stream:
        for line in stream:yield json.loads(line)


def gz_writer(path):
    return io.TextIOWrapper(gzip.GzipFile(filename="",mode="wb",fileobj=path.open("xb"),mtime=0),encoding="utf-8")


def read_candidates(directory,metadata,dtype=DTYPE):
    maps={p.name:np.memmap(p,dtype=dtype,mode="r") for p in directory.glob("candidate_scores_*.bin")}
    for meta in metadata:
        assert meta["byte_offset"]%dtype.itemsize==0
        start=meta["byte_offset"]//dtype.itemsize
        array=maps[meta["candidate_file"]][start:start+meta["candidate_count"]]
        assert len(array)==meta["candidate_count"]
        yield meta,array


def prepare_control(config,run):
    cfg=read_conf(config)
    # Explicit sequencing gate: primary derivation and analysis checkpoint first.
    assert (run/"primary_checkpoint"/"phase3c_exact_pool_summary.json").exists()
    source=Path(cfg["phase3b_pq64_pool"])
    target=run/"control_candidates.bin"
    count=0
    with target.open("xb") as out:
        for q,a in read_candidates(source,jsonl(source/"queries.jsonl"),OLD_DTYPE):
            if q["query_id"]%int(cfg["control_query_stride"]):continue
            out.write(struct.pack("<ii",q["query_id"],len(a)))
            out.write(a["id"].astype("<i4").tobytes());count+=1
    dump(run/"control_input_provenance.json",{
        "queries":count,"sampling":"query_id modulo control_query_stride equals zero",
        "source":str(source),"input_sha256":{str(p):digest(p) for p in sorted(source.glob("*")) if p.is_file()},
        "control_candidates_sha256":digest(target)})
    print(f"control_input queries={count} status=PASS")


def derive(config,run,derived_root,condition):
    cfg=read_conf(config);raw=run/condition;out=derived_root/condition
    out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((raw/"manifest.json").read_text())
    assert manifest["graph_fingerprint"]==cfg["graph_fingerprint"]
    prior={r["query_id"]:r for r in jsonl(Path(cfg["phase3a_run"])/"decomposition_ef64.jsonl")}
    control=None
    if condition=="pq64_pool_control":
        source=Path(cfg["phase3b_pq64_pool"])
        control=iter(read_candidates(source,(m for m in jsonl(source/"queries.jsonl") if m["query_id"]%int(cfg["control_query_stride"])==0),OLD_DTYPE))
    n,total,native_tied=0,0,0
    with (out/"queries.jsonl").open("x") as result, gz_writer(out/"replacement_pairs.jsonl.gz") as replacements, \
            gz_writer(out/"inversion_transitions.jsonl.gz") as transitions, (out/"random_pair_indices.bin").open("xb") as random:
        for meta,a in read_candidates(raw,jsonl(raw/"queries.jsonl")):
            qid=meta["query_id"];old=prior[qid]
            assert qid==n*(int(cfg["control_query_stride"]) if control is not None else 1)
            pool_hash=hashlib.sha256(a["id"].astype("<i8").tobytes()).hexdigest()
            assert pool_hash==meta["candidate_order_sha256"]
            assert meta["ground_truth_ids"]==old["ground_truth_ids"]
            ids=a["id"].tolist();order=np.argsort(a["exact"],kind="stable")
            if control is None:
                assert ids and len(ids)==old["exact_L0_distance_evaluations"]
                if old["exact_L0_evaluated_ids"] is not None:assert sorted(ids)==old["exact_L0_evaluated_ids"]
                assert meta["exact_native_ids"]==old["exact_native_result_ids"]
                lookup=dict(zip(ids,a["exact"].tolist()))
                nd=[lookup[v] for v in meta["exact_native_ids"]]
                assert nd==meta["exact_native_distances"]
                assert nd==a["exact"][order[:10]].tolist(),"non-tied exact-control anomaly"
                delta_ids=set(ids[i] for i in order[:10])^set(meta["exact_native_ids"])
                assert all(lookup[v]==nd[-1] for v in delta_ids),"non-boundary exact tie mismatch"
                native_tied+=bool(delta_ids)
            else:
                previous,b=next(control)
                assert previous["query_id"]==qid and np.array_equal(a["id"],b["id"])
                assert np.array_equal(a["exact"],b["exact"]) and np.array_equal(a["pq64"],b["pq"])
            row,repl,pairs,samples=measure(a["id"],a["exact"],a["pq32"],a["pq64"],meta["ground_truth_ids"],
                int(cfg["k"]),int(cfg["random_pair_seed"])+qid,int(cfg["random_pairs_per_query"]),
                float(cfg["epsilon"]),float(cfg["relative_small_margin"]),float(cfg["relative_large_margin"]),
                int(cfg["near_boundary_inside_rank"]),int(cfg["near_boundary_outside_rank"]))
            row.update(query_id=qid,condition=condition,candidate_order_sha256=pool_hash,
                exact_candidate_hash=pool_hash,pq32_candidate_hash=pool_hash,pq64_candidate_hash=pool_hash,
                ground_truth_ids=meta["ground_truth_ids"],random_pair_byte_offset=n*int(cfg["random_pairs_per_query"])*8)
            if control is None:
                row["phase3a_exact_native_recall"]=old["recall_exact_native"]
                row["stable_oracle_minus_exact_native_recall"]=row["oracle_recall"]-old["recall_exact_native"]
            else:
                assert row["exact_topk_ids"]==old["pq_L0_oracle_result_ids"]
            result.write(json.dumps(row,allow_nan=False)+"\n")
            for p in repl:
                p.update(query_id=qid);replacements.write(json.dumps(p,allow_nan=False)+"\n")
            for p in pairs:
                p.update(query_id=qid);transitions.write(json.dumps(p,allow_nan=False)+"\n")
            random.write(samples.tobytes())
            n+=1;total+=len(ids)
            if n%2000==0:print(f"{condition}: {n} queries",flush=True)
    assert n==manifest["query_count"] and total==manifest["candidate_count"]
    if control is not None:assert next(control,None) is None
    input_paths=[raw/"manifest.json",raw/"queries.jsonl",config]+sorted(raw.glob("candidate_scores_*.bin"))
    dump(out/"validation.json",{"queries":n,"candidates":total,"shared_candidate_hashes_identical":True,
        "phase3a_comparison_passed":True,"exact_native_tied_membership_cases":native_tied,
        "control_matches_saved_pq64_pool_and_scores":control is not None,
        "input_sha256":{str(p):digest(p) for p in input_paths},
        "analysis_sha256":digest(__file__),"metrics_sha256":digest(Path(__file__).with_name("phase3c_metrics.py")),
        "numpy_version":np.__version__})
    print(f"derive {condition} queries={n} exact_native_tied_sets={native_tied} status=PASS",flush=True)


def stats(values):
    values=[v for v in values if v is not None]
    if not values:return {k:None for k in ("mean","median","p90","p95","p99","min","max")}
    return {"mean":float(np.mean(values)),**{name:float(np.quantile(values,q)) for name,q in
                (("median",.5),("p90",.9),("p95",.95),("p99",.99),("min",0),("max",1))}}


def overlap_summary(rows):
    a={r["query_id"] for r in rows if r["loss_hits_32"]>0}
    b={r["query_id"] for r in rows if r["loss_hits_64"]>0}
    output={"harmful32_count":len(a),"harmful64_count":len(b),"harmful_intersection":len(a&b),
            "p_harmful64_given32":len(a&b)/len(a) if a else None,
            "p_harmful32_given64":len(a&b)/len(b) if b else None,
            "harmful_jaccard":len(a&b)/len(a|b) if a|b else None}
    records=[]
    for fraction in (.1,.2,.5):
        n=math.ceil(fraction*len(rows)); sets={};inclusive={};cutoffs={};ties={}
        for Q in (32,64):
            ranked=sorted(rows,key=lambda r:(-r[f"loss_hits_{Q}"],r["query_id"]))
            cutoff=ranked[n-1][f"loss_hits_{Q}"]
            sets[Q]={r["query_id"] for r in ranked[:n]}
            inclusive[Q]={r["query_id"] for r in rows if r[f"loss_hits_{Q}"]>=cutoff}
            cutoffs[Q]=cutoff/10;ties[Q]=sum(r[f"loss_hits_{Q}"]==cutoff for r in rows)
        a,b=sets[32],sets[64];ia,ib=inclusive[32],inclusive[64]
        records.append({"top_fraction":fraction,"exact_count_per_set":n,"intersection_count":len(a&b),
            "overlap_fraction":len(a&b)/n,"jaccard":len(a&b)/len(a|b),
            "cutoff_loss_32":cutoffs[32],"cutoff_loss_64":cutoffs[64],
            "cutoff_ties_32":ties[32],"cutoff_ties_64":ties[64],
            "tie_inclusive_count_32":len(ia),"tie_inclusive_count_64":len(ib),
            "tie_inclusive_jaccard":len(ia&ib)/len(ia|ib)})
    return output,records


def summary(rows):
    output={"queries":len(rows),"oracle_recall":stats([r["oracle_recall"] for r in rows])["mean"]}
    for field in ("candidate_count","delta_loss_precision","delta_critical_inversions"):
        output[field]=stats([r[field] for r in rows])
    for Q in (32,64):
        output[f"pq{Q}"]={field:stats([r[f"{field}_{Q}"] for r in rows]) for field in
            CHANGES+["fixed_candidate_recall","ranking_recall_loss","topk_agreement","intruder_count",
                     "max_selected_intruders_overtaking_one_topk","max_outside_overtaking_one_topk"]}
    output["transition_flags"]={field:{"count":sum(r[field] for r in rows),"fraction":sum(r[field] for r in rows)/len(rows)} for field in FLAGS}
    output["improved_and_persistent"]=sum(r["rescued_improved"] and r["persistent_harmful"] for r in rows)
    output["regressed_and_persistent"]=sum(r["regressed"] and r["persistent_harmful"] for r in rows)
    output["outside_four_original_flags"]=sum(not any(r[f] for f in FLAGS[:4]) for r in rows)
    output["negative_loss_queries_32"]=sum(r["loss_hits_32"]<0 for r in rows)
    output["negative_loss_queries_64"]=sum(r["loss_hits_64"]<0 for r in rows)
    output["queries_with_fewer_inversions"]=sum(r["delta_critical_inversions"]>0 for r in rows)
    output["queries_with_more_inversions"]=sum(r["delta_critical_inversions"]<0 for r in rows)
    output["regressed_despite_lower_global_mae"]=sum(r["regressed"] and r["delta_global_mae"]>0 for r in rows)
    output["regressed_despite_fewer_inversions"]=sum(r["regressed"] and r["delta_critical_inversions"]>0 for r in rows)
    output["harmful_persistence"],_=overlap_summary(rows)
    output["metric_changes"]={metric:stats([r["delta_"+metric] for r in rows]) for metric in CHANGES}
    for state in ("corrected","persistent","new"):
        total=sum(r[state+"_inversion_count"] for r in rows)
        active=[r for r in rows if r[state+"_inversion_count"]]
        output[state+"_inversions"]={"count":total,"queries_with_state":len(active)}
        for suffix in ("small_margin","large_margin","near_rank_boundary"):
            output[state+"_inversions"][suffix+"_pair_fraction"]=sum(r[state+"_"+suffix+"_count"] for r in rows)/total if total else None
            output[state+"_inversions"][suffix+"_query_mean_fraction"]=float(np.mean([r[state+"_"+suffix+"_count"]/r[state+"_inversion_count"] for r in active])) if active else None
    opportunities=sum(r["cross_boundary_pair_count"] for r in rows)
    output["opportunity_pairs"]=opportunities
    output["opportunity_small_margin_fraction"]=sum(r["opportunity_margin_le_small_count"] for r in rows)/opportunities
    return output


def summarize(config,derived_root,tables,figures,condition):
    cfg=read_conf(config);directory=derived_root/condition
    rows=list(jsonl(directory/"queries.jsonl"));report=summary(rows)
    tables.mkdir(parents=True,exist_ok=True);figures.mkdir(parents=True,exist_ok=True)
    prefix="phase3c_"+condition
    dump(tables/(prefix+"_summary.json"),report)
    core=[]
    for label in ("exact",32,64):
        core.append({"score":str(label),"mean_recall":report["oracle_recall"] if label=="exact" else report[f"pq{label}"]["fixed_candidate_recall"]["mean"],
            "mean_loss":0 if label=="exact" else report[f"pq{label}"]["ranking_recall_loss"]["mean"],
            "mean_inversions":0 if label=="exact" else report[f"pq{label}"]["cross_boundary_inversion_count"]["mean"]})
    write_csv(tables/(prefix+"_core.csv"),core)
    flags=[dict(flag=name,**r) for name,r in report["transition_flags"].items()]
    write_csv(tables/(prefix+"_transitions.csv"),flags)
    _,overlaps=overlap_summary(rows);write_csv(tables/(prefix+"_top_query_overlap.csv"),overlaps)
    correlations=[];bins=[]
    for subset in ("all","no_exact_boundary_ties"):
        selected=rows if subset=="all" else [r for r in rows if not r["exact_boundary_tie"]]
        for field in CHANGES:
            finite=[r for r in selected if r["delta_"+field] is not None]
            correlations.append({"subset":subset,"delta_metric":field,"query_count":len(finite),
                "spearman_with_loss_recovery":spearman([r["delta_"+field] for r in finite],[r["delta_loss_precision"] for r in finite])})
    write_csv(tables/(prefix+"_change_correlations.csv"),correlations)
    groups=[]
    for group in FLAGS:
        members=[r for r in rows if r[group]]
        for field in GEOMETRY+["oracle_recall"]:
            groups.append({"group":group,"metric":field,"query_count":len(members),**stats([r[field] for r in members])})
    write_csv(tables/(prefix+"_group_geometry.csv"),groups)
    # Exploratory sensitivity, added after observing unequal oracle recall
    # across the primary stability groups; it does not redefine those groups.
    full_gt_geometry=[]
    for group in FLAGS:
        members=[r for r in rows if r[group] and r["oracle_recall"]==1]
        for field in GEOMETRY:
            full_gt_geometry.append({"group":group,"metric":field,"query_count":len(members),**stats([r[field] for r in members])})
    write_csv(tables/(prefix+"_group_geometry_oracle1_exploratory.csv"),full_gt_geometry)
    pair_data={state:[] for state in ("corrected","persistent","new")}
    for pair in jsonl(directory/"inversion_transitions.jsonl.gz"):pair_data[pair["state"]].append(pair)
    pair_summary=[]
    for state,records in pair_data.items():
        for metric in ("exact_margin","relative_exact_margin","violation_32","violation_64","inside_exact_rank","outside_exact_rank"):
            pair_summary.append({"state":state,"metric":metric,"pair_count":len(records),**stats([r[metric] for r in records])})
    write_csv(tables/(prefix+"_pair_distributions.csv"),pair_summary)
    cutoffs=[]
    for state,records in pair_data.items():
        for threshold in (.01,float(cfg["relative_small_margin"]),float(cfg["relative_large_margin"])):
            hits=sum(r["relative_exact_margin"]<=threshold for r in records)
            cutoffs.append({"state":state,"relative_margin_cutoff":threshold,"pair_count":len(records),
                            "pairs_at_or_below_cutoff":hits,"fraction":hits/len(records) if records else None})
    for threshold,key in ((.01,"le_001"),(float(cfg["relative_small_margin"]),"le_small"),(float(cfg["relative_large_margin"]),"le_large")):
        count=sum(r["cross_boundary_pair_count"] for r in rows);hits=sum(r["opportunity_margin_"+key+"_count"] for r in rows)
        cutoffs.append({"state":"all_cross_boundary_opportunities","relative_margin_cutoff":threshold,
                        "pair_count":count,"pairs_at_or_below_cutoff":hits,"fraction":hits/count})
    write_csv(tables/(prefix+"_margin_cutoffs.csv"),cutoffs)
    for field in ("delta_global_mae","delta_cross_boundary_inversion_count"):
        edges=np.unique(np.quantile([r[field] for r in rows],np.linspace(0,1,int(cfg["quantile_bins"])+1)))
        labels=np.searchsorted(edges[1:-1],[r[field] for r in rows],side="right")
        for b in np.unique(labels):
            members=[r for r,l in zip(rows,labels) if l==b]
            ys=[r["delta_loss_precision"] for r in members]
            bins.append({"feature":field,"bin":int(b),"count":len(members),"mean_x":float(np.mean([r[field] for r in members])),
                "mean_recovery":float(np.mean(ys)),"query_standard_error":float(np.std(ys,ddof=1)/math.sqrt(len(ys))) if len(ys)>1 else 0})
    write_csv(tables/(prefix+"_change_bins.csv"),bins)
    clean=[r for r in rows if not r["exact_boundary_tie"]]
    dump(tables/(prefix+"_no_exact_ties_summary.json"),summary(clean))
    if condition=="exact_pool":
        dump(tables/(prefix+"_control_matched_subset.json"),summary([r for r in rows if r["query_id"]%int(cfg["control_query_stride"])==0]))
    plot(rows,report,pair_data,bins,figures,prefix)
    print(json.dumps({"condition":condition,"queries":len(rows),"core":core,"flags":report["transition_flags"],
                      "persistence":report["harmful_persistence"],"correlations":correlations[:len(CHANGES)]},indent=2))


def plot(rows,report,pairs,bins,destination,prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"svg.hashsalt":"phase3c","font.size":10})
    def save(fig,name):
        fig.tight_layout();path=destination/(prefix+"_"+name+".svg")
        fig.savefig(path,metadata={"Date":None});plt.close(fig)
        # Canonical whitespace avoids matplotlib SVG path trailing spaces.
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines())+"\n")
    for field,name in (("ranking_recall_loss","loss_32_vs_64"),("cross_boundary_inversion_count","inversions_32_vs_64")):
        fig,ax=plt.subplots(figsize=(5.7,4.7));x=np.array([r[field+"_32"] for r in rows]);y=np.array([r[field+"_64"] for r in rows])
        hb=ax.hexbin(x,y,gridsize=30,mincnt=1,bins="log");fig.colorbar(hb,ax=ax,label="Query count (log scale)")
        low=min(x.min(),y.min());high=max(x.max(),y.max());ax.plot([low,high],[low,high],"k--",lw=1)
        ax.set(xlabel="PQ32 "+field.replace("_"," "),ylabel="PQ64 "+field.replace("_"," "));save(fig,name)
    fig,ax=plt.subplots(figsize=(6.3,4.2));v=[r["delta_loss_precision"] for r in rows]
    ax.hist(v,bins=np.arange(-1.05,1.06,.1),edgecolor="white");ax.axvline(0,c="black",lw=1)
    ax.set(xlabel="Ranking recall loss PQ32 minus PQ64",ylabel="Queries");save(fig,"loss_improvement")
    fig,ax=plt.subplots(figsize=(7.3,4.5));fs=FLAGS[:4]+["fully_rescued"]
    ax.bar(range(len(fs)),[report["transition_flags"][f]["fraction"] for f in fs])
    ax.set_xticks(range(len(fs)),["Stable both","Improved\n(rescued)","Persistent\nharmful","Regressed","Fully rescued"])
    ax.set(ylabel="Fraction of all queries",title="Flags overlap; bars do not form a partition");save(fig,"transition_flags")
    fig,ax=plt.subplots(figsize=(6.3,4.2))
    for state,pp in pairs.items():
        if pp:
            x=np.sort([p["relative_exact_margin"] for p in pp]);points=np.unique(np.quantile(x,np.linspace(0,1,201)))
            ax.step(points,np.searchsorted(x,points,side="right")/len(x),where="post",label=f"{state} (n={len(x)})")
    ax.axvline(.05,c="black",ls="--",lw=1);ax.set(xscale="log",xlabel="Exact pair margin / exact dk",ylabel="Pair-weighted CDF",title="Empirical CDF at 201 quantile thresholds");ax.legend();save(fig,"pair_margin_cdf")
    for feature in ("delta_global_mae","delta_cross_boundary_inversion_count"):
        b=[r for r in bins if r["feature"]==feature];fig,ax=plt.subplots(figsize=(6.3,4.2))
        ax.errorbar([r["mean_x"] for r in b],[r["mean_recovery"] for r in b],yerr=[r["query_standard_error"] for r in b],marker="o")
        ax.set(xlabel=feature.replace("_"," ")+" (PQ32 minus PQ64)",ylabel="Mean fixed-candidate recall recovery",title="Quantile bins; query SE");save(fig,feature)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest="mode",required=True)
    c=sub.add_parser("prepare-control")
    for name in ("config","run"):c.add_argument(name,type=Path)
    c=sub.add_parser("derive")
    for name in ("config","run","derived"):c.add_argument(name,type=Path)
    c.add_argument("condition",choices=("exact_pool","pq64_pool_control"))
    c=sub.add_parser("summarize")
    for name in ("config","derived","tables","figures"):c.add_argument(name,type=Path)
    c.add_argument("condition",choices=("exact_pool","pq64_pool_control"))
    a=p.parse_args()
    if a.mode=="prepare-control":prepare_control(a.config,a.run)
    elif a.mode=="derive":derive(a.config,a.run,a.derived,a.condition)
    else:summarize(a.config,a.derived,a.tables,a.figures,a.condition)
