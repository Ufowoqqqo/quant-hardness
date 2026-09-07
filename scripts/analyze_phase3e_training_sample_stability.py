#!/usr/bin/env python3
"""Balanced 3 x 3 score-only training-sample experiment; separate historical O."""
import argparse
import itertools
import json
import shutil
from pathlib import Path
import numpy as np
from analyze_phase3a_sift1m import read_conf, spearman, write_csv
from analyze_phase3b_fixed_candidate_ranking import digest, dump
from analyze_phase3c_precision_transition import jsonl, read_candidates, gz_writer, stats
from analyze_phase3d_quantizer_seed_stability import prepare as prepare_inputs, verify_hashes, variability, savefig
from phase3d_metrics import single, GEOMETRY
from phase3e_metrics import measure_factorial, CATEGORIES

NAMES=[f"{s}{i}" for s in "ABC" for i in (1,2,3)]
OLD_NAMES=[f"O{i}" for i in range(1,6)]


def prepare(config,run):
    cfg=read_conf(config);old=Path(cfg["phase3d_run"])
    # Preserve and validate the full original source/model provenance.
    verify_hashes(json.loads((old/"input_provenance.json").read_text())["input_sha256"])
    verify_hashes(json.loads((old/"final_verification.json").read_text())["sha256"])
    prepare_inputs(config,run)
    paths=[old/"analysis"/"queries.jsonl.gz",old/"analysis"/"validation.json",old/"resolved_config.conf"]
    for i in range(5):
        directory=old/f"seed_{i}";m=json.loads((directory/"manifest.json").read_text())
        assert m["pq_m"]==64 and m["pq_nbits"]==8 and m["training_count"]==65536
        assert m["queries"]==10000 and m["candidates"]==10353047
        assert m["requests_sha256"]==digest(run/"candidate_requests.tsv")
        paths += sorted(p for p in directory.iterdir() if p.is_file())
    dump(run/"secondary_provenance.json",{"eligible":True,"scope":"secondary aggregate and query stability only; O excluded from balanced3x3",
        "input_sha256":{str(p):digest(p) for p in paths}})
    print("secondary O metadata compatibility=PASS",flush=True)


def validate_models(config,run):
    cfg=read_conf(config);manifests=[];samples={}
    for index,name in enumerate(NAMES):
        root=run/name;m=json.loads((root/"manifest.json").read_text());manifests.append(m)
        assert m["model_name"]==name and m["training_subset"]==name[0]
        assert m["seed"]==int(cfg["training_seeds"].split(",")[index])
        assert m["training_sample_seed"]==int(cfg["subset_sampling_seeds"].split(",")[index//3])
        assert m["pq_m"]==64 and m["pq_nbits"]==8 and m["training_count"]==65536
        for file,key in (("scores.f32","scores_sha256"),("pq64.index","model_file_sha256"),("training_ids.i32","training_ids_sha256")):
            assert digest(root/file)==m[key],(name,file)
        ids=np.fromfile(root/"training_ids.i32",dtype="<i4")
        assert len(ids)==65536 and len(np.unique(ids))==65536 and ids.min()>=0 and ids.max()<1000000
        if name[0] in samples:assert np.array_equal(ids,samples[name[0]])
        samples[name[0]]=ids
    for key in ("requests_sha256","graph_fingerprint","faiss_commit","config_sha256"):
        assert len({m[key] for m in manifests})==1,key
    assert len({m["codebook_sha256"] for m in manifests})==9
    assert len({m["training_ids_sha256"] for m in manifests})==3
    for group in "ABC":
        assert len({m["training_vectors_sha256"] for m in manifests if m["training_subset"]==group})==1
    old=Path(cfg["phase3d_run"]);samples["O"]=np.fromfile(old/"seed_0"/"training_ids.i32",dtype="<i4")
    overlaps=[]
    for a,b in itertools.combinations(samples,2):
        A=set(samples[a].tolist());B=set(samples[b].tolist())
        overlaps.append({"subset_a":a,"subset_b":b,"intersection":len(A&B),"jaccard":len(A&B)/len(A|B),
                         "expected_independent_intersection":65536**2/1000000})
    return manifests,overlaps


def derive(config,run,out):
    cfg=read_conf(config);source=Path(cfg["candidate_source"]);old=Path(cfg["phase3d_run"])
    for name in ("input_provenance.json","secondary_provenance.json"):
        verify_hashes(json.loads((run/name).read_text())["input_sha256"])
    manifests,overlaps=validate_models(config,run)
    maps=[np.memmap(run/name/"scores.f32",dtype="<f4",mode="r") for name in NAMES]
    old_maps=[np.memmap(old/f"seed_{s}"/"scores.f32",dtype="<f4",mode="r") for s in range(5)]
    old_rows=iter(jsonl(old/"analysis"/"queries.jsonl.gz"))
    prior_rows=iter(jsonl(Path(cfg["phase3c_analysis"])/"queries.jsonl"))
    old_random=np.memmap(Path(cfg["phase3c_analysis"])/"random_pair_indices.bin",dtype="<i4",mode="r").reshape(10000,-1,2)
    out.mkdir(parents=True,exist_ok=False);offset=0
    pair_inter=np.zeros((9,9),dtype=np.int64);pair_union=pair_inter.copy()
    histogram=np.zeros((10,4),dtype=np.int64);within_repeat=np.zeros((4,4),dtype=np.int64)
    with gz_writer(out/"queries.jsonl.gz") as queries,gz_writer(out/"replacement_pairs.jsonl.gz") as repl, \
            gz_writer(out/"inversion_pairs.jsonl.gz") as pairs:
        for meta,a in read_candidates(source,jsonl(source/"queries.jsonl")):
            qid=meta["query_id"];n=len(a);prior=next(prior_rows);o=next(old_rows)
            assert prior["query_id"]==o["query_id"]==qid
            score=np.array([p[offset:offset+n] for p in maps])
            row,replacements,union,random=measure_factorial(a["id"],a["exact"],score,meta["ground_truth_ids"],
                int(cfg["k"]),int(cfg["random_pair_seed"])+qid,int(cfg["random_pairs_per_query"]),float(cfg["epsilon"]))
            for key in GEOMETRY+["exact_topk_ids","exact_topk_set","exact_boundary_tie"]:
                assert row[key]==prior[key]==o[key],(qid,key)
            assert meta["candidate_order_sha256"]==prior["candidate_order_sha256"]==o["candidate_order_sha256"]
            assert meta["ground_truth_ids"]==prior["ground_truth_ids"]==o["ground_truth_ids"]
            assert np.array_equal(random,old_random[qid])
            # Secondary saved metrics have full historical verification; audit100
            # deterministic queries/model directly from their immutable raw scores.
            if qid%100==0:
                for s in range(5):
                    check,_,_,_=single(a["id"],a["exact"],old_maps[s][offset:offset+n],meta["ground_truth_ids"],
                        int(cfg["k"]),int(cfg["random_pair_seed"])+qid,int(cfg["random_pairs_per_query"]),float(cfg["epsilon"]))
                    for key,value in check.items():assert value==o["models"][s][key],(qid,s,key)
            row.update(query_id=qid,candidate_order_sha256=meta["candidate_order_sha256"],
                       ground_truth_ids=meta["ground_truth_ids"],score_offset=offset)
            queries.write(json.dumps(row,allow_nan=False)+"\n")
            for p in replacements:
                p["query_id"]=qid;repl.write(json.dumps(p,allow_nan=False)+"\n")
            for p in union:
                p["query_id"]=qid;pairs.write(json.dumps(p,allow_nan=False)+"\n")
                mask=np.array(p["model_mask"])
                pair_inter+=mask[:,None]&mask[None,:];pair_union+=mask[:,None]|mask[None,:]
                histogram[p["model_count"],p["subset_frequency"]]+=1
                within_repeat[p["subsets_with_init_majority"],p["subsets_with_all_inits"]]+=1
            offset+=n
            if (qid+1)%1000==0:print(f"derive queries={qid+1}",flush=True)
    assert next(prior_rows,None) is None and next(old_rows,None) is None
    assert all(len(p)==offset for p in maps+old_maps) and qid==9999
    dump(out/"pair_summary.json",{"model_subset_frequency":histogram.tolist(),"intersection":pair_inter.tolist(),
        "union":pair_union.tolist(),"subset_majority_all_init_counts":within_repeat.tolist()})
    dump(out/"sample_overlap.json",overlaps)
    for name in ("input_provenance.json","secondary_provenance.json"):
        verify_hashes(json.loads((run/name).read_text())["input_sha256"])
    dump(out/"validation.json",{"queries":10000,"candidates":offset,"balanced_models":NAMES,
        "shared_exact_pool_gt_geometry_identity":True,"variance_identity_all_queries":True,
        "old_O_score_audit_queries_per_model":100,"random_pair_identity":True,"old_sources_unchanged":True,
        "training_sample_identity_within_subset":True,"distinct_training_subsets":3,"distinct_codebooks":9,
        "analysis_sha256":digest(__file__),"metrics_sha256":digest(Path(__file__).with_name("phase3e_metrics.py")),
        "model_manifests_sha256":{name:digest(run/name/"manifest.json") for name in NAMES},"numpy":np.__version__})
    print("derive status=PASS",flush=True)


FIELDS=["fixed_candidate_recall","ranking_recall_loss","cross_boundary_inversion_count",
        "global_mae","random_pair_inversion_rate","topk_agreement"]
PAIR_FIELDS=["harmful_jaccard","loss_spearman","inversion_spearman"]
TARGETS=["harmful_frequency_all","harmful_subset_frequency","overall_mean_loss",
         "between_subset_variability","within_subset_variability"]


def table(root,name,rows):
    write_csv(root/f"phase3e_{name}.csv",rows);dump(root/f"phase3e_{name}.json",rows)


def pairwise(names,losses,inversions,pair_summary=None):
    records=[]
    for a,b in itertools.combinations(range(len(names)),2):
        A=losses[a]>0;B=losses[b]>0;pa=float(A.mean());pb=float(B.mean())
        union=int((A|B).sum())
        r={"model_a":names[a],"model_b":names[b],
           "relation":"within" if names[a][0]==names[b][0] else "between",
           "harmful_jaccard":float((A&B).sum()/union) if union else 1.0,
           "loss_spearman":spearman(losses[a].tolist(),losses[b].tolist()),
           "inversion_spearman":spearman(inversions[a].tolist(),inversions[b].tolist()),
           "homogeneous_independent_jaccard":pa*pb/(pa+pb-pa*pb) if pa+pb-pa*pb else 1.0}
        if pair_summary is not None:
            u=pair_summary["union"][a][b];r["inversion_pair_jaccard"]=pair_summary["intersection"][a][b]/u if u else 1.0
        records.append(r)
    return records


def distribution(v):
    v=[x for x in v if x is not None]
    return {**stats(v),"std":float(np.std(v,ddof=1)) if len(v)>1 else None}


def quality(names,model_rows):
    return [{"model":name,"subset":name[0],**{f:float(np.mean([r[m][f] for r in model_rows])) for f in FIELDS}}
            for m,name in enumerate(names)]


def summarize(config,run,derived,tables,figures):
    cfg=read_conf(config);rows=list(jsonl(derived/"queries.jsonl.gz"));n=len(rows)
    assert n==10000
    tables.mkdir(parents=True,exist_ok=True);figures.mkdir(parents=True,exist_ok=True)
    models=[r["models"] for r in rows];aggregate=quality(NAMES,models)
    for m,a in enumerate(aggregate):
        meta=json.loads((run/NAMES[m]/"manifest.json").read_text())
        a.update(initialization_seed=meta["seed"],sampling_seed=meta["training_sample_seed"],
                 training_ids_sha256=meta["training_ids_sha256"],codebook_sha256=meta["codebook_sha256"])
    table(tables,"models",aggregate)
    quality_variation=[]
    for field in FIELDS:
        values=np.array([r[field] for r in aggregate]).reshape(3,3)
        for j,subset in enumerate("ABC"):
            quality_variation.append({"metric":field,"scope":"within_subset","subset":subset,"values":3,**variability(values[j])})
        quality_variation.append({"metric":field,"scope":"between_subset_means","subset":"ABC","values":3,**variability(values.mean(axis=1))})
        quality_variation.append({"metric":field,"scope":"all9_model_values","subset":"ABC","values":9,**variability(values.ravel())})
    table(tables,"aggregate_variation",quality_variation)
    loss=np.array([[r["models"][m]["ranking_recall_loss"] for r in rows] for m in range(9)])
    inv=np.array([[r["models"][m]["cross_boundary_inversion_count"] for r in rows] for m in range(9)])
    pair_data=json.loads((derived/"pair_summary.json").read_text())
    primary_pairs=pairwise(NAMES,loss,inv,pair_data);table(tables,"pairwise_primary",primary_pairs)
    pair_dist=[{"relation":relation,"metric":metric,"pairs":sum(r["relation"]==relation for r in primary_pairs),
                **distribution([r[metric] for r in primary_pairs if r["relation"]==relation])}
               for relation in ("within","between") for metric in PAIR_FIELDS+["inversion_pair_jaccard"]]
    table(tables,"pairwise_distributions",pair_dist)
    difference={metric:float(np.median([r[metric] for r in primary_pairs if r["relation"]=="within"])-
                             np.median([r[metric] for r in primary_pairs if r["relation"]=="between"])) for metric in PAIR_FIELDS+["inversion_pair_jaccard"]}
    var_fields=["within_subset_variance","between_subset_variance","total_model_variability",
                "within_subset_sample_variance","subset_means_sample_variance","signed_subset_variance_contrast",
                "initialization_sd","observed_subset_mean_sd","total_model_sd"]
    variance=[{"metric":f,**distribution([r[f] for r in rows])} for f in var_fields]
    table(tables,"variance",variance)
    w=float(np.mean([r["within_subset_variance"] for r in rows]));b=float(np.mean([r["between_subset_variance"] for r in rows]));t=float(np.mean([r["total_model_variability"] for r in rows]))
    between_query=float(np.var([r["overall_mean_loss"] for r in rows]));joint=float(np.var(loss))
    assert abs(w+b-t)<1e-12 and abs(joint-between_query-t)<1e-12
    variance_accounting={"mean_initialization_variability":w,"mean_observed_subset_mean_variability":b,
        "mean_total_model_variability":t,"within_share_of_observed_model_variance":w/t,
        "between_share_of_observed_model_variance":b/t,
        "mean_W_sample":float(np.mean([r["within_subset_sample_variance"] for r in rows])),
        "mean_B_sample":float(np.mean([r["subset_means_sample_variance"] for r in rows])),
        "mean_signed_subset_contrast":float(np.mean([r["signed_subset_variance_contrast"] for r in rows])),
        "negative_contrast_queries":sum(r["signed_subset_variance_contrast"] < -1e-14 for r in rows),
        "zero_model_variance_queries":sum(r["total_model_variability"] < 1e-14 for r in rows),
        "between_query_mean_loss_variance_not_causal_geometry":between_query,
        "all_query_model_variance":joint}
    dump(tables/"phase3e_variance_accounting.json",variance_accounting)
    total=float(loss.sum());positive=float(np.maximum(loss,0).sum());category_rows=[]
    for scope in ("all","oracle1"):
        subset=rows if scope=="all" else [r for r in rows if r["oracle_recall"]==1]
        denom=sum(sum(m["ranking_recall_loss"] for m in r["models"]) for r in subset)
        for label in CATEGORIES+["subset-persistent","not-subset-persistent"]:
            members=[r for r in subset if (r["subset_persistent"] if label=="subset-persistent" else not r["subset_persistent"] if label=="not-subset-persistent" else r["category"]==label)]
            value=sum(sum(m["ranking_recall_loss"] for m in r["models"]) for r in members)
            pos=sum(max(m["ranking_recall_loss"],0) for r in members for m in r["models"])
            category_rows.append({"scope":scope,"category":label,"queries":len(members),"fraction":len(members)/len(subset),
                "signed_loss_sum":value,"positive_loss_sum":pos,"fraction_scope_signed_loss":value/denom,
                "mean_oracle_recall":float(np.mean([r["oracle_recall"] for r in members])) if members else None})
    table(tables,"categories",category_rows)
    frequency=[{"harmful_count_all":i,"harmful_frequency_all":i/9,"queries":sum(r["harmful_count_all"]==i for r in rows)} for i in range(10)]
    subset_freq=[{"harmful_subset_count":i,"harmful_subset_frequency":i/3,"queries":sum(r["harmful_subset_count"]==i for r in rows)} for i in range(4)]
    table(tables,"harmful_frequency",frequency);table(tables,"harmful_subset_frequency",subset_freq)
    hist=np.array(pair_data["model_subset_frequency"]);union=int(hist.sum())
    pair_freq=[{"model_count":m,"model_frequency":m/9,"subset_frequency":s,"pairs":int(hist[m,s]),"fraction_union":float(hist[m,s]/union)} for m in range(1,10) for s in range(1,4)]
    table(tables,"pair_frequency",pair_freq)
    repetition=np.array(pair_data["subset_majority_all_init_counts"])
    table(tables,"pair_within_repetition",[{"subsets_with_init_majority":m,"subsets_with_all_inits":s,"pairs":int(repetition[m,s]),"fraction_union":float(repetition[m,s]/union)} for m in range(4) for s in range(4)])
    pair_frequency_summary={"union_pairs":union,"one_model_pairs":int(hist[1].sum()),
        "one_model_fraction":float(hist[1].sum()/union),"one_subset_pairs":int(hist[:,1].sum()),
        "one_subset_fraction":float(hist[:,1].sum()/union),"at_least_two_subsets_fraction":float(hist[:,2:].sum()/union),
        "all_three_subsets_fraction":float(hist[:,3].sum()/union),"all9_models_pairs":int(hist[9].sum()),
        "repeated_within_any_subset_fraction":float(repetition[1:].sum()/union)}
    dump(tables/"phase3e_pair_frequency_summary.json",pair_frequency_summary)
    table(tables,"sample_overlap",json.loads((derived/"sample_overlap.json").read_text()))
    geom=[];corr=[];bins=[];realization=[]
    for scope in ("all","oracle1"):
        subset=rows if scope=="all" else [r for r in rows if r["oracle_recall"]==1]
        for descriptor in GEOMETRY:
            xs=[r[descriptor] for r in subset]
            for target in TARGETS:
                corr.append({"scope":scope,"descriptor":descriptor,"target":target,"queries":len(subset),
                             "spearman":spearman(xs,[r[target] for r in subset])})
                edges=np.unique(np.quantile(xs,np.linspace(0,1,int(cfg["quantile_bins"])+1)))
                labels=np.searchsorted(edges[1:-1],xs,side="right")
                for bindex in sorted(set(labels.tolist())):
                    members=[r for r,label in zip(subset,labels) if label==bindex]
                    bins.append({"scope":scope,"descriptor":descriptor,"target":target,"bin":bindex,"queries":len(members),
                        "descriptor_mean":float(np.mean([r[descriptor] for r in members])),"target_mean":float(np.mean([r[target] for r in members]))})
            mean_r=spearman(xs,[r["overall_mean_loss"] for r in subset])
            for m in range(9):
                realization.append({"scope":scope,"descriptor":descriptor,"model":NAMES[m],"queries":len(subset),
                    "realized_loss_spearman":spearman(xs,[r["models"][m]["ranking_recall_loss"] for r in subset]),
                    "mean_loss_spearman":mean_r})
            for label in CATEGORIES+["subset-persistent","not-subset-persistent"]:
                members=[r[descriptor] for r in subset if (r["subset_persistent"] if label=="subset-persistent" else not r["subset_persistent"] if label=="not-subset-persistent" else r["category"]==label)]
                geom.append({"scope":scope,"category":label,"descriptor":descriptor,"queries":len(members),**stats(members)})
    table(tables,"geometry",geom);table(tables,"geometry_correlations",corr);table(tables,"geometry_bins",bins);table(tables,"geometry_mean_vs_realized",realization)
    sensitivity=[]
    for scope in ("oracle1","no_exact_boundary_ties"):
        subset=[r for r in rows if (r["oracle_recall"]==1 if scope=="oracle1" else not r["exact_boundary_tie"])]
        l=np.array([[r["models"][m]["ranking_recall_loss"] for r in subset] for m in range(9)])
        v=np.array([[r["models"][m]["cross_boundary_inversion_count"] for r in subset] for m in range(9)])
        sensitivity += [{"scope":scope,"queries":len(subset),**r} for r in pairwise(NAMES,l,v)]
    table(tables,"stability_sensitivity",sensitivity)
    # Secondary O data remain separate from all primary calculations above.
    old_rows=list(jsonl(Path(cfg["phase3d_run"])/"analysis"/"queries.jsonl.gz"))
    old_models=[r["models"] for r in old_rows]
    secondary_quality=quality(NAMES,models)+quality(OLD_NAMES,old_models)
    table(tables,"secondary_O_models",secondary_quality)
    old_loss=np.array([[r["models"][m]["ranking_recall_loss"] for r in old_rows] for m in range(5)])
    old_inv=np.array([[r["models"][m]["cross_boundary_inversion_count"] for r in old_rows] for m in range(5)])
    secondary_pairs=pairwise(NAMES+OLD_NAMES,np.vstack([loss,old_loss]),np.vstack([inv,old_inv]))
    table(tables,"secondary_O_pairwise",secondary_pairs)
    sec_summary=[]
    for relation in ("O_within","O_to_ABC","ABC_within","ABC_between"):
        members=[r for r in secondary_pairs if (r["model_a"].startswith("O") and r["model_b"].startswith("O") if relation=="O_within" else
            r["model_a"].startswith("O") != r["model_b"].startswith("O") if relation=="O_to_ABC" else
            not r["model_a"].startswith("O") and not r["model_b"].startswith("O") and r["relation"]==relation.split("_")[1])]
        for metric in PAIR_FIELDS:sec_summary.append({"scope":relation,"metric":metric,"pairs":len(members),**distribution([r[metric] for r in members])})
    table(tables,"secondary_O_pair_distributions",sec_summary)
    table(tables,"secondary_O_quality_variation",[{"subset":group,"metric":field,**variability([r[field] for r in secondary_quality if r["subset"]==group])} for group in "ABCO" for field in FIELDS])
    prevalence=(loss>0).mean(axis=1);subset_majority=np.array([r["harmful_count_by_subset"] for r in rows])>=2
    summary={"design":"balanced ABC3x3 only","queries":n,"oracle_recall":float(np.mean([r["oracle_recall"] for r in rows])),
        "model_quality":aggregate,"quality_variation":quality_variation,"pair_distributions":pair_dist,
        "within_minus_between_medians":difference,"variance_accounting":variance_accounting,
        "categories":category_rows,"pair_frequency":pair_frequency_summary,
        "negative_query_model_losses":int((loss<0).sum()),"signed_total_loss":total,"positive_total_loss":positive,
        "harmful_prevalence_by_model":prevalence.tolist(),"subset_majority_prevalence":subset_majority.mean(axis=0).tolist(),
        "independent_model_marginals_expected_universal":float(n*np.prod(prevalence)),
        "independent_subset_majority_marginals_expected_subset_persistent":float(n*np.prod(subset_majority.mean(axis=0))),
        "oracle1_queries":sum(r["oracle_recall"]==1 for r in rows),"exact_boundary_tie_queries":sum(r["exact_boundary_tie"] for r in rows),
        "pq_boundary_ties_by_model":[sum(r["models"][m]["pq_boundary_tie"] for r in rows) for m in range(9)]}
    dump(tables/"phase3e_summary.json",summary)
    plots(rows,primary_pairs,frequency,subset_freq,bins,hist,figures)
    print(json.dumps({"within_minus_between_medians":difference,"variance":variance_accounting}),flush=True)


def plots(rows,pairs,frequency,subset_freq,bins,hist,out):
    import matplotlib
    matplotlib.use("Agg");matplotlib.rcParams["svg.hashsalt"]="phase3e"
    import matplotlib.pyplot as plt
    for metric,name in (("harmful_jaccard","within_between_jaccard"),("loss_spearman","within_between_loss_spearman")):
        fig,ax=plt.subplots(figsize=(5,4))
        ax.boxplot([[r[metric] for r in pairs if r["relation"]==relation] for relation in ("within","between")],labels=["Within subset (9)","Between subsets (27)"])
        ax.set(ylabel=metric);fig.tight_layout();savefig(fig,out/f"phase3e_{name}.svg");plt.close(fig)
    for data,field,name in ((frequency,"harmful_count_all","harmful_frequency_all"),(subset_freq,"harmful_subset_count","harmful_subset_frequency")):
        fig,ax=plt.subplots(figsize=(6,4));ax.bar([r[field] for r in data],[r["queries"] for r in data])
        ax.set(xlabel="Harmful models / 9" if field=="harmful_count_all" else "Subsets with >=2/3 harmful initializations / 3",ylabel="Queries")
        fig.tight_layout();savefig(fig,out/f"phase3e_{name}.svg");plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4));im=ax.hexbin([r["within_subset_variance"] for r in rows],[r["between_subset_variance"] for r in rows],gridsize=30,mincnt=1)
    lim=max(max(r["within_subset_variance"],r["between_subset_variance"]) for r in rows)
    ax.plot([0,lim],[0,lim],"k--",lw=.7);fig.colorbar(im,ax=ax,label="Queries")
    ax.set(xlabel="Within-subset population variance",ylabel="Observed variance of subset means")
    fig.tight_layout();savefig(fig,out/"phase3e_variability.svg");plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,descriptor in zip(axes,("relative_boundary_margin","boundary_gap_r2","boundary_gap_r5")):
        for scope in ("all","oracle1"):
            selected=[r for r in bins if r["descriptor"]==descriptor and r["target"]=="harmful_frequency_all" and r["scope"]==scope]
            ax.plot([r["descriptor_mean"] for r in selected],[r["target_mean"] for r in selected],"o-",label=scope)
        ax.set(xlabel=descriptor,ylabel="Mean fraction of harmful models");ax.legend()
    fig.tight_layout();savefig(fig,out/"phase3e_geometry.svg");plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].bar(range(1,10),hist[1:,:].sum(axis=1));axes[0].set(xlabel="Models inverting pair / 9",ylabel="Union inversion pairs")
    axes[1].bar(range(1,4),hist[:,1:].sum(axis=0));axes[1].set(xlabel="Subsets with >=1 inverted realization / 3",ylabel="Union inversion pairs")
    fig.tight_layout();savefig(fig,out/"phase3e_pair_frequency.svg");plt.close(fig)


def checkpoint(run,tables):
    summary=json.loads((tables/"phase3e_summary.json").read_text())
    assert summary["queries"]==10000 and not (run/"ensemble").exists()
    out=run/"primary_checkpoint";out.mkdir(exist_ok=False)
    for file in sorted(tables.glob("phase3e_*")):shutil.copyfile(file,out/file.name)
    dump(out/"checkpoint.json",{"stage":"balanced primary and secondary O analysis complete before diagnostics",
        "summary_sha256":digest(out/"phase3e_summary.json"),"raw_validation_sha256":digest(run/"analysis"/"validation.json")})


def ensemble(config,run,out,tables,figures):
    cfg=read_conf(config);source=Path(cfg["candidate_source"])
    check=run/"primary_checkpoint"/"phase3e_summary.json";assert check.exists()
    primary=json.loads(check.read_text());out.mkdir(parents=True,exist_ok=False)
    groups={name:cfg[name].split(",") for name in ("initialization_ensemble","cross_subset_ensemble")}
    assert groups=={"initialization_ensemble":["A1","A2","A3"],"cross_subset_ensemble":["A1","B1","C1"]}
    maps={name:np.memmap(run/name/"scores.f32",dtype="<f4",mode="r") for name in NAMES}
    offset=0;records=[]
    with gz_writer(out/"queries.jsonl.gz") as target:
        for meta,a in read_candidates(source,jsonl(source/"queries.jsonl")):
            qid=meta["query_id"];n=len(a);entry={"query_id":qid,"candidate_order_sha256":meta["candidate_order_sha256"]}
            for name,members in groups.items():
                scores=np.array([maps[m][offset:offset+n] for m in members],dtype=np.float64).mean(axis=0)
                metric,geometry,_,_=single(a["id"],a["exact"],scores,meta["ground_truth_ids"],int(cfg["k"]),
                    int(cfg["random_pair_seed"])+qid,int(cfg["random_pairs_per_query"]),float(cfg["epsilon"]))
                entry[name]=metric;entry["oracle_recall"]=geometry["oracle_recall"]
            target.write(json.dumps(entry,allow_nan=False)+"\n");records.append(entry);offset+=n
    summary=[]
    for name,members in groups.items():
        r={"diagnostic":name,"members":",".join(members),"queries":len(records)}
        for metric in ("fixed_candidate_recall","ranking_recall_loss","cross_boundary_inversion_count"):
            r[metric]=float(np.mean([x[name][metric] for x in records]))
            r["member_mean_"+metric]=float(np.mean([x[metric] for x in primary["model_quality"] if x["model"] in members]))
        r["fraction_member_mean_loss_recovered"]=1-r["ranking_recall_loss"]/r["member_mean_ranking_recall_loss"]
        summary.append(r)
    table(tables,"ensemble",summary)
    dump(out/"summary.json",{"diagnostics":summary,"primary_checkpoint_sha256":digest(check),"accumulation":"float64; three uniform weights"})
    import matplotlib
    matplotlib.use("Agg");matplotlib.rcParams["svg.hashsalt"]="phase3e"
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,metric in zip(axes,("fixed_candidate_recall","ranking_recall_loss","cross_boundary_inversion_count")):
        ax.bar([0,1],[r[metric] for r in summary]);ax.set_xticks([0,1],["A1/A2/A3","A1/B1/C1"]);ax.set(ylabel=metric)
    fig.tight_layout();savefig(fig,figures/"phase3e_ensemble.svg");plt.close(fig)
    print(json.dumps(summary),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument("mode",choices=["prepare","derive","summarize","checkpoint","ensemble"])
    p.add_argument("--config",type=Path,default=Path("configs/indexes/phase3e_training_sample_stability.conf"))
    p.add_argument("--run",type=Path,default=Path("runs/phase3e_training_sample_stability_v1"))
    p.add_argument("--derived",type=Path);p.add_argument("--tables",type=Path,default=Path("results/tables"));p.add_argument("--figures",type=Path,default=Path("results/figures"))
    a=p.parse_args();derived=a.derived or a.run/"analysis"
    if a.mode=="prepare":prepare(a.config,a.run)
    elif a.mode=="derive":derive(a.config,a.run,derived)
    elif a.mode=="summarize":summarize(a.config,a.run,derived,a.tables,a.figures)
    elif a.mode=="checkpoint":checkpoint(a.run,a.tables)
    else:ensemble(a.config,a.run,a.derived or a.run/"ensemble",a.tables,a.figures)


if __name__=="__main__":main()
