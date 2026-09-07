"""Simple balanced descriptive variance and nine-model inversion references."""
import numpy as np
from phase3d_metrics import single

NAMES=[f"{s}{i}" for s in "ABC" for i in (1,2,3)]
CATEGORIES=["Robust", "Model-specific", "Usually harmful", "Universally harmful"]


def variance_components(loss):
    x=np.asarray(loss,dtype=np.float64)
    if x.shape!=(3,3) or not np.isfinite(x).all():raise ValueError("finite balanced3x3 required")
    # Center first so exactly constant rows/designs produce exactly zero,
    # not spurious positive variances from summing repeated decimal losses.
    residual=x-x[:,:1];means=x[:,0]+residual.mean(axis=1);mu=float(means.mean())
    within=float(np.var(residual,axis=1).mean())
    between=float(np.var(means-means[0]));total=float(np.var(x-x[0,0]))
    assert abs(total-within-between)<1e-12
    ws=float(residual.var(axis=1,ddof=1).mean());bs=float((means-means[0]).var(ddof=1))
    return {"overall_mean_loss":mu,"subset_mean_loss":means.tolist(),
            "within_subset_variance":within,"between_subset_variance":between,
            "within_subset_variability":within,"between_subset_variability":between,
            "initialization_variability":within,"training_subset_variability":between,
            "total_model_variability":total,"within_subset_sample_variance":ws,
            "subset_means_sample_variance":bs,
            "signed_subset_variance_contrast":bs-ws/3,
            "initialization_sd":float(np.sqrt(within)),
            "observed_subset_mean_sd":float(np.sqrt(between)),
            "total_model_sd":float(np.sqrt(total))}


def classify(loss):
    x=np.asarray(loss,dtype=float)
    if x.shape!=(3,3):raise ValueError("balanced3x3 required")
    per_subset=(x>0).sum(axis=1);all_count=int(per_subset.sum())
    subset_count=int((per_subset>=2).sum())
    return {"harmful_count_all":all_count,"harmful_frequency_all":all_count/9,
            "harmful_count_by_subset":per_subset.tolist(),
            "harmful_subset_count":subset_count,"harmful_subset_frequency":subset_count/3,
            "subset_persistent":subset_count==3,
            "category":CATEGORIES[0 if all_count<=2 else 1 if all_count<=5 else 2 if all_count<=8 else 3]}


def inversion_union(ids,exact,scores,k):
    ids=np.asarray(ids,dtype=np.int64);d=np.asarray(exact,dtype=np.float64)
    p=np.asarray(scores,dtype=np.float64)
    if p.shape!=(9,len(ids)) or len(set(ids))!=len(ids):raise ValueError("unaligned unique candidates")
    order=np.argsort(d,kind="stable");inside,outside=order[:k],order[k:]
    margins=d[outside][None,:]-d[inside][:,None]
    inv=np.array([(margins>0)&(p[m,inside,None]>p[m,outside][None,:]) for m in range(9)])
    count=inv.sum(axis=0);pairs=[]
    for a,b in zip(*np.nonzero(count)):
        i,j=inside[a],outside[b];mask=inv[:,a,b];sc=mask.reshape(3,3).sum(axis=1)
        models=int(mask.sum());subsets=int((sc>0).sum())
        pairs.append({"inside_id":int(ids[i]),"outside_id":int(ids[j]),
            "inside_exact_rank":int(a+1),"outside_exact_rank":int(b+k+1),
            "exact_margin":float(margins[a,b]),"violation_by_model":(p[:,i]-p[:,j]).tolist(),
            "model_mask":mask.tolist(),"model_count":models,"model_frequency":models/9,
            "subset_init_counts":sc.tolist(),"subset_frequency":subsets,"subset_frequency_fraction":subsets/3,
            "subsets_with_init_majority":int((sc>=2).sum()),"subsets_with_all_inits":int((sc==3).sum())})
    return pairs,inv.sum(axis=(1,2))


def measure_factorial(ids,exact,scores,truth,k=10,seed=1,samples=512,epsilon=1e-12):
    p=np.asarray(scores)
    if p.shape!=(9,len(ids)):raise ValueError("exactly9 scorers required")
    models=[];replacements=[];geometry=None
    for m in range(9):
        metric,g,repl,random=single(ids,exact,p[m],truth,k,seed,samples,epsilon)
        if geometry is not None:assert geometry==g
        geometry=g;metric.update(model=NAMES[m],subset=NAMES[m][0],harmful=metric["ranking_recall_loss"]>0)
        models.append(metric)
        for pair in repl:
            pair.pop("quantizer");pair["model"]=NAMES[m];replacements.append(pair)
    pairs,counts=inversion_union(ids,exact,p,k)
    assert [m["cross_boundary_inversion_count"] for m in models]==counts.tolist()
    loss=np.array([m["ranking_recall_loss"] for m in models]).reshape(3,3)
    return {**geometry,**variance_components(loss),**classify(loss),"models":models,
            "union_inversion_pairs":len(pairs),
            "one_model_pairs":sum(r["model_count"]==1 for r in pairs),
            "one_subset_pairs":sum(r["subset_frequency"]==1 for r in pairs),
            "all_subset_pairs":sum(r["subset_frequency"]==3 for r in pairs)},replacements,pairs,random
