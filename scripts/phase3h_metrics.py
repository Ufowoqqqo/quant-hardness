"""Hierarchical reference permutations and PQ-only observables; no ANN code."""
import numpy as np
from phase3g_metrics import permutations

CONDITIONS=['N0','N1','N2','N3','N4','N_distance']
EDGES={'N1':[10],'N2':[5,10,20],'N3':[5,10,15,20,40],'N4':[5,8,10,12,15,20,40]}
OBSERVABLES=['g10_11_pq','g9_12_pq','relative_g10_11_pq','relative_g9_12_pq','count_1pct','count_2pct','count_5pct']


def bins(order,condition):
    order=np.asarray(order);n=len(order)
    if condition=='N0':return [np.arange(n)]
    if condition=='N_distance':return [b for b in np.array_split(order,4) if len(b)]
    if condition not in EDGES:raise ValueError(condition)
    bounds=[0]+[x for x in EDGES[condition] if x<n]+[n]
    return [order[a:b] for a,b in zip(bounds,bounds[1:]) if b>a]


def conditioned_permutations(order,count,seed,model,query,condition,n0seed=90700011):
    n=len(order)
    if condition=='N0':return permutations(n,count,n0seed,model,query)
    groups=bins(order,condition);ci=CONDITIONS.index(condition);out=np.tile(np.arange(n),(count,1))
    for p in range(count):
        rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed,model,query,ci,p])))
        for b in groups:out[p,b]=rng.permutation(b)
    return out


def verify_assignment(idx,order,condition):
    for group in bins(order,condition):
        if not np.array_equal(np.sort(idx[:,group],axis=1),np.broadcast_to(np.sort(group),idx[:,group].shape)):
            raise AssertionError('permutation left its registered bin')


def closure(n0,nj,observed,epsilon=1e-6):
    denominator=n0-observed
    return float((n0-nj)/denominator) if denominator>epsilon else None


def pq_observables(scores,epsilon=1e-12):
    p=np.asarray(scores,dtype=float)
    if p.ndim!=1 or len(p)<12 or not np.isfinite(p).all():raise ValueError('finite PQ candidate scores required')
    s=np.sort(p,kind='stable');scale=max(abs(float(s[9])),epsilon)
    a=float(s[10]-s[9]);b=float(s[11]-s[8])
    return dict(zip(OBSERVABLES,[a,b,a/scale,b/scale,*[int((np.abs(p-s[9])<=f*scale).sum()) for f in (.01,.02,.05)]]))


def quintiles(x,count=5):
    x=np.asarray(x,dtype=float);edges=np.unique(np.quantile(x,np.linspace(0,1,count+1)))
    return np.searchsorted(edges[1:-1],x,side='right')


def residual_stats(x):
    a=np.asarray(x,dtype=float);v=np.abs(a)
    return {'count':len(a),'signed_mean':float(a.mean()),'mae':float(v.mean()),'std':float(a.std()),
            'variance':float(a.var()),'abs_p50':float(np.quantile(v,.5)),
            'abs_p90':float(np.quantile(v,.9)),'abs_p95':float(np.quantile(v,.95))}
