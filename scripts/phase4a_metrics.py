"""Frozen workload-budget policies. No exact labels enter GAP or RANDOM."""
import numpy as np


def score_gap(scores, epsilon=1e-12):
    s=np.sort(np.asarray(scores,dtype=float),kind='stable')
    if len(s)<12:return float('inf'),float('inf')
    gap=float(s[11]-s[8])
    return gap,gap/max(abs(float(s[9])),epsilon)


def hit_count(ids,truth):
    ids=list(map(int,ids));truth=list(map(int,truth))
    if len(ids)!=10 or len(set(ids))!=10:raise ValueError('unique top10 required')
    return len(set(ids)&set(truth))


def refinement(ids,pq,exact,L,k=10):
    ids=np.asarray(ids);pq=np.asarray(pq);exact=np.asarray(exact)
    if len(ids)<L or L<k:raise ValueError('insufficient candidates for fixed cost')
    selected=np.argsort(pq,kind='stable')[:L]
    # Tie precedence is original candidate evaluation order, not PQ order.
    selected=np.sort(selected)
    result=selected[np.argsort(exact[selected],kind='stable')[:k]]
    return ids[result],selected


def selector(gaps,count):
    return np.argsort(gaps,kind='stable')[:count]


def oracle_selector(gains,count):
    return np.argsort(-np.asarray(gains),kind='stable')[:count]


def random_order(n,seed,scope,replicate):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed,scope,replicate]))).permutation(n)


def ratio(a,b,epsilon=1e-12):
    return float(a/b) if b>epsilon else None


def policy_value(native_hits,refined_hits,selection,L):
    native_hits=np.asarray(native_hits);refined_hits=np.asarray(refined_hits)
    selection=np.asarray(selection,dtype=int)
    if len(np.unique(selection))!=len(selection):raise ValueError('duplicate selected queries')
    n=len(native_hits);gain=float((refined_hits[selection]-native_hits[selection]).sum()/(10*n))
    return {'recall':float(native_hits.mean()/10+gain),'recall_gain':gain,
            'exact_distance_evals_per_query':len(selection)*L/n}
