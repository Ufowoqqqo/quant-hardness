"""Fixed Phase4B budget and timing summaries; exploratory, no model fitting."""
import numpy as np


def recall(ids,truth):
    if len(ids)!=10 or len(set(ids))!=10:raise ValueError('unique top10 IDs required')
    return len(set(ids)&set(truth[:10]))/10


def latency_stats(values):
    a=np.asarray(values,dtype=float)
    if not len(a) or not np.isfinite(a).all() or (a<=0).any():raise ValueError('positive finite timings required')
    return {'mean_us':float(a.mean()),'median_us':float(np.median(a)),
            **{f'p{p}_us':float(np.percentile(a,p)) for p in [90,95,99]},'qps':float(1e6/a.mean())}


def systems_metrics(native,gap,all16,gap_latency,all_latency,count,n):
    gain=all16-native
    return {'exact_eval_saving':1-count/n,'latency_saving':1-gap_latency/all_latency,
            'all16_gain_retained':(gap-native)/gain if gain>1e-12 else None}


def mask_replicates(n,count,seed,reps):
    if not 0<=count<=n:raise ValueError('invalid count')
    out=np.zeros((reps,n),dtype=np.uint8)
    for r in range(reps):out[r,np.random.default_rng(np.random.SeedSequence([seed,r])).permutation(n)[:count]]=1
    return out
