"""Transparent repeated-run scaling summaries (no latency from throughput)."""
import numpy as np

def latency(a):
    a=np.asarray(a,float)
    if not len(a) or not np.isfinite(a).all() or (a<=0).any():raise ValueError('invalid latency')
    return {'mean_latency_us':float(a.mean()),'median_latency_us':float(np.median(a)),
            **{f'p{p}_latency_us':float(np.quantile(a,p/100)) for p in [90,95,99]}}

def summarize(a):
    a=np.asarray(a,float)
    if not len(a) or not np.isfinite(a).all():raise ValueError('invalid samples')
    return {'mean':float(a.mean()),'std':float(a.std(ddof=1)) if len(a)>1 else 0.,'min':float(a.min()),'max':float(a.max())}

def scaling(qps,baseline,workers):
    if baseline<=0 or qps<=0 or workers<1:raise ValueError('invalid throughput')
    speedup=qps/baseline
    return {'speedup':speedup,'parallel_efficiency':speedup/workers}

def relative(native,all16):
    if min(native['qps'],all16['qps'])<=0:raise ValueError('invalid QPS')
    r={'throughput_penalty':1-all16['qps']/native['qps']}
    for name in ['mean','p95','p99']:
        key=name+'_latency_us'
        if native[key]<=0:raise ValueError('invalid latency')
        r[name+'_latency_overhead']=all16[key]/native[key]-1
    r['p99_ratio']=all16['p99_latency_us']/native['p99_latency_us']
    return r
