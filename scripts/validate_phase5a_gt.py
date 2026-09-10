"""Validate 100 frozen samples: separate C++ FP64 exhaustive L2 and cosine.

ID Recall@10 is never adjusted. Disagreements must be tolerance-boundary
ties (<=2e-5 distance); retain their IDs and measured deltas explicitly.
"""
import json
from pathlib import Path
import numpy as np
from prepare_phase5a import digest


def main():
    root=Path('runs/phase5a_highdim_external_validity_v1'); prep=root/'prepared'
    assert not (root/'gt_validation.json').exists()
    sample=np.fromfile(prep/'gt_validation_query_ids.i64',dtype='<i8')
    assert len(sample)==100 and len(np.unique(sample))==100
    gt=np.fromfile(root/'gt_ids.i64',dtype='<i8').reshape(10000,11)
    gd=np.fromfile(root/'gt_distances.f64',dtype='<f8').reshape(10000,11)
    ref=np.fromfile(root/'gt-reference_ids.i64',dtype='<i8').reshape(100,11)
    rd=np.fromfile(root/'gt-reference_distances.f64',dtype='<f8').reshape(100,11)
    assert np.all(np.diff(gd,axis=1)>=0) and np.all((gt>=0)&(gt<990000))
    assert all(len(set(row))==11 for row in gt)
    base=np.memmap(prep/'base.f32',dtype='<f4',mode='r',shape=(990000,1536))
    query=np.memmap(prep/'query.f32',dtype='<f4',mode='r',shape=(10000,1536))
    q=np.asarray(query[sample],dtype=np.float64); qnorm=np.linalg.norm(q,axis=1)
    best=np.empty((100,0)); bestids=np.empty((100,0),dtype=np.int64)
    for start in range(0,len(base),8192):
        block=np.asarray(base[start:start+8192],dtype=np.float64)
        cosine_l2=2-2*(q@block.T)/(qnorm[:,None]*np.linalg.norm(block,axis=1)[None,:])
        cand=np.concatenate([best,cosine_l2],axis=1)
        ids=np.concatenate([bestids,np.broadcast_to(np.arange(start,start+len(block)),(100,len(block)))],axis=1)
        take=np.argpartition(cand,10,axis=1)[:,:11]
        best=np.take_along_axis(cand,take,axis=1); bestids=np.take_along_axis(ids,take,axis=1)
    cosine_ids=np.empty_like(bestids)
    for i in range(100): cosine_ids[i]=bestids[i,np.lexsort((bestids[i],best[i]))]
    cosine_ids.astype('<i8').tofile(root/'gt-cosine-reference_ids.i64')
    disagreements=[]; max_error=0
    for i,qi in enumerate(sample):
        union=np.unique(np.r_[gt[qi],ref[i],cosine_ids[i]])
        x=np.asarray(base[union],dtype=np.float64)
        exact=np.sum((x-q[i])**2,axis=1)
        lookup=dict(zip(union,exact))
        err=max(abs(gd[qi,j]-lookup[gt[qi,j]]) for j in range(11));max_error=max(max_error,float(err))
        assert err<=2e-5
        for label,other in [('fp64_l2',ref[i]),('fp64_cosine',cosine_ids[i])]:
            missing=set(gt[qi,:10])-set(other[:10]); extra=set(other[:10])-set(gt[qi,:10])
            if missing or extra:
                distances=[lookup[j] for j in missing|extra]
                spread=max(distances)-min(distances)
                assert spread<=2e-5,(qi,label,spread)
                disagreements.append(dict(query_id=int(qi),reference=label,missing=sorted(map(int,missing)),
                                          extra=sorted(map(int,extra)),fp64_distance_spread=spread))
    result=dict(status='PASS',queries=100,absolute_distance_tolerance=2e-5,
                max_primary_fp32_vs_fp64_distance_error=max_error,
                tolerance_boundary_disagreements=disagreements,
                exact_fp32_rank10_rank11_tie_queries=int(np.sum(gd[:,9]==gd[:,10])),
                reference_note='independent scalar FP64 squared-L2 exhaustive C++ plus NumPy FP64 cosine exhaustive',
                recall_semantics='strict primary FP32 GT ID intersection /10; no tolerance recall substitution',
                sha256={p.name:digest(p) for p in root.glob('gt*.*') if p.is_file()})
    with (root/'gt_validation.json').open('x') as f:json.dump(result,f,indent=2)
    print('GT PASS',len(disagreements),'documented tolerance-boundary disagreements',flush=True)


if __name__=='__main__':main()
