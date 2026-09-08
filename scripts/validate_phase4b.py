#!/usr/bin/env python3
"""Validate frozen inputs, exact distances/GT; derive content-independent masks."""
import json
from pathlib import Path
import numpy as np
from analyze_phase3a_sift1m import read_conf
from analyze_phase3b_fixed_candidate_ranking import digest,dump
from prepare_phase4a import fvecs
from phase4b_metrics import mask_replicates

c=read_conf(Path('configs/indexes/phase4b_streaming_feasibility.conf'));root=Path(c['run']);n=int(c['query_count'])
assert not (root/'validation.json').exists()
for p,h in json.loads((root/'provenance.json').read_text())['sha256'].items():assert digest(p)==h,p
assert json.loads((root/'reference_validation.json').read_text())['status']=='PASS'
assert json.loads((root/'export_validation.json').read_text())['status']=='PASS'
base=fvecs(c['base_path']);q=np.memmap(root/'queries.f32',dtype='<f4',mode='r').reshape(n,128)
gi=np.fromfile(root/'gt_ids.i64',dtype='<i8').reshape(n,11);gd=np.fromfile(root/'gt_distances.f32',dtype='<f4').reshape(n,11)
sample=np.sort(np.random.default_rng(int(c['gt_seed'])).choice(n,int(c['gt_validation_queries']),replace=False));checks=[]
for qi in sample:
    ds=np.empty(len(base),dtype=float)
    for start in range(0,len(base),10000):
        diff=base[start:start+10000].astype(float)-q[qi].astype(float);ds[start:start+len(diff)]=(diff*diff).sum(1)
    threshold=float(np.partition(ds,9)[9]);ids=gi[qi,:10]
    assert (ds[ids]<=threshold).all() and set(np.flatnonzero(ds<threshold)).issubset(set(ids))
    assert np.array_equal(ds[ids],gd[qi,:10])
    checks.append({'query_id':int(qi),'strict_neighbors':int((ds<threshold).sum()),'boundary_ties':int((ds==threshold).sum()),'status':'PASS'})
offset=np.fromfile(root/'candidate_offsets.i64',dtype='<i8');ids=np.memmap(root/'candidate_ids.i32',dtype='<i4',mode='r');de=np.memmap(root/'candidate_exact.f32',dtype='<f4',mode='r');dp=np.memmap(root/'candidate_pq.f32',dtype='<f4',mode='r')
gap=np.fromfile(root/'gaps.f64',dtype='<f8');decisions=np.fromfile(root/'gap_decisions.u8',dtype='u1');native=np.fromfile(root/'native_ids.i64',dtype='<i8').reshape(n,10);all16=np.fromfile(root/'all16_ids.i64',dtype='<i8').reshape(n,10)
gapids=np.fromfile(root/'gap_ids.i64',dtype='<i8').reshape(n,10);oracle=np.fromfile(root/'oracle_ids.i64',dtype='<i8').reshape(n,10)
for qi in range(n):
    sl=slice(offset[qi],offset[qi+1]);cand=ids[sl];exact=de[sl];approx=dp[sl]
    assert len(set(cand))==len(cand) and len(cand)>=16
    diff=base[cand].astype(float)-q[qi].astype(float);assert np.array_equal((diff*diff).sum(1),exact)
    order=sorted(range(len(cand)),key=lambda j:(float(approx[j]),j))
    assert gap[qi]==float(approx[order[11]])-float(approx[order[8]])
    chosen=sorted(order[:16],key=lambda j:(float(exact[j]),j))[:10]
    assert np.array_equal(all16[qi],cand[chosen]);assert bool(decisions[qi])==(gap[qi]<=float(c['gap_threshold']))
    assert np.array_equal(gapids[qi],all16[qi] if decisions[qi] else native[qi])
    eo=sorted(range(len(cand)),key=lambda j:(float(exact[j]),j))[:10];assert np.array_equal(oracle[qi],cand[eo])
    assert set(native[qi]).issubset(set(cand))
masks=mask_replicates(n,int(decisions.sum()),int(c['random_seed']),int(c['random_replicates']));masks.tofile(root/'random_masks.u8')
dump(root/'gt_validation.json',checks)
files=[p for p in root.iterdir() if p.suffix in ['.i32','.i64','.f32','.f64','.u8']]+[Path('build/faiss_phase4b'),Path('src/instrumentation/streaming_refinement.cpp'),Path('src/experiments/faiss_phase4b.cpp')]
dump(root/'validation.json',{'status':'PASS','all_candidate_FP64_distance_checks':len(ids),'gt_full_base_validation_queries':len(checks),
    'random_replicates':len(masks),'each_random_count':int(decisions.sum()),'mask_generation':'PCG64 SeedSequence([94100037,replicate]), content-independent permutation prefix',
    'sha256':{str(p):digest(p) for p in files}})
print('PASS: full-sort policy references,',len(ids),'exact candidate distances and',len(checks),'independent exhaustive GT queries')
