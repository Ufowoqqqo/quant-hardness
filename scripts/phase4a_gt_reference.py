#!/usr/bin/env python3
"""Independent deterministic direct-FP64 GT validation sample (not a policy)."""
from pathlib import Path
import numpy as np
from analyze_phase3a_sift1m import read_conf
from analyze_phase3b_fixed_candidate_ranking import dump,digest
from prepare_phase4a import fvecs

c=read_conf(Path('configs/indexes/phase4a_gap_guided_refinement.conf'));root=Path(c['run'])
assert not (root/'gt_direct_reference.json').exists()
base=fvecs(c['base_path']);q=np.memmap(root/'queries.f32',dtype='<f4',mode='r').reshape(-1,128)
sample=np.sort(np.random.default_rng(int(c['gt_validation_seed'])).choice(len(q),int(c['gt_validation_queries']),replace=False))
rows=[]
for qi in sample:
    ds=np.empty(len(base),dtype=np.float64)
    for start in range(0,len(base),10000):
        diff=base[start:start+10000].astype(np.float64)-q[qi].astype(np.float64)
        ds[start:start+len(diff)]=(diff*diff).sum(1)
    threshold=float(np.partition(ds,9)[9]);valid=np.flatnonzero(ds<=threshold)
    rows.append({'query_id':int(qi),'distance10':threshold,'ids_at_or_below_d10':valid.tolist(),
                 'distances':ds[valid].tolist(),'strictly_closer_ids':np.flatnonzero(ds<threshold).tolist()})
dump(root/'gt_direct_reference.json',{'rows':rows,'method':'direct FP64 squared difference, full base, deterministic sample',
    'base_sha256':digest(c['base_path']),'queries_sha256':digest(root/'queries.f32')})
print('independent direct FP64 reference completed for',len(rows),'queries')
