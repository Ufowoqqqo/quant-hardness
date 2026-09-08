#!/usr/bin/env python3
"""Execute top-L FP32 distances afresh: exact candidate score cache is NOT read."""
import gzip
import json
from pathlib import Path
import numpy as np
from analyze_phase3a_sift1m import read_conf
from analyze_phase3b_fixed_candidate_ranking import dump
from prepare_phase4a import fvecs

c=read_conf(Path('configs/indexes/phase4a_gap_guided_refinement.conf'));root=Path(c['run'])
assert (root/'analysis/primary_checkpoint.json').exists()
base=fvecs(c['base_path']);queries=np.memmap(root/'queries.f32',dtype='<f4',mode='r').reshape(-1,128)
offsets=np.fromfile(root/'candidate_offsets.i64',dtype='<i8');ids=np.memmap(root/'candidate_ids.i64',dtype='<i8',mode='r');pq=np.memmap(root/'candidate_pq.f32',dtype='<f4',mode='r')
native=np.fromfile(root/'pq_native_ids.i64',dtype='<i8').reshape(-1,10);gt=np.fromfile(root/'gt_ids.i64',dtype='<i8').reshape(-1,11)
selection=np.load(root/'analysis/selections.npz');rows=[json.loads(l) for l in gzip.open(root/'analysis/queries.jsonl.gz','rt')]
records=[]
for L in [16,32,64]:
    for fraction in [.1,.25,.5]:
        for policy in ['GAP','ORACLE-QUERY','RANDOM-replicate-0']:
            count=int(len(queries)*fraction)
            selected=selection[f'all_L{L}_f{fraction}_{policy}'] if policy!='RANDOM-replicate-0' else selection['all_random_order_0'][:count]
            returned=native.copy();evaluations=0
            for qi in selected:
                sl=slice(offsets[qi],offsets[qi+1]);pool=ids[sl]
                # Selection reads PQ scores only; exactly L vector distances follow.
                local=np.sort(np.argsort(pq[sl],kind='stable')[:L]);candidate_ids=pool[local]
                diff=np.subtract(base[candidate_ids],queries[qi],dtype=np.float32)
                distances=np.sum(np.multiply(diff,diff,dtype=np.float32),axis=1,dtype=np.float32)
                evaluations+=len(candidate_ids)
                returned[qi]=candidate_ids[np.argsort(distances,kind='stable')[:10]]
                assert returned[qi].tolist()==rows[qi][f'top{L}_ids']
            mask=np.ones(len(queries),dtype=bool);mask[selected]=False
            assert np.array_equal(returned[mask],native[mask]);assert evaluations==count*L
            recall=sum(len(set(r)&set(t[:10])) for r,t in zip(returned,gt))/(10*len(queries))
            records.append({'L':L,'fraction':fraction,'policy':policy,'refined_queries':count,'actual_FP32_distance_calls':evaluations,
                            'actual_FP32_distance_calls_per_query':evaluations/len(queries),'recall':recall,'native_unchanged_unselected':True})
dump(root/'physical_refinement_validation.json',{'status':'PASS','method':'fresh FP32 direct squared differences on exactly selected top-L base rows; no candidate_exact cache read',
    'scope':'all 9 primary budgets, GAP and offline ORACLE plus RANDOM replicate0; all100 RANDOM repetitions separately scalar-audited',
    'wallclock_benchmark':False,'records':records})
print('PASS',len(records),'fresh-distance execution controls;',sum(r['actual_FP32_distance_calls'] for r in records),'counted FP32 distances')
