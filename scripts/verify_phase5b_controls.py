"""Post-run frozen-input and independently sorted candidate-rank audit."""
import csv
import json
from pathlib import Path
import numpy as np
from prepare_phase5b import sha


def main():
    root=Path('runs/phase5b_compression_robustness_v1');old=Path('runs/phase5a_highdim_external_validity_v1')
    inputs=json.loads((root/'input_verification.json').read_text())
    for path,digest in inputs['sha256'].items():assert sha(path)==digest,path
    freeze=json.loads((root/'analysis_freeze.json').read_text())
    for path,digest in freeze.items():assert sha(path)==digest,path
    gt=np.fromfile(root/'gt_ids.i64',dtype='<i8').reshape(10000,11)[:,:10]
    rank_rows=iter(csv.DictReader((root/'candidate_oracle_item_ranks.csv').open()))
    controls=[];count=0
    for ef in (32,64,128):
        path=root/f'recall_ef{ef}'
        rows=np.genfromtxt(path/'per_query.csv',delimiter=',',names=True)
        np.testing.assert_array_equal(rows['recall_exact_oracle'],rows['coverage_exact'])
        np.testing.assert_array_equal(rows['recall_candidate_oracle'],rows['coverage_pq'])
        assert np.all(rows['delta_exact_control']==0)
        for name in ['exact_ids.i64','exact_oracle_ids.i64']:
            assert (path/name).read_bytes()==(old/f'recall_ef{ef}'/name).read_bytes()
        offsets=np.fromfile(path/'pq_offsets.i64',dtype='<i8')
        candidates=np.memmap(path/'pq_candidates.i32',mode='r',dtype='<i4')
        scores=np.memmap(path/'pq_scores.f32',mode='r',dtype='<f4')
        oracle=np.fromfile(path/'oracle_ids.i64',dtype='<i8').reshape(10000,10)
        excluded=0
        for qi in range(10000):
            ids=candidates[offsets[qi]:offsets[qi+1]];s=scores[offsets[qi]:offsets[qi+1]]
            # Different sorting expression than primary lexsort: stable argsort.
            order=np.argsort(s,kind='stable');lookup={int(ids[j]):r+1 for r,j in enumerate(order)}
            outside=0
            for j,item in enumerate(oracle[qi]):
                r=next(rank_rows);count+=1
                expected=dict(ef=ef,query_id=qi,oracle_rank=j+1,candidate_id=int(item),pq_rank=lookup[int(item)],
                    gt_relevant=int(item in gt[qi]),candidate_count=len(ids))
                assert {k:int(v) for k,v in r.items()}==expected
                outside+=int(expected['gt_relevant'] and expected['pq_rank']>16)
            assert abs((rows['recall_candidate_oracle'][qi]-rows['recall_all16'][qi])-outside/10)<1e-12
            excluded+=outside
        loss=rows['discovery_loss']
        controls.append(dict(ef=ef,GT_relevant_oracle_items_outside_top16=excluded,
            mean_positive_discovery_mass=float(np.maximum(loss,0).mean()),
            mean_negative_discovery_mass=float(np.minimum(loss,0).mean())))
    assert next(rank_rows,None) is None and count==300000
    result=dict(status='PASS',all_frozen_input_hashes_unchanged=len(inputs['sha256']),
        exact_oracle_coverage_equal_all_queries=True,exact_controls_equal_phase5a=True,
        independently_validated_rank_rows=count,all16_residual_gap_equals_excluded_GT_items_per_query=True,
        controls=controls,script_sha256=sha(__file__))
    with (root/'postrun_controls.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
