"""Validate saved per-query IDs/metrics and diagnose exact-control ties."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',default='runs/phase5a_highdim_external_validity_v1');args=parser.parse_args()
    root=Path(args.root)
    assert not (root/'recall_audit.json').exists()
    gt=np.fromfile(root/'gt_ids.i64',dtype='<i8').reshape(10000,11)[:,:10]
    base=np.memmap(root/'prepared/base.f32',dtype='<f4',mode='r',shape=(990000,1536))
    queries=np.memmap(root/'prepared/query.f32',dtype='<f4',mode='r',shape=(10000,1536))
    controls=[];fingerprints=[]
    for ef in [32,64,128]:
        path=root/f'recall_ef{ef}'; meta=json.loads((path/'complete.json').read_text());assert meta['status']=='PASS'
        fingerprints.append(meta['graph_fingerprint'])
        rows=np.genfromtxt(path/'per_query.csv',delimiter=',',names=True)
        np.testing.assert_array_equal(rows['query_id'],np.arange(10000))
        values={}
        for name,field in [('native','recall_pq'),('exact','recall_exact'),('all16','recall_all16'),('oracle','recall_candidate_oracle'),('exact_oracle','recall_exact_oracle')]:
            ids=np.fromfile(path/(name+'_ids.i64'),dtype='<i8').reshape(10000,10);values[name]=ids
            assert np.all((ids>=0)&(ids<990000)) and all(len(set(r))==10 for r in ids)
            computed=np.array([np.isin(a,b).sum()/10 for a,b in zip(ids,gt)])
            np.testing.assert_allclose(rows[field],computed,rtol=0,atol=1e-14)
        top=np.fromfile(path/'top16_ids.i64',dtype='<i8').reshape(10000,16)
        offsets=np.fromfile(path/'pq_offsets.i64',dtype='<i8')
        candidates=np.memmap(path/'pq_candidates.i32',dtype='<i4',mode='r');scores=np.memmap(path/'pq_scores.f32',dtype='<f4',mode='r')
        assert len(offsets)==10001 and offsets[0]==0 and offsets[-1]==len(candidates)==len(scores)
        for qi in range(10000):
            ids=candidates[offsets[qi]:offsets[qi+1]];s=scores[offsets[qi]:offsets[qi+1]]
            assert len(np.unique(ids))==len(ids) and np.isfinite(s).all()
            expected=ids[np.lexsort((np.arange(len(ids)),s))[:16]]
            np.testing.assert_array_equal(top[qi],expected)
            assert np.isin(values['all16'][qi],top[qi]).all()
            assert np.isin(values['native'][qi],ids).all()
            assert np.isin(values['oracle'][qi],ids).all()
            assert abs(rows['coverage_pq'][qi]-np.isin(gt[qi],ids).sum()/10)<1e-14
        for qi in np.flatnonzero(np.abs(rows['delta_exact_control'])>1e-12):
            exact=values['exact'][qi];oracle=values['exact_oracle'][qi]
            ex=np.sum((base[exact].astype('f8')-queries[qi])**2,axis=1)
            ore=np.sum((base[oracle].astype('f8')-queries[qi])**2,axis=1)
            err=float(np.max(np.abs(np.sort(ex)-np.sort(ore))))
            assert err<=2e-5,('non-tie exact control anomaly',ef,qi,err)
            controls.append(dict(ef=ef,query_id=int(qi),delta=float(rows['delta_exact_control'][qi]),
                                 max_sorted_fp64_distance_difference=err,classification='numerical/boundary tie; strict ID metric preserved'))
        np.testing.assert_allclose(rows['total_loss'],rows['discovery_loss']+rows['ranking_loss'],atol=1e-14,rtol=0)
        np.testing.assert_allclose(rows['recall_candidate_oracle'],rows['coverage_pq'],atol=1e-14,rtol=0)
        np.testing.assert_allclose(rows['recall_exact_oracle'],rows['coverage_exact'],atol=1e-14,rtol=0)
    assert len(set(fingerprints))==1
    assert fingerprints[0]==json.loads((root/'graph.json').read_text())['fingerprint']
    with (root/'recall_audit.json').open('x') as f:json.dump(dict(status='PASS',queries_per_ef=10000,
        graph_fingerprint=fingerprints[0],full_offline_top16_identity=True,
        exact_control_anomalies=controls,strict_ID_recall_preserved=True),f,indent=2)
    print('RECALL AUDIT PASS; exact-control tie cases:',len(controls))


if __name__=='__main__':main()
