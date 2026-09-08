#!/usr/bin/env python3
"""Full saved-ID/summary audit plus deterministic null replay and post-primary centering."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from analyze_phase3a_sift1m import read_conf
from analyze_phase3b_fixed_candidate_ranking import digest, dump
from analyze_phase3c_precision_transition import read_candidates, jsonl
from phase3g_metrics import evaluate, permutations, summary, empirical_rank


def verify(config,run,tables):
    cfg=read_conf(config); assert (tables/'phase3g_summary.json').exists()
    checkpoint=run/'primary_checkpoint.json'
    if checkpoint.exists():
        for path,h in json.loads(checkpoint.read_text())['tables_sha256'].items(): assert digest(path)==h,path
    else:
        dump(checkpoint,{'stage':'primary tables and figures complete before centered diagnostic',
            'tables_sha256':{str(p):digest(p) for p in sorted(tables.glob('phase3g_*'))}})
    provenance=json.loads((run/'provenance.json').read_text()); checks=[]; k=int(cfg['k']); P=int(cfg['permutations'])
    for path,h in provenance['input_sha256'].items(): assert digest(path)==h,path
    for path,h in provenance['source_sha256'].items(): assert digest(path)==h,path
    for mi,name in enumerate(cfg['models'].split(',')):
        with np.load(run/name/'permutations.npz') as archive:
            raw={key:archive[key] for key in archive.files}
        rows=list(jsonl(run/name/'queries.jsonl.gz'))
        score=np.memmap(Path(cfg['model_source'])/name/'scores.f32',dtype='<f4',mode='r'); offset=0; count=0
        centered_order_changes=0; centered_set_changes=0; max_roundoff=0.; negative_losses=0
        for meta,a in read_candidates(Path(cfg['candidate_source']),jsonl(Path(cfg['candidate_source'])/'queries.jsonl')):
            q=meta['query_id']; ids=a['id']; n=len(a); d=a['exact'].astype(float); p=score[offset:offset+n].astype(float); offset+=n
            r=rows[q]; chosen=raw['topk'][q]; exact_ids=r['exact_topk_ids']; truth=r['ground_truth_ids']
            assert hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest()==r['candidate_order_sha256']
            assert np.isin(chosen,ids).all() and np.all(np.diff(np.sort(chosen,axis=1),axis=1)!=0)
            hits=np.isin(chosen,truth).sum(1); agreement=np.isin(chosen,exact_ids).sum(1)
            np.testing.assert_array_equal(hits,raw['hits'][q]); np.testing.assert_array_equal(agreement,raw['agreement_hits'][q])
            np.testing.assert_array_equal(k-agreement,raw['displaced_count'][q])
            np.testing.assert_array_equal(round(r['oracle_recall']*k)-hits,raw['loss_hits'][q])
            assert summary(raw['loss_hits'][q]/k)==r['null_loss'] and summary(raw['inversions'][q])==r['null_inversions']
            assert empirical_rank(round(r['observed_loss']*k),raw['loss_hits'][q])==r['loss_rank']
            assert empirical_rank(r['observed_inversions'],raw['inversions'][q])==r['inversion_rank']
            negative_losses+=int((raw['loss_hits'][q]<0).sum())
            if q%int(cfg['centered_control_query_stride']): continue
            e=p-d; idx=permutations(n,P,int(cfg['permutation_seed']),mi,q); scores=d+e[idx]
            replay=evaluate(ids,d,scores,truth,k)
            for key in replay: np.testing.assert_array_equal(replay[key],raw[key][q])
            centered=d+(e-e.mean())[idx]; control=evaluate(ids,d,centered,truth,k)
            max_roundoff=max(max_roundoff,float(np.max(np.abs(centered-(scores-e.mean())))))
            different=np.any(control['topk']!=replay['topk'],axis=1)
            centered_order_changes+=int(different.sum())
            centered_set_changes+=int(np.any(np.sort(control['topk'],axis=1)!=np.sort(replay['topk'],axis=1),axis=1).sum())
            # The exact score-order change may only split numerical ties, never reverse a non-tied pair.
            for j in np.where(different)[0]:
                order=np.argsort(centered[j],kind='stable')
                assert np.all(np.diff(scores[j,order])>=-1e-9),(name,q,j,'non-tied centered change')
            count+=1
        checks.append({'model':name,'all_query_saved_topk_membership_hits_and_summaries_verified':len(rows),
            'replayed_queries':count,'replayed_permutations':count*P,'centered_order_changes':centered_order_changes,
            'centered_set_changes':centered_set_changes,'centered_max_fp64_roundoff':max_roundoff,
            'centered_non_tied_order_reversals':0,'negative_null_losses_preserved':negative_losses})
        print(f'{name}: full saved-ID audit and deterministic replay PASS',flush=True)
    for path,h in provenance['input_sha256'].items(): assert digest(path)==h,path
    dump(run/'verification.json',{'status':'PASS','models':checks,'all_frozen_input_hashes_unchanged':True,
        'replay_sampling':'query_id % 100 == 0; all50 permutations; all5 models',
        'centered_control_after_primary':True,'primary_checkpoint_sha256':digest(checkpoint),
        'source_sha256':{str(p):digest(p) for p in [Path(__file__),Path(__file__).with_name('analyze_phase3g_residual_permutation.py')]}})
    print(json.dumps(checks,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=Path('configs/indexes/phase3g_residual_permutation.conf'))
    p.add_argument('--run',type=Path,default=Path('runs/phase3g_residual_permutation_v1'))
    p.add_argument('--tables',type=Path,default=Path('results/tables'))
    a=p.parse_args();verify(a.config,a.run,a.tables)
