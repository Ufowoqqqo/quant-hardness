#!/usr/bin/env python3
"""Independent saved-ID audit, sampled full regeneration and PQ-only bridge audit."""
import argparse
import json
from pathlib import Path
import numpy as np
from analyze_phase3a_sift1m import read_conf
from analyze_phase3b_fixed_candidate_ranking import dump,digest
from analyze_phase3c_precision_transition import jsonl,read_candidates
from phase3g_metrics import evaluate
from phase3h_metrics import CONDITIONS,conditioned_permutations,verify_assignment,OBSERVABLES


def verify(config,run):
    cfg=read_conf(config);provenance=json.loads((run/'provenance.json').read_text());P=int(cfg['permutations']);k=int(cfg['k']);records=[]
    for p,h in provenance['input_sha256'].items():assert digest(p)==h,p
    for p,h in provenance['source_sha256'].items():assert digest(p)==h,p
    features=list(jsonl(run/'bridge/observables.jsonl.gz'));agg=json.loads(Path('results/tables/phase3h_aggregate.json').read_text())
    for mi,name in enumerate(cfg['models'].split(',')):
        rows=list(jsonl(run/name/'queries.jsonl.gz'));oracle=np.array([r['oracle_recall'] for r in rows])
        score=np.memmap(Path(cfg['model_source'])/name/'scores.f32',dtype='<f4',mode='r')
        for c in CONDITIONS:
            with np.load(run/name/f'{c}.npz') as f:raw={key:f[key] for key in f.files}
            replayed=0;negative=0;offset=0
            for meta,a in read_candidates(Path(cfg['candidate_source']),jsonl(Path(cfg['candidate_source'])/'queries.jsonl')):
                q=meta['query_id'];r=rows[q];ids=a['id'];d=a['exact'].astype(float);n=len(a);p=score[offset:offset+n].astype(float);offset+=n
                chosen=raw['topk'][q];truth=r['ground_truth_ids'];exact=r['exact_topk_ids'];nstat=r['nulls'][c]
                assert np.isin(chosen,ids).all() and np.all(np.diff(np.sort(chosen,axis=1),axis=1)!=0)
                hits=np.isin(chosen,truth).sum(1);agree=np.isin(chosen,exact).sum(1);lh=round(r['oracle_recall']*k)-hits
                for key,value in [('hits',hits),('agreement_hits',agree),('displaced_count',k-agree),('loss_hits',lh)]:np.testing.assert_array_equal(raw[key][q],value)
                assert nstat['loss']['mean']==int(lh.sum())/(P*k)
                assert nstat['loss']['std']==float((lh/k).std(ddof=1))
                assert nstat['excess_loss']==(round(r['observed_loss']*k)*P-int(lh.sum()))/(P*k)
                assert nstat['inversions']['mean']==float(raw['inversions'][q].mean())
                bad=np.where(lh<0)[0];negative+=len(bad)
                if len(bad):
                    assert r['exact_boundary_tie'];boundary=float(np.sort(d)[k-1]);lookup=dict(zip(ids.tolist(),d.tolist()))
                    for j in bad:
                        gained=(set(chosen[j])&set(truth))-set(exact)
                        assert gained and all(lookup[v]==boundary for v in gained)
                if q%int(cfg['replay_query_stride']):continue
                order=np.argsort(d,kind='stable');idx=conditioned_permutations(order,P,int(cfg['conditioned_seed']),mi,q,c,int(cfg['unrestricted_seed']))
                verify_assignment(idx,order,c);null=evaluate(ids,d,d+(p-d)[idx],truth,k)
                for key,value in null.items():np.testing.assert_array_equal(value,raw[key][q])
                replayed+=P
            for scope,mask in [('all',np.ones(len(rows),dtype=bool)),('oracle1',oracle==1)]:
                vals={'loss':raw['loss_hits'][mask].mean()/k,'recall':raw['hits'][mask].mean()/k,'inversions':raw['inversions'][mask].mean(),
                    'agreement':raw['agreement_hits'][mask].mean()/k,'harmful_fraction':(raw['loss_hits'][mask]>0).mean()}
                for key,v in vals.items():
                    expected=next(r for r in agg if r['model']==name and r['scope']==scope and r['condition']==c and r['metric']==key)
                    assert abs(expected['mean']-v)<1e-12
            records.append({'model':name,'condition':c,'saved_queries_audited':len(rows),'replayed_permutations':replayed,
                'negative_losses_retained':negative,'negative_losses_only_gt_boundary_tie_substitutions':True})
            print(f'{name} {c}: saved-ID audit and {replayed} replays PASS',flush=True)
        ff=[f for f in features if f['model']==name];offset=0
        for q,r in enumerate(rows):
            p=score[offset:offset+r['candidate_count']].astype(float);offset+=len(p);s=sorted(p.tolist());scale=max(abs(s[9]),float(cfg['epsilon']))
            expected=[s[10]-s[9],s[11]-s[8],(s[10]-s[9])/scale,(s[11]-s[8])/scale,
                *[sum(abs(v-s[9])<=f*scale for v in p) for f in (.01,.02,.05)]]
            assert ff[q]['query_id']==q
            for key,v in zip(OBSERVABLES,expected):assert ff[q][key]==v
    for p,h in provenance['input_sha256'].items():assert digest(p)==h,p
    dump(run/'verification.json',{'status':'PASS','records':records,'pq_only_observables_scalar_reference_all50000':True,
        'all_frozen_inputs_unchanged':True,'total_sampled_replays':sum(r['replayed_permutations'] for r in records),
        'total_saved_samples_audited':len(records)*int(cfg['queries'])*P})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=Path('configs/indexes/phase3h_rank_conditioned_nulls.conf'))
    p.add_argument('--run',type=Path,default=Path('runs/phase3h_rank_conditioned_nulls_v1'))
    a=p.parse_args();verify(a.config,a.run)
