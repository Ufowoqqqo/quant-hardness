#!/usr/bin/env python3
"""Reference hierarchical rank nulls over immutable saved FP32 candidate pools."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
import numpy as np
from analyze_phase3a_sift1m import read_conf
from analyze_phase3b_fixed_candidate_ranking import dump,digest
from analyze_phase3c_precision_transition import jsonl,read_candidates,gz_writer
from phase3g_metrics import evaluate,scalar_reference,summary
from phase3h_metrics import CONDITIONS,conditioned_permutations,verify_assignment,residual_stats


def initialize(config,run):
    assert not (run/'provenance.json').exists();cfg=read_conf(config);prior=Path(cfg['phase3g_source'])
    old=json.loads((prior/'provenance.json').read_text());hashes=dict(old['input_sha256'])
    audit=json.loads((prior/'final_audit.json').read_text())['sha256']
    for name in cfg['models'].split(','):
        for file in ('permutations.npz','queries.jsonl.gz'):
            p=prior/name/file;hashes[str(p)]=audit[str(p)]
    hashes[str(prior/'provenance.json')]=digest(prior/'provenance.json')
    for p,h in hashes.items():assert digest(p)==h,p
    sources=[config,run/'preregistration.md',Path(__file__),Path('scripts/phase3h_metrics.py'),Path('tests/test_phase3h_metrics.py'),
        Path('scripts/phase3g_metrics.py'),Path('scripts/analyze_phase3c_precision_transition.py')]
    dump(run/'provenance.json',{'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'git_status':subprocess.check_output(['git','status','--short'],text=True),'config':cfg,
        'utc_start':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'machine':platform.uname()._asdict(),
        'cpu':subprocess.check_output(['lscpu'],text=True),'numpy':np.__version__,'input_sha256':hashes,
        'source_sha256':{str(p):digest(p) for p in sources},'model_manifests':old['models'],
        'dataset_search_config':old['dataset_search_config'],'no_training_no_search':True})
    (run/'resolved_config.conf').write_bytes(config.read_bytes())


def residual_diagnostics(d,e,ranks,distance_bins):
    records=[]
    groups=[(f'rank_{j}',ranks==j) for j in range(1,31)]
    groups += [(label,(ranks>=lo)&(ranks<=hi)) for label,lo,hi in
        [('rank31_40',31,40),('rank41_100',41,100),('rank101plus',101,int(ranks.max())),
         ('rank1_5',1,5),('rank6_10',6,10),('rank1_10',1,10),('rank11_20',11,20),('rank21plus',21,int(ranks.max()))]]
    for label,mask in groups:
        if mask.any():records.append({'group':label,**residual_stats(e[mask])})
    edges=np.unique(np.quantile(d,np.linspace(0,1,distance_bins+1)));labels=np.searchsorted(edges[1:-1],d,side='right')
    for b in sorted(set(labels)):
        mask=labels==b;records.append({'group':f'pooled_distance_quantile_{b+1}',**residual_stats(e[mask]),
            'distance_min':float(d[mask].min()),'distance_max':float(d[mask].max()),'distance_mean':float(d[mask].mean())})
    return records


def model_worker(config,run,name):
    cfg=read_conf(config);mi=cfg['models'].split(',').index(name);nq=int(cfg['queries']);P=int(cfg['permutations']);k=int(cfg['k'])
    out=run/name;out.mkdir(exist_ok=False);start=time.monotonic();offset=0
    geometry=list(jsonl(Path(cfg['geometry_source'])));previous=list(jsonl(Path(cfg['phase3g_source'])/name/'queries.jsonl.gz'))
    with np.load(Path(cfg['phase3g_source'])/name/'permutations.npz') as f:prefix={key:f[key] for key in f.files}
    score=np.memmap(Path(cfg['model_source'])/name/'scores.f32',dtype='<f4',mode='r')
    exact=np.empty(len(score));ranks=np.empty(len(score),dtype='<i4')
    arrays={c:{key:np.empty((nq,P,k) if key=='topk' else (nq,P),dtype='<i4' if key in ('topk','inversions') else '<i2')
        for key in ('topk','hits','loss_hits','agreement_hits','displaced_count','inversions')} for c in CONDITIONS}
    negative_scores={c:0 for c in CONDITIONS};scalar_checks=0;min_pool=len(score)
    with gz_writer(out/'queries.jsonl.gz') as target:
        for meta,a in read_candidates(Path(cfg['candidate_source']),jsonl(Path(cfg['candidate_source'])/'queries.jsonl')):
            q=meta['query_id'];g=geometry[q];prev=previous[q];ids=a['id'];d=a['exact'].astype(float);n=len(a);p=score[offset:offset+n].astype(float)
            assert hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest()==meta['candidate_order_sha256']==prev['candidate_order_sha256']==g['candidate_order_sha256']
            truth=meta['ground_truth_ids'];assert truth==g['ground_truth_ids']==prev['ground_truth_ids'];order=np.argsort(d,kind='stable')
            assert ids[order[:k]].tolist()==g['exact_topk_ids'];exact[offset:offset+n]=d
            ranks[offset+order]=np.arange(1,n+1);offset+=n;min_pool=min(min_pool,n);e=p-d;assert np.array_equal(d+e,p)
            obs=evaluate(ids,d,p,truth,k)
            assert obs['topk'][0].tolist()==prev['observed_topk_ids'] and obs['loss_hits'][0]/k==prev['observed_loss']
            assert obs['inversions'][0]==prev['observed_inversions'];stats={}
            for c in CONDITIONS:
                idx=conditioned_permutations(order,P,int(cfg['conditioned_seed']),mi,q,c,int(cfg['unrestricted_seed']))
                verify_assignment(idx,order,c);s=d+e[idx];null=evaluate(ids,d,s,truth,k)
                for key in arrays[c]:arrays[c][key][q]=null[key]
                if c=='N0':
                    for key in null:np.testing.assert_array_equal(null[key][:50],prefix[key][q])
                if q%int(cfg['replay_query_stride'])==0:
                    ref=scalar_reference(ids.tolist(),d.tolist(),s[0].tolist(),truth,k)
                    for key in ref:np.testing.assert_array_equal(ref[key],null[key][0])
                    scalar_checks+=1
                negative_scores[c]+=int((s<0).sum());ls=summary(null['loss_hits']/k)
                ls['mean']=float(null['loss_hits'].sum()/(P*k))
                stats[c]={'loss':ls,'inversions':summary(null['inversions']),
                    'mean_recall':float(null['hits'].sum()/(P*k)),'harmful_fraction':float((null['loss_hits']>0).mean()),
                    'mean_agreement':float(null['agreement_hits'].sum()/(P*k)),
                    'excess_loss':float((int(obs['loss_hits'][0])*P-int(null['loss_hits'].sum()))/(P*k))}
            quartiles=[]
            for b in np.array_split(order,4):quartiles.append(residual_stats(e[b]))
            target.write(json.dumps({'query_id':q,'candidate_order_sha256':meta['candidate_order_sha256'],
                'candidate_count':n,'oracle_recall':g['oracle_recall'],'exact_topk_ids':g['exact_topk_ids'],
                'ground_truth_ids':truth,'exact_boundary_tie':g['exact_boundary_tie'],'observed_topk_ids':obs['topk'][0].tolist(),
                'observed_loss':int(obs['loss_hits'][0])/k,'observed_recall':int(obs['hits'][0])/k,
                'observed_inversions':int(obs['inversions'][0]),'observed_agreement':int(obs['agreement_hits'][0])/k,
                'nulls':stats,'distance_quartile_residuals':quartiles},allow_nan=False)+'\n')
            if (q+1)%500==0:print(f'{name}: {q+1}/{nq}; {time.monotonic()-start:.1f}s',flush=True)
    assert offset==len(score)==10353047
    for c,raw in arrays.items():
        with (out/f'{c}.npz').open('xb') as f:np.savez_compressed(f,**raw)
    del arrays
    dump(out/'residual_diagnostics.json',residual_diagnostics(exact,score.astype(float)-exact,ranks,int(cfg['distance_bins'])))
    dump(out/'validation.json',{'status':'PASS','queries':nq,'conditions':CONDITIONS,'permutations':P,
        'n0_first50_all_metrics_and_ids_identical_to_3G':True,'within_bin_assignments_all_checked':True,
        'observed_outputs_and_frozen_pools_match':True,'scalar_reference_checks':scalar_checks,
        'negative_null_scores_not_clipped':negative_scores,'min_candidate_count':min_pool,'elapsed_seconds':time.monotonic()-start})
    return name


def run_all(config,run):
    initialize(config,run);cfg=read_conf(config);start=time.monotonic()
    with ProcessPoolExecutor(max_workers=int(cfg['workers'])) as pool:
        futures=[pool.submit(model_worker,config,run,name) for name in cfg['models'].split(',')]
        for f in futures:print('completed '+f.result(),flush=True)
    for p,h in json.loads((run/'provenance.json').read_text())['input_sha256'].items():assert digest(p)==h,p
    dump(run/'derive_validation.json',{'status':'PASS','all_frozen_input_hashes_unchanged':True,'elapsed_seconds':time.monotonic()-start})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=Path('configs/indexes/phase3h_rank_conditioned_nulls.conf'))
    p.add_argument('--run',type=Path,default=Path('runs/phase3h_rank_conditioned_nulls_v1'))
    a=p.parse_args();run_all(a.config,a.run)
