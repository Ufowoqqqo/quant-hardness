#!/usr/bin/env python3
"""Final raw/aggregate/tie audit and byte-identical analysis reproduction check."""
import argparse
import json
from pathlib import Path
import numpy as np
from analyze_phase3a_sift1m import read_conf
from analyze_phase3b_fixed_candidate_ranking import dump, digest
from analyze_phase3c_precision_transition import jsonl, read_candidates


def audit(config,run,reproduction):
    cfg=read_conf(config); validation=json.loads((run/'verification.json').read_text()); assert validation['status']=='PASS'
    compared=[]
    for kind in ('tables','figures'):
        for path in sorted((Path('results')/kind).glob('phase3g_*')):
            assert digest(path)==digest(reproduction/kind/path.name),path
            compared.append(str(path))
    geometry=list(jsonl(Path(cfg['geometry_source']))); records=[]; aggregate=json.loads(Path('results/tables/phase3g_aggregate.json').read_text())
    for name in cfg['models'].split(','):
        with np.load(run/name/'permutations.npz') as f:
            loss=f['loss_hits'];top=f['topk'];hits=f['hits'];inv=f['inversions']
        negative_q,negative_p=np.where(loss<0); negatives={int(q) for q in negative_q}
        for meta,a in read_candidates(Path(cfg['candidate_source']),jsonl(Path(cfg['candidate_source'])/'queries.jsonl')):
            q=meta['query_id']
            if q not in negatives:continue
            g=geometry[q]; assert g['exact_boundary_tie']; lookup=dict(zip(a['id'].tolist(),a['exact'].tolist()))
            for p in negative_p[negative_q==q]:
                assert loss[q,p]==-1
                gained=(set(top[q,p])&set(g['ground_truth_ids']))-set(g['exact_topk_ids'])
                assert gained and all(lookup[v]==g['dk'] for v in gained)
        for scope in ('all','oracle1'):
            mask=np.ones(len(geometry),dtype=bool) if scope=='all' else np.array([g['oracle_recall']==1 for g in geometry])
            expected={'loss':loss[mask].mean()/10,'recall':hits[mask].mean()/10,'inversions':inv[mask].mean(),'harmful_fraction':(loss[mask]>0).mean()}
            for metric,value in expected.items():
                row=next(r for r in aggregate if r['model']==name and r['scope']==scope and r['metric']==metric)
                assert abs(row['null_mean']-value)<1e-12
        records.append({'model':name,'negative_null_samples':len(negative_q),'negative_queries':len(negatives),
            'all_negative_losses_minus_point_one_and_gt_tie_substitutions':True})
    files=[p for p in run.rglob('*') if p.is_file() and p.name!='final_audit.json']
    files += [Path(p) for p in compared]+[Path(p) for p in ('scripts/phase3g_metrics.py','scripts/run_phase3g_residual_permutation.py',
        'scripts/analyze_phase3g_residual_permutation.py','scripts/verify_phase3g_artifacts.py','scripts/audit_phase3g_final.py',
        'tests/test_phase3g_metrics.py',str(config),'docs/phase3g_residual_permutation.md')]
    dump(run/'final_audit.json',{'status':'PASS','byte_identical_tables_and_figures':len(compared),'reproduction_directory':str(reproduction),
        'negative_loss_tie_audit':records,'independent_aggregate_integer_counts_match':True,
        'sha256':{str(p):digest(p) for p in files}})
    print(json.dumps({'status':'PASS','byte_identical_files':len(compared),'negative_loss_tie_audit':records},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=Path('configs/indexes/phase3g_residual_permutation.conf'))
    p.add_argument('--run',type=Path,default=Path('runs/phase3g_residual_permutation_v1'));p.add_argument('--reproduction',type=Path,required=True)
    a=p.parse_args();audit(a.config,a.run,a.reproduction)
