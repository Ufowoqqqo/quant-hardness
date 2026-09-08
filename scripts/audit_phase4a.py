#!/usr/bin/env python3
"""Scalar audit of raw outcomes/selection budgets plus bytewise regeneration."""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np
from analyze_phase3a_sift1m import read_conf
from analyze_phase3b_fixed_candidate_ranking import digest,dump
from analyze_phase4a import arrays


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reproduction',required=True);args=parser.parse_args()
    c=read_conf(Path('configs/indexes/phase4a_gap_guided_refinement.conf'));root=Path(c['run']);repro=Path(args.reproduction)
    n=int(c['query_count']);a=arrays(root,n)
    rows=[json.loads(line) for line in gzip.open(root/'analysis/queries.jsonl.gz','rt')]
    assert [r['query_id'] for r in rows]==list(range(n))
    tests=0;negative_loss_ties=[]
    for qi in range(n):
        sl=slice(a['offset'][qi],a['offset'][qi+1]);ids=list(map(int,a['ids'][sl]));dp=list(map(float,a['pq'][sl]));de=list(map(float,a['exact'][sl]))
        order=sorted(range(len(ids)),key=lambda j:(dp[j],j));truth=set(map(int,a['gt'][qi,:10]));row=rows[qi]
        assert row['g9_12_pq']==dp[order[11]]-dp[order[8]]
        assert row['native_recall']==len(set(row['native_ids'])&truth)/10
        oracle=sorted(range(len(ids)),key=lambda j:(de[j],j))[:10]
        assert row['oracle_ids']==[ids[j] for j in oracle]
        if row['ranking_recall_loss']<0:
            from phase4a_tie_sensitivity import tie_hits
            lookup=dict(zip(ids,de));gt=a['gd'][qi]
            assert tie_hits([lookup[i] for i in row['oracle_ids']],gt)>=tie_hits([lookup[i] for i in row['native_ids']],gt)
            exchanged=(set(row['native_ids'])&truth)-set(row['oracle_ids'])
            assert all(lookup[i]==gt[9] for i in exchanged)
            negative_loss_ties.append(qi)
        for L in [16,32,64]:
            reference=sorted(order[:L],key=lambda j:(de[j],j))[:10]
            assert row[f'top{L}_ids']==[ids[j] for j in reference]
            assert row[f'top{L}_recall']==len(set(row[f'top{L}_ids'])&truth)/10
        tests+=1
    policies=json.loads(Path('results/tables/phase4a_policies.json').read_text())
    sel=np.load(root/'analysis/selections.npz');policy_checks=0
    for r in policies:
        if r['policy']=='RANDOM':continue
        scope=r['scope'];ix=list(range(n)) if scope=='all' else list(range(int(scope[-1]),n,4));L=r['L'];count=int(r['fraction']*len(ix))
        chosen=sel[f"{scope}_L{L}_f{r['fraction']}_{r['policy']}"]
        assert len(chosen)==count and len(set(chosen))==count
        if r['policy']=='GAP':expected=sorted(ix,key=lambda i:(rows[i]['g9_12_pq'],i))[:count]
        elif r['policy']=='ORACLE-QUERY':expected=sorted(ix,key=lambda i:(-round(10*(rows[i][f'top{L}_recall']-rows[i]['native_recall'])),i))[:count]
        else:expected=sorted(ix,key=lambda i:(-round(10*rows[i]['ranking_recall_loss']),i))[:count]
        assert list(chosen)==expected
        chosen=set(map(int,chosen));hits=sum(round(10*rows[i][f'top{L}_recall' if i in chosen else 'native_recall']) for i in ix)
        assert abs(r['recall']-hits/(10*len(ix)))<1e-14
        assert r['exact_distance_evals_per_query']==count*L/len(ix)
        policy_checks+=1
    draws=json.loads(Path('results/tables/phase4a_random_draws.json').read_text())
    for r in draws:
        scope=r['scope'];ix=list(range(n)) if scope=='all' else list(range(int(scope[-1]),n,4));L=r['L'];count=int(r['fraction']*len(ix))
        order=sel[f"{scope}_random_order_{r['replicate']}"];assert set(order)==set(ix)
        chosen=set(map(int,order[:count]));hits=sum(round(10*rows[i][f'top{L}_recall' if i in chosen else 'native_recall']) for i in ix)
        assert abs(r['recall']-hits/(10*len(ix)))<1e-14
        assert r['exact_distance_evals_per_query']==count*L/len(ix)
    # Inputs and frozen design must remain identical, independently of analysis.
    for manifest,key in [('provenance.json','sha256'),('validation.json','export_sha256')]:
        for name,h in json.loads((root/manifest).read_text())[key].items():assert digest(name)==h,name
    regenerated=[]
    for dirname in ['tables','figures']:
        for p in sorted((Path('results')/dirname).glob('phase4a_*')):
            rp=repro/dirname/p.name;assert digest(p)==digest(rp),str(p);regenerated.append(str(p))
    assert gzip.open(root/'analysis/queries.jsonl.gz','rb').read()==gzip.open(repro/'analysis/queries.jsonl.gz','rb').read()
    other=np.load(repro/'analysis/selections.npz');assert set(sel.files)==set(other.files)
    for k in sel.files:assert np.array_equal(sel[k],other[k])
    files=list(root.rglob('*'))+list(Path('results/tables').glob('phase4a_*'))+list(Path('results/figures').glob('phase4a_*'))
    files += [Path('build/faiss_phase4a'),Path('CMakeLists.txt')]+list(Path('scripts').glob('*phase4a*.py'))+[Path('src/experiments/faiss_phase4a.cpp'),Path('tests/test_phase4a_metrics.py')]
    hashes={str(p):digest(p) for p in files if p.is_file() and p.name not in ['audit.log','audit_final.log','final_audit.json']}
    dump(root/'final_audit.json',{'status':'PASS','scalar_query_checks':tests,'scalar_policy_checks':policy_checks,
        'random_budget_replicates_checked':len(draws),'byte_identical_tables_figures':regenerated,
        'query_rows_and_selection_arrays_regenerated':True,'negative_loss_verified_boundary_ties':negative_loss_ties,'artifact_sha256':hashes})
    print('PASS',tests,'scalar queries;',policy_checks,'policy points;',len(draws),'random budgets;',len(regenerated),'regenerated tables/figures')


if __name__=='__main__':main()
