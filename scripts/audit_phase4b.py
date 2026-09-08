#!/usr/bin/env python3
"""Final immutable-input, raw-trace and reproduction audit."""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np
from analyze_phase3a_sift1m import read_conf
from analyze_phase3b_fixed_candidate_ranking import digest,dump
from phase4b_metrics import mask_replicates

p=argparse.ArgumentParser();p.add_argument('--reproduction',required=True);args=p.parse_args()
c=read_conf(Path('configs/indexes/phase4b_streaming_feasibility.conf'));root=Path(c['run']);repro=Path(args.reproduction);n=int(c['query_count'])
for file,key in [('provenance.json','sha256'),('validation.json','sha256')]:
    for name,h in json.loads((root/file).read_text())[key].items():assert digest(name)==h,name
assert json.loads((root/'benchmark_complete.json').read_text())['status']=='PASS'
assert json.loads((root/'batch_complete.json').read_text())['status']=='PASS'
m=np.fromfile(root/'random_masks.u8',dtype='u1').reshape(30,n);d=np.fromfile(root/'gap_decisions.u8',dtype='u1');g=np.fromfile(root/'gaps.f64',dtype='<f8')
assert np.array_equal(d,g<=710.40625);assert np.array_equal(m,mask_replicates(n,int(d.sum()),94100037,30))
rows=[json.loads(l) for l in gzip.open(root/'analysis/queries.jsonl.gz','rt')];assert [r['query_id'] for r in rows]==list(range(n))
gt=np.fromfile(root/'gt_ids.i64',dtype='<i8').reshape(n,11)
for name,file in [('native','native_ids.i64'),('gap','gap_ids.i64'),('all16','all16_ids.i64'),('oracle','oracle_ids.i64')]:
    ids=np.fromfile(root/file,dtype='<i8').reshape(n,10)
    for qi in range(n):assert rows[qi][name+'_recall']==len(set(ids[qi])&set(gt[qi,:10]))/10
summary=json.loads(Path('results/tables/phase4b_summary.json').read_text())
assert summary['actual_refined_queries']==int(d.sum())
policies=json.loads(Path('results/tables/phase4b_policies.json').read_text())
trace_count=0
for policy in ['NATIVE','GAP','RANDOM','ALL16']:
    means=[]
    for clocks in [0,1]:
        for rep in range(7):
            a=np.genfromtxt(root/'timings'/f'{policy}_r{rep}_c{clocks}.csv',delimiter=',',names=True)
            assert len(a)==n;expected=np.zeros(n) if policy=='NATIVE' else np.ones(n) if policy=='ALL16' else d if policy=='GAP' else m[0]
            assert np.array_equal(a['refined'],expected);assert np.array_equal(a['exact_evals'],expected*16)
            if clocks==0:means.append(float(a['total_us'].sum()/n))
            trace_count+=n
    entry=next(r for r in policies if r['policy']==policy);assert abs(entry['mean_us']-sum(means)/7)<1e-10
    assert entry['exact_evals']==int(expected.sum())*16
regenerated=[]
for folder in ['tables','figures']:
    for file in sorted((Path('results')/folder).glob('phase4b_*')):
        assert digest(file)==digest(repro/folder/file.name),str(file);regenerated.append(str(file))
assert gzip.open(root/'analysis/queries.jsonl.gz','rb').read()==gzip.open(repro/'analysis/queries.jsonl.gz','rb').read()
files=list(root.rglob('*'))+list(Path('results/tables').glob('phase4b_*'))+list(Path('results/figures').glob('phase4b_*'))
files+=list(Path('scripts').glob('*phase4b*.py'))+[Path('src/experiments/faiss_phase4b.cpp'),Path('src/instrumentation/streaming_refinement.h'),Path('src/instrumentation/streaming_refinement.cpp'),Path('tests/test_phase4b_metrics.py'),Path('CMakeLists.txt')]
hashes={str(f):digest(f) for f in files if f.is_file() and f.name not in ['audit.log','final_audit.json']}
dump(root/'final_audit.json',{'status':'PASS','timed_query_traces':trace_count,'policy_recall_query_checks':4*n,
    'random_masks_exactly_regenerated':30,'query_rows_regenerated':True,'byte_identical_tables_figures':regenerated,'artifact_sha256':hashes})
print('PASS',trace_count,'timed traces;',len(regenerated),'byte-identical tables/figures')
