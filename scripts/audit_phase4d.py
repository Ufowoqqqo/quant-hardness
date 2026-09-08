#!/usr/bin/env python3
"""Verify all concurrent output IDs, schedules, thread ownership and reproduction."""
import argparse
import json
from pathlib import Path
import numpy as np
from analyze_phase3a_sift1m import read_conf
from analyze_phase3b_fixed_candidate_ranking import digest,dump

p=argparse.ArgumentParser();p.add_argument('--reproduction',required=True);args=p.parse_args()
c=read_conf(Path('configs/indexes/phase4d_multithread_scalability.conf'));root=Path(c['run']);old=Path(c['phase4b']);repro=Path(args.reproduction);n=int(c['query_count'])
prov=json.loads((root/'provenance.json').read_text())
for file,h in prov['immutable_sha256'].items():assert digest(file)==h,file
levels=list(map(int,c['workers'].split(',')));cpus=list(map(int,c['worker_cpus'].split(',')));reps=int(c['repetitions']);passes=int(c['passes_per_repetition'])
conditions=[(t,p) for t in levels for p in [0,1]];rng=np.random.default_rng(int(c['order_seed']));order=[]
for rep in range(reps):
    for idx in rng.permutation(len(conditions)):order.append([rep,*conditions[idx]])
assert np.array_equal(np.array(order,dtype='<i4'),np.fromfile(root/'benchmark_order.i32',dtype='<i4').reshape(-1,3))
native=np.fromfile(old/'native_ids.i64',dtype='<i8').reshape(n,10);all16=np.fromfile(old/'all16_ids.i64',dtype='<i8').reshape(n,10)
checked={};models=set();graphs=set()
for phase in ['stress','benchmark','profile']:
    total=0
    for meta in sorted((root/phase).glob('*.json')):
        m=json.loads(meta.read_text());assert m['status']=='PASS';T=m['workers'];assert m['live_process_threads_at_gate']==T+1
        graphs.add(m['graph_fingerprint']);models.add(m['model_memory_sha256'])
        assert len(set(w['visited_TLS_address'] for w in m['per_worker']))==T
        assert all(w['cpu']==cpus[w['worker']] and w['omp_max_threads']==1 for w in m['per_worker'])
        a=np.genfromtxt(meta.with_suffix('.csv'),delimiter=',',names=True);q=a['query_id'].astype(int);flags=a['policy'].astype(bool)
        ids=np.fromfile(meta.with_name(meta.stem+'_ids.i64'),dtype='<i8').reshape(-1,10)
        assert len(a)==m['queries'] and np.array_equal(ids,np.where(flags[:,None],all16[q],native[q]))
        assert np.array_equal(a['exact_evals'],flags*16) and int(a['exact_evals'].sum())==m['exact_evals']
        if phase!='stress':
            assert len(a)==n*passes and np.array_equal(np.sort(a['pass']*n+q),np.arange(n*passes))
            assert np.array_equal(a['worker'],q%T)
        total+=len(a)
    checked[phase]=total
assert checked['benchmark']==reps*len(levels)*2*n*passes
assert checked['stress']==sum(levels)*int(c['stress_calls_per_worker'])
assert len(models)==1 and graphs=={c['graph_fingerprint']}
regenerated=[]
for folder in ['tables','figures']:
    for file in sorted((Path('results')/folder).glob('phase4d_*')):
        assert digest(file)==digest(repro/folder/file.name),str(file);regenerated.append(str(file))
files=list(root.rglob('*'))+[Path(f) for f in regenerated]+list(Path('scripts').glob('*phase4d*.py'))+[Path('src/experiments/faiss_phase4d.cpp'),Path('tests/test_phase4d_metrics.py'),Path('CMakeLists.txt')]
hashes={str(f):digest(f) for f in files if f.is_file() and f.name not in ['audit.log','final_audit.json']}
dump(root/'final_audit.json',{'status':'PASS','checked_concurrent_output_rows':checked,'deterministic_schedule_regenerated':True,
    'all_immutable_source_binary_input_hashes_match':True,'shared_model_memory_hash_unchanged':True,
    'byte_identical_tables_figures':regenerated,'artifact_sha256':hashes})
print('PASS',checked,';',len(regenerated),'byte-identical tables/figures')
