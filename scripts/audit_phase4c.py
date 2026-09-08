#!/usr/bin/env python3
"""Audit frozen input identity, every timed output, and independent regeneration."""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np
from analyze_phase3a_sift1m import read_conf
from analyze_phase3b_fixed_candidate_ranking import digest,dump

p=argparse.ArgumentParser();p.add_argument('--reproduction',required=True);args=p.parse_args()
c=read_conf(Path('configs/indexes/phase4c_bounded_candidate_retention.conf'));root=Path(c['run']);old=Path(c['phase4b']);repro=Path(args.reproduction);n=int(c['query_count'])
prov=json.loads((root/'provenance.json').read_text())
for file,h in prov['immutable_sha256'].items():assert digest(file)==h,file
val=json.loads((root/'validation.json').read_text());assert val['status']=='PASS' and val['mode_clock_checks']==n*18
top=np.fromfile(root/'top16_ids.i64',dtype='<i8').reshape(n,16)
ps=np.fromfile(root/'top16_scores.f32',dtype='<f4').reshape(n,16)
order=np.fromfile(root/'top16_order.u32',dtype='<u4').reshape(n,16)
offset=np.fromfile(old/'candidate_offsets.i64',dtype='<i8');ci=np.memmap(old/'candidate_ids.i32',dtype='<i4',mode='r');cs=np.memmap(old/'candidate_pq.f32',dtype='<f4',mode='r')
for q in range(n):
    sl=slice(offset[q],offset[q+1]);o=np.argsort(cs[sl],kind='stable')[:16]
    assert np.array_equal(order[q],o) and np.array_equal(top[q],ci[sl][o]) and np.array_equal(ps[q],cs[sl][o])
arrays={k:np.fromfile(old/f'{k}_ids.i64',dtype='<i8').reshape(n,10) for k in ['native','all16','gap']}
mask=np.fromfile(old/'gap_decisions.u8',dtype='u1');traces=0
for phase in ['primary','diagnostic','detail']:
    assert json.loads((root/f'{phase}_complete.json').read_text())['status']=='PASS'
    schedule=np.fromfile(root/f'{phase}_order.i32',dtype='<i4').reshape(-1,3)
    files=list((root/phase).glob('*.csv'));assert len(files)==len(schedule)
    names=['NATIVE','FULL','BOUNDED','FULL_ALL16','BOUNDED_ALL16','GAP_BOUNDED']
    for rep,mode,clock in schedule:
        stem=f'{names[mode]}_r{rep}_c{clock}';file=root/phase/f'{stem}.csv';a=np.genfromtxt(file,delimiter=',',names=True)
        assert np.array_equal(a['query_id'],np.arange(n));expected=arrays['all16' if mode in [3,4] else 'gap' if mode==5 else 'native']
        assert np.array_equal(np.fromfile(root/phase/f'{stem}_ids.i64',dtype='<i8').reshape(n,10),expected)
        refined=np.ones(n) if mode in [3,4] else mask if mode==5 else np.zeros(n)
        assert np.array_equal(a['exact_evals'],16*refined);traces+=n
generated=[]
for folder in ['tables','figures']:
    for file in sorted((Path('results')/folder).glob('phase4c_*')):
        assert digest(file)==digest(repro/folder/file.name),str(file);generated.append(str(file))
assert gzip.open(root/'analysis/queries.jsonl.gz','rb').read()==gzip.open(repro/'analysis/queries.jsonl.gz','rb').read()
files=list(root.rglob('*'))+[Path(f) for f in generated]+list(Path('scripts').glob('*phase4c*.py'))+[Path('CMakeLists.txt'),Path('src/experiments/faiss_phase4c.cpp'),Path('src/instrumentation/bounded_retention.h'),Path('src/instrumentation/bounded_retention.cpp'),Path('tests/bounded_retention_correctness.cpp'),Path('tests/test_phase4c_metrics.py')]
hashes={str(f):digest(f) for f in files if f.is_file() and f.name not in ['audit.log','final_audit.json']}
dump(root/'final_audit.json',{'status':'PASS','immutable_inputs_sources_binaries_unchanged':True,'query_top16_independent_full_sort_checks':n,'timed_output_count_checks':traces,
    'byte_identical_tables_figures':generated,'identical_derived_query_rows':True,'artifact_sha256':hashes})
print('PASS',traces,'timed outputs;',n,'independent top16 references;',len(generated),'byte-identical tables/figures')
