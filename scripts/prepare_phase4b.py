#!/usr/bin/env python3
"""Copy the Phase4A numerical threshold and untouched split suffix; no B scores."""
import gzip
import json
import platform
import subprocess
from pathlib import Path
import numpy as np
from analyze_phase3a_sift1m import read_conf
from analyze_phase3b_fixed_candidate_ranking import digest,dump
from prepare_phase4a import fvecs

c=read_conf(Path('configs/indexes/phase4b_streaming_feasibility.conf'));root=Path(c['run']);old=Path(c['phase4a'])
assert not (root/'provenance.json').exists()
oldprov=json.loads((old/'provenance.json').read_text())
for p,h in oldprov['sha256'].items():assert digest(p)==h,p
for p,h in json.loads((old/'validation.json').read_text())['export_sha256'].items():assert digest(p)==h,p
rows=[json.loads(l) for l in gzip.open(old/'analysis/queries.jsonl.gz','rt')]
g=np.array([r['g9_12_pq'] for r in rows]);threshold=float(c['gap_threshold'])
selected=np.load(old/'analysis/selections.npz')['all_L16_f0.25_GAP']
assert threshold==np.quantile(g,.25)==max(g[selected])==710.40625
dump(root/'threshold_provenance.json',{'threshold':threshold,'comparison':'<=','source_phase':'Phase4A ONLY',
    'linear_q25':float(np.quantile(g,.25)),'selected_rank2500_gap':float(max(g[selected])),
    'phase4a_inclusive_count':int((g<=threshold).sum()),'phase4a_original_quota':2500,
    'score_source_sha256':digest(old/'analysis/queries.jsonl.gz'),'selection_source_sha256':digest(old/'analysis/selections.npz')})
excluded={r['learn_id'] for r in json.loads((old/'excluded_learning_ids.json').read_text())}
eligible=np.array([i for i in range(100000) if i not in excluded],dtype='<i8')
perm=np.random.Generator(np.random.PCG64(94000011)).permutation(eligible)
assert np.array_equal(perm[:65536],np.fromfile(old/'training_learn_ids.i64',dtype='<i8'))
assert np.array_equal(perm[65536:75536],np.fromfile(old/'queries_learn_ids.i64',dtype='<i8'))
test=perm[75536:];assert len(test)==14252
learn=fvecs(old/'sift_learn.fvecs');np.asarray(learn[test],dtype='<f4').tofile(root/'queries.f32');test.tofile(root/'queries_learn_ids.i64')
np.fromfile(old/'queries.f32',dtype='<f4').reshape(10000,128)[:int(c['warmup_queries'])].tofile(root/'warmup.f32')
rng=np.random.default_rng(int(c['order_seed']));order=[]
for rep in range(int(c['repetitions'])):
    for i in rng.permutation(8):order.append([rep,int(i)//2,int(i)%2])
np.asarray(order,dtype='<i4').tofile(root/'benchmark_order.i32')
rng=np.random.default_rng(int(c['batch_order_seed']));batch=[]
for rep in range(int(c['repetitions'])):
    for method in rng.permutation(2):batch.append([rep,int(method)])
np.asarray(batch,dtype='<i4').tofile(root/'batch_order.i32')
files=[Path('configs/indexes/phase4b_streaming_feasibility.conf'),root/'preregistration.md',root/'threshold_provenance.json',Path(c['graph_path']),Path(c['model_path'])]
files+=list(root.glob('*.f32'))+list(root.glob('*.i64'))+list(root.glob('*.i32'))
dump(root/'provenance.json',{'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    'faiss_commit':subprocess.check_output(['git','-C','third_party/faiss','rev-parse','HEAD'],text=True).strip(),
    'config':c,'sha256':{str(p):digest(p) for p in files},'machine':platform.uname()._asdict(),
    'lscpu':subprocess.check_output(['lscpu'],text=True),'memory':Path('/proc/meminfo').read_text(),
    'compiler':subprocess.check_output(['c++','--version'],text=True),
    'frequency':{str(p):p.read_text().strip() for p in [Path('/sys/devices/system/cpu/cpu2/cpufreq/scaling_governor'),Path('/sys/devices/system/cpu/intel_pstate/no_turbo')] if p.exists()},
    'disjoint_roles':{'training':65536,'phase4a_queries':10000,'phase4b_queries':14252,'warmup':'Phase4A first512','content_exclusion_inherited_from_verified_Phase4A':True}})
print('threshold',threshold,'heldout queries',len(test),'policy order frozen')
