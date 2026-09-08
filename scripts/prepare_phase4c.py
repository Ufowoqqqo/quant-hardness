#!/usr/bin/env python3
"""Freeze sources, unchanged4B inputs and deterministic timing order."""
import json
import platform
import subprocess
from pathlib import Path
import numpy as np
from analyze_phase3a_sift1m import read_conf
from analyze_phase3b_fixed_candidate_ranking import digest, dump

cp=Path('configs/indexes/phase4c_bounded_candidate_retention.conf');c=read_conf(cp)
root=Path(c['run']);old=Path(c['phase4b']);assert not (root/'provenance.json').exists()
for p,h in json.loads((old/'provenance.json').read_text())['sha256'].items():assert digest(p)==h,p
assert float(c['gap_threshold'])==json.loads((old/'threshold_provenance.json').read_text())['threshold']==710.40625
assert json.loads((old/'final_audit.json').read_text())['status']=='PASS'
for name,modes,seed in [('primary',range(5),int(c['order_seed'])),('diagnostic',[0,4,5],int(c['diagnostic_order_seed']))]:
    rng=np.random.default_rng(seed);conditions=[(m,clock) for m in modes for clock in [0,1]];order=[]
    for rep in range(int(c['repetitions'])):
        for j in rng.permutation(len(conditions)):order.append([rep,*conditions[j]])
    np.array(order,dtype='<i4').tofile(root/f'{name}_order.i32')
np.array([[0,1,2],[0,2,2]],dtype='<i4').tofile(root/'detail_order.i32')
sources=[cp,Path('CMakeLists.txt'),root/'preregistration.md',Path('build/faiss_phase4c'),Path('build/bounded_retention_correctness')]
sources+=list(Path('src/instrumentation').glob('*retention*'))+[Path('src/instrumentation/streaming_refinement.cpp'),Path('src/instrumentation/streaming_refinement.h'),Path('src/experiments/faiss_phase4c.cpp'),Path('tests/bounded_retention_correctness.cpp'),Path(__file__)]
inputs=[Path(c['graph_path']),Path(c['model_path'])]+[old/f for f in ['queries.f32','warmup.f32','queries_learn_ids.i64','candidate_ids.i32','candidate_pq.f32','candidate_offsets.i64','native_ids.i64','all16_ids.i64','gap_ids.i64','gaps.f64','gap_decisions.u8','gt_ids.i64','threshold_provenance.json']]
sources+=list(root.glob('*_order.i32'))
dump(root/'provenance.json',{'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    'faiss_commit':subprocess.check_output(['git','-C','third_party/faiss','rev-parse','HEAD'],text=True).strip(),
    'config':c,'immutable_sha256':{str(p):digest(p) for p in sources+inputs},
    'machine':platform.uname()._asdict(),'lscpu':subprocess.check_output(['lscpu'],text=True),
    'memory':Path('/proc/meminfo').read_text(),'compiler':subprocess.check_output(['c++','--version'],text=True),
    'compiler_flags':Path('build/CMakeFiles/faiss_phase4c.dir/flags.make').read_text(),
    'faiss_flags':Path('build/faiss-build/faiss/CMakeFiles/faiss.dir/flags.make').read_text(),
    'frequency':{str(p):p.read_text().strip() for p in [Path('/sys/devices/system/cpu/cpu2/cpufreq/scaling_governor'),Path('/sys/devices/system/cpu/intel_pstate/no_turbo')] if p.exists()}})
print('Phase4C unchanged inputs/source/binary/order frozen; no outcome inspection')
