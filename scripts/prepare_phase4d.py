#!/usr/bin/env python3
"""Freeze unchanged inputs, physical-core assignments, schedules and build."""
import json
import os
import platform
import subprocess
from pathlib import Path
import numpy as np
from analyze_phase3a_sift1m import read_conf
from analyze_phase3b_fixed_candidate_ranking import digest,dump

cp=Path('configs/indexes/phase4d_multithread_scalability.conf');c=read_conf(cp);root=Path(c['run']);old=Path(c['phase4b']);prev=Path(c['phase4c'])
assert not (root/'provenance.json').exists()
prior=json.loads((prev/'provenance.json').read_text())['immutable_sha256']
engine=[Path('src/instrumentation')/n for n in ['bounded_retention.h','bounded_retention.cpp','streaming_refinement.h','streaming_refinement.cpp']]
inputs=[Path(c['graph_path']),Path(c['model_path'])]+[old/n for n in ['queries.f32','warmup.f32','queries_learn_ids.i64','native_ids.i64','all16_ids.i64','gt_ids.i64','candidate_offsets.i64']]
for p in inputs+engine:assert digest(p)==prior[str(p)],str(p)
reference=[prev/n for n in ['top16_ids.i64','top16_order.u32','top16_scores.f32','validation_queries.csv']]
prior_audit=json.loads((prev/'final_audit.json').read_text());assert prior_audit['status']=='PASS'
for p in reference:assert digest(p)==prior_audit['artifact_sha256'][str(p)],str(p)
levels=list(map(int,c['workers'].split(',')));cpus=list(map(int,c['worker_cpus'].split(',')))
assert levels==[1,2,4,8] and cpus==list(range(2,10)) and max(levels)<=len(cpus)
cores=[];topology={}
for cpu in cpus:
    assert cpu in os.sched_getaffinity(0)
    rootcpu=Path(f'/sys/devices/system/cpu/cpu{cpu}')
    v={k:(rootcpu/'topology'/k).read_text().strip() for k in ['core_id','physical_package_id','thread_siblings_list']}
    cores.append((v['physical_package_id'],v['core_id']));topology[str(cpu)]=v
assert len(set(cores))==8,'not distinct physical cores'
rng=np.random.default_rng(int(c['order_seed']));conditions=[(t,p) for t in levels for p in [0,1]];order=[]
for rep in range(int(c['repetitions'])):
    for idx in rng.permutation(len(conditions)):order.append([rep,*conditions[idx]])
np.array(order,dtype='<i4').tofile(root/'benchmark_order.i32')
val=np.genfromtxt(prev/'validation_queries.csv',delimiter=',',names=True)
ties=val['query_id'][val['pq_rank16_boundary_tie']!=0].astype(np.int32)
rng=np.random.default_rng(int(c['stress_seed']));rest=np.setdiff1d(np.arange(int(c['query_count'])),ties)
sample=np.concatenate([ties,rng.choice(rest,int(c['stress_sample_count'])-len(ties),replace=False)])
rng.shuffle(sample);np.asarray(sample,dtype='<i4').tofile(root/'stress_queries.i32')
sources=[cp,root/'preregistration.md',Path('CMakeLists.txt'),Path('src/experiments/faiss_phase4d.cpp'),Path('build/faiss_phase4d'),Path(__file__)]+engine
files=sources+inputs+reference+list(root.glob('*.i32'))
dump(root/'provenance.json',{'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    'faiss_commit':subprocess.check_output(['git','-C','third_party/faiss','rev-parse','HEAD'],text=True).strip(),
    'config':c,'immutable_sha256':{str(p):digest(p) for p in files},'physical_core_assignments':topology,
    'machine':platform.uname()._asdict(),'lscpu':subprocess.check_output(['lscpu'],text=True),
    'lscpu_extended':subprocess.check_output(['lscpu','-e=CPU,CORE,SOCKET,NODE,ONLINE'],text=True),
    'memory':Path('/proc/meminfo').read_text(),'compiler':subprocess.check_output(['c++','--version'],text=True),
    'compiler_flags':Path('build/CMakeFiles/faiss_phase4d.dir/flags.make').read_text(),
    'bounded_flags':Path('build/CMakeFiles/qh_bounded.dir/flags.make').read_text(),
    'faiss_flags':Path('build/faiss-build/faiss/CMakeFiles/faiss.dir/flags.make').read_text(),
    'cache_cpu2':{str(p):p.read_text().strip() for p in Path('/sys/devices/system/cpu/cpu2/cache').glob('index*/*') if p.name in ['size','level','type','shared_cpu_list']},
    'frequency_policy':{str(p):p.read_text().strip() for p in [Path('/sys/devices/system/cpu/cpu2/cpufreq/scaling_governor'),Path('/sys/devices/system/cpu/intel_pstate/no_turbo')] if p.exists()},
    'stress_sample_known_tie_queries':len(ties)})
print('Phase4D frozen;8 distinct physical cores; historical engine/input hashes unchanged')
