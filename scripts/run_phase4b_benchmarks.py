#!/usr/bin/env python3
"""Noninteractive benchmark runner: no concurrent analysis/GT worker launched."""
import json
import os
import subprocess
import time
from pathlib import Path
from analyze_phase3a_sift1m import read_conf
from analyze_phase3b_fixed_candidate_ranking import digest,dump

cp=Path('configs/indexes/phase4b_streaming_feasibility.conf');c=read_conf(cp);root=Path(c['run'])
assert not (root/'benchmark_environment.json').exists()
for file,key in [('provenance.json','sha256'),('validation.json','sha256')]:
    for p,h in json.loads((root/file).read_text())[key].items():assert digest(p)==h,p
def snapshot():
    freq=Path('/sys/devices/system/cpu/cpu2/cpufreq/scaling_cur_freq')
    return {'time_ns':time.time_ns(),'loadavg':os.getloadavg(),'proc_stat':Path('/proc/stat').read_text(),
            'cpu2_khz':freq.read_text().strip() if freq.exists() else None,'meminfo':Path('/proc/meminfo').read_text()}
record={'before':snapshot(),'commands':[],'environment':{'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','affinity_cpu':int(c['cpu'])},
        'compiler_flags':Path('build/CMakeFiles/faiss_phase4b.dir/flags.make').read_text(),
        'faiss_flags':Path('build/faiss-build/faiss/CMakeFiles/faiss.dir/flags.make').read_text()}
env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
for mode in ['benchmark','batch']:
    command=['taskset','-c',c['cpu'],'./build/faiss_phase4b',mode,str(cp)];record['commands'].append(command)
    print('START',mode,flush=True)
    with (root/f'{mode}.log').open('x') as f:subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    record['after_'+mode]=snapshot();print('COMPLETE',mode,flush=True)
for p,h in json.loads((root/'validation.json').read_text())['sha256'].items():assert digest(p)==h,p
dump(root/'benchmark_environment.json',record)
