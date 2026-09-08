#!/usr/bin/env python3
"""Run primary then secondary, with immutable gate and no concurrent workers."""
import json
import os
import subprocess
import time
from pathlib import Path
from analyze_phase3a_sift1m import read_conf
from analyze_phase3b_fixed_candidate_ranking import digest,dump

cp=Path('configs/indexes/phase4c_bounded_candidate_retention.conf');c=read_conf(cp);root=Path(c['run'])
assert not (root/'benchmark_environment.json').exists()
prov=json.loads((root/'provenance.json').read_text())
def verify():
    for p,h in prov['immutable_sha256'].items():assert digest(p)==h,p
verify();assert json.loads((root/'validation.json').read_text())['status']=='PASS'
def snapshot():
    freq=Path('/sys/devices/system/cpu/cpu2/cpufreq/scaling_cur_freq')
    return {'time_ns':time.time_ns(),'loadavg':os.getloadavg(),'proc_stat':Path('/proc/stat').read_text(),
        'memory':Path('/proc/meminfo').read_text(),'cpu2_khz':freq.read_text().strip() if freq.exists() else None,
        'processes':subprocess.check_output(['ps','-eo','pid,comm,pcpu','--sort=-pcpu'],text=True)}
record={'before':snapshot(),'commands':[],'thread_settings':{'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','cpu':c['cpu']},
        'validation_sha256':digest(root/'validation.json')}
env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
for mode in ['primary','diagnostic','detail']:
    command=['taskset','-c',c['cpu'],'./build/faiss_phase4c',mode,str(cp)];record['commands'].append(command)
    print('START',mode,flush=True)
    with (root/f'{mode}.log').open('x') as f:subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    record['after_'+mode]=snapshot();print('COMPLETE',mode,flush=True)
verify();dump(root/'benchmark_environment.json',record)
