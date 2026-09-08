#!/usr/bin/env python3
"""Sequential primary and optional gated perf diagnostics; no heavy co-workers."""
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from analyze_phase3a_sift1m import read_conf
from analyze_phase3b_fixed_candidate_ranking import digest,dump

cp=Path('configs/indexes/phase4d_multithread_scalability.conf');c=read_conf(cp);root=Path(c['run'])
assert not (root/'benchmark_environment.json').exists()
prov=json.loads((root/'provenance.json').read_text())
def verify():
    for p,h in prov['immutable_sha256'].items():assert digest(p)==h,p
verify();assert json.loads((root/'stress_complete.json').read_text())['status']=='PASS'
env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OMP_DYNAMIC':'FALSE','OMP_MAX_ACTIVE_LEVELS':'1'}
prefix=['taskset','-c','0,2-9','./build/faiss_phase4d']
def snapshot():return {'time_ns':time.time_ns(),'loadavg':os.getloadavg(),'processes_namespace_local':subprocess.check_output(['ps','-eo','pid,comm,pcpu','--sort=-pcpu'],text=True)}
record={'before':snapshot(),'environment':{k:env[k] for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_DYNAMIC','OMP_MAX_ACTIVE_LEVELS']},'commands':[],'counter_diagnostics':[]}
command=prefix+['benchmark',str(cp)];record['commands'].append(command)
print('START primary benchmark',flush=True)
with (root/'benchmark.log').open('x') as f:subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
record['after_primary']=snapshot();print('COMPLETE primary benchmark',flush=True)
# Smoke-test counters; inability to collect them never invalidates main results.
events='task-clock,cycles,instructions,cache-references,cache-misses'
try:
    probe=subprocess.run(['perf','stat','--all-user','-e',events,'--','true'],env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=15)
    (root/'perf_probe.log').write_text(probe.stdout);available=probe.returncode==0
except (OSError,subprocess.TimeoutExpired) as e:
    (root/'perf_probe.log').write_text(str(e));available=False
if available:
    for t in [1,8]:
        for p,name in [(0,'NATIVE'),(1,'BOUNDED_ALL16')]:
            temp=Path(tempfile.mkdtemp(prefix='phase4d-perf-',dir='/tmp'))
            ctl=temp/'control';ack=temp/'ack';os.mkfifo(ctl);os.mkfifo(ack)
            result=root/f'perf_{name}_T{t}.csv';log=root/f'perf_{name}_T{t}.log'
            command=['perf','stat','--all-user','-D','-1','--control',f'fifo:{ctl},{ack}','-x',',','-e',events,'-o',str(result),'--']+prefix+['profile',str(cp),str(t),str(p),str(ctl),str(ack)]
            record['commands'].append(command);print('START gated perf',name,t,flush=True)
            try:
                with log.open('x') as f:r=subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=120)
                status='PASS' if r.returncode==0 else 'UNAVAILABLE_OR_FAILED'
            except (OSError,subprocess.TimeoutExpired) as e:status=str(e)
            record['counter_diagnostics'].append({'policy':name,'workers':t,'status':status,'command':command})
            print('COMPLETE gated perf',name,t,status,flush=True)
else:record['counter_diagnostics'].append({'status':'SKIPPED counters unavailable; primary unaffected'})
record['after_counters']=snapshot();verify();dump(root/'benchmark_environment.json',record)
