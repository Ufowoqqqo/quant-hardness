#!/usr/bin/env python3
"""Non-interactive primary-only pipeline; stops on any failed gate.

Start only after preparation/build/tests. No alternative configurations or
exploratory L values exist in this runner. All commands, logs and environment
are preserved. Progress heartbeats are not experiment results.
"""
import argparse
import hashlib
import json
import os
import shlex
import subprocess
import time
from pathlib import Path


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['offline','systems'],required=True);args=p.parse_args()
    root=Path('runs/phase5a_highdim_external_validity_v1');config='configs/indexes/phase5a_highdim_external_validity.conf'
    py='/rwproject/kdd-db/kluaq/miniconda3/envs/build_env/bin/python';plot='/tmp/phase3b-plot-env/bin/python'
    env=dict(os.environ,MPLCONFIGDIR='/tmp/phase5a-mpl-cache',OMP_NUM_THREADS='1',OMP_DYNAMIC='FALSE',OMP_MAX_ACTIVE_LEVELS='1',
             OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',BLIS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    logs=root/'execution_logs';logs.mkdir(exist_ok=True)
    if args.stage=='offline':
        assert json.loads((root/'prepared/provenance.json').read_text())['status']=='PASS'
        assert not (root/'execution_provenance.json').exists()
        cpus={}
        for cpu in range(2,10):
            assert cpu in os.sched_getaffinity(0)
            path=Path(f'/sys/devices/system/cpu/cpu{cpu}')
            cpus[cpu]={key:(path/'topology'/key).read_text().strip() for key in ['core_id','physical_package_id','thread_siblings_list']}
        assert len(set((v['physical_package_id'],v['core_id']) for v in cpus.values()))==8
        sources=[Path(config),Path(__file__),Path('CMakeLists.txt'),Path('tests/phase5a_retention_correctness.cpp'),root/'implementation_tests.log']+list(Path('src/experiments').glob('*phase5a*'))+list(Path('src/instrumentation').glob('bounded_retention.*'))+list(Path('src/instrumentation').glob('streaming_refinement.*'))+list(Path('scripts').glob('*phase5a*.py'))+list(Path('build').glob('*phase5a*'))
        info=dict(amendment_commit='fbdf4784a505dfbf201be9e78f788e67d31fd4b2',
            git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),implementation_dirty=True,
            sha256={str(s):sha(s) for s in sources if s.is_file()},physical_worker_topology=cpus,
            environment={k:env[k] for k in ['OMP_NUM_THREADS','OMP_DYNAMIC','OMP_MAX_ACTIVE_LEVELS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']},
            lscpu=subprocess.check_output(['lscpu'],text=True),memory=Path('/proc/meminfo').read_text(),
            compiler=subprocess.check_output(['c++','--version'],text=True),
            flags={str(s):s.read_text() for s in Path('build').glob('**/flags.make') if any(x in str(s) for x in ['faiss_phase5a.dir','faiss_phase5a_bench.dir','qh_bounded.dir','faiss.dir'])},
            frequency={str(s):s.read_text() for s in [Path('/sys/devices/system/cpu/intel_pstate/no_turbo'),Path('/sys/devices/system/cpu/cpu2/cpufreq/scaling_governor')] if s.exists()},
            cache={str(s):s.read_text() for s in Path('/sys/devices/system/cpu/cpu2/cache').glob('index*/*') if s.name in ['size','level','type','shared_cpu_list']})
        (root/'execution_provenance.json').write_text(json.dumps(info,indent=2))
        commands=[('gt',['build/faiss_phase5a','gt',config]),('gt_reference',['build/faiss_phase5a','gt-reference',config]),
                  ('gt_validate',[py,'scripts/validate_phase5a_gt.py']),('graph',['build/faiss_phase5a','graph',config]),
                  ('pq',['build/faiss_phase5a','pq',config]),('sanity',['build/faiss_phase5a','sanity',config]),
                  ('recall',['build/faiss_phase5a','recall',config]),('recall_audit',[py,'scripts/audit_phase5a_recall.py']),
                  ('recall_analysis',[plot,'scripts/analyze_phase5a.py','--stage','recall'])]
    else:
        assert json.loads((root/'recall_audit.json').read_text())['status']=='PASS'
        previous=json.loads((root/'execution_provenance.json').read_text())
        for path,digest in previous['sha256'].items():assert sha(path)==digest,('implementation changed after freeze',path)
        # Keep orchestration/logging off the eight measured physical cores.
        os.sched_setaffinity(0,{0})
        commands=[('single',['build/faiss_phase5a_bench','single',config]),('single_analysis',[plot,'scripts/analyze_phase5a.py','--stage','single']),
                  ('concurrent',['build/faiss_phase5a_bench','concurrent',config]),('final_analysis',[plot,'scripts/analyze_phase5a.py','--stage','final'])]
    with (logs/(args.stage+'_commands.json')).open('x') as f:json.dump(dict(commands=commands,environment={k:env[k] for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']}),f,indent=2)
    for label,cmd in commands:
        print('START',label,shlex.join(cmd),flush=True);start=time.time()
        with (logs/(label+'.log')).open('x') as log:
            proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
            while proc.poll() is None:
                time.sleep(10)
                if int(time.time()-start)%60<10:print('RUNNING',label,'elapsed_s',int(time.time()-start),flush=True)
        print('END',label,'exit',proc.returncode,'seconds',round(time.time()-start,1),flush=True)
        if proc.returncode:raise SystemExit('STOP: '+label+' failed; inspect '+str(logs/(label+'.log')))
    print(args.stage,'COMPLETE; no exploratory diagnostics',flush=True)


if __name__=='__main__':main()
