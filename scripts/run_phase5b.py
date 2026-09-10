"""Exclusive-output, primary-only Phase5B stages; no alternate configurations."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path('runs/phase5b_compression_robustness_v1')
CONFIG='configs/indexes/phase5b_compression_robustness.conf'
PY='/rwproject/kdd-db/kluaq/miniconda3/envs/build_env/bin/python'
PLOT='/tmp/phase3b-plot-env/bin/python'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['model','recall','systems'],required=True);a=p.parse_args()
    env=dict(os.environ,OMP_NUM_THREADS='1',OMP_DYNAMIC='FALSE',OMP_MAX_ACTIVE_LEVELS='1',OPENBLAS_NUM_THREADS='1',
             MKL_NUM_THREADS='1',BLIS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1',MPLCONFIGDIR='/tmp/phase5b-mpl-cache')
    logs=ROOT/'execution_logs';logs.mkdir(exist_ok=True)
    assert json.loads((ROOT/'input_verification.json').read_text())['status']=='PASS'
    files=[Path(CONFIG),ROOT/'preregistration.md',Path(__file__),Path('CMakeLists.txt')]
    files+=list(Path('src/experiments').glob('*phase5a*'))+list(Path('src/instrumentation').glob('bounded_retention.*'))+list(Path('src/instrumentation').glob('streaming_refinement.*'))
    files += [Path('build/faiss_phase5a'),Path('build/faiss_phase5a_bench'),Path('tests/phase5a_retention_correctness.cpp')]
    if a.stage=='model':
        original=json.loads(Path('runs/phase5a_highdim_external_validity_v1/execution_provenance.json').read_text())
        for name in original['sha256']:
            if '/instrumentation/' in name: assert sha(name)==original['sha256'][name],('online implementation changed',name)
        info=dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True,timeout=30).strip(),implementation_dirty=True,
            sha256={str(f):sha(f) for f in files},environment={k:env[k] for k in env if k.startswith(('OMP_','OPENBLAS_','MKL_','BLIS_','VECLIB_','NUMEXPR_'))},
            lscpu=subprocess.check_output(['lscpu'],text=True),memory=Path('/proc/meminfo').read_text(),
            compiler=subprocess.check_output(['c++','--version'],text=True),
            flags={str(f):f.read_text() for f in Path('build').glob('**/flags.make') if any(k in str(f) for k in ['faiss_phase5a.dir','faiss_phase5a_bench.dir','qh_bounded.dir','faiss.dir'])},
            cpu_topology={cpu:{k:Path(f'/sys/devices/system/cpu/cpu{cpu}/topology/{k}').read_text().strip() for k in ['core_id','physical_package_id','thread_siblings_list']} for cpu in range(2,10)},
            frequency={str(f):f.read_text() for f in [Path('/sys/devices/system/cpu/intel_pstate/no_turbo'),Path('/sys/devices/system/cpu/cpu2/cpufreq/scaling_governor')] if f.exists()})
        assert len({(v['core_id'],v['physical_package_id']) for v in info['cpu_topology'].values()})==8
        for name,flag in info['flags'].items():assert flag==original['flags'][name],('flags changed',name)
        with (ROOT/'execution_provenance.json').open('x') as f:json.dump(info,f,indent=2)
        commands=[('tests',['ctest','--test-dir','build','--output-on-failure']),
            ('pq',['build/faiss_phase5a','pq',CONFIG]),('sanity',['build/faiss_phase5a','sanity',CONFIG])]
    else:
        previous=json.loads((ROOT/'execution_provenance.json').read_text())
        for f,digest in previous['sha256'].items():assert sha(f)==digest,('frozen code changed',f)
        if a.stage=='recall':
            assert json.loads((ROOT/'sanity_complete.json').read_text())['status']=='PASS'
            commands=[('recall',['build/faiss_phase5a','recall',CONFIG]),
                ('recall_audit',[PY,'scripts/audit_phase5a_recall.py','--root',str(ROOT)]),
                ('recall_analysis',[PLOT,'scripts/analyze_phase5b.py','--stage','recall'])]
            hashes={str(f):sha(f) for f in [Path('scripts/analyze_phase5b.py'),Path('scripts/audit_phase5a_recall.py')]}
            with (ROOT/'analysis_freeze.json').open('x') as f:json.dump(hashes,f,indent=2)
        else:
            assert json.loads((ROOT/'recall_audit.json').read_text())['status']=='PASS'
            for f,digest in json.loads((ROOT/'analysis_freeze.json').read_text()).items():assert sha(f)==digest
            os.sched_setaffinity(0,{0})
            commands=[('single',['build/faiss_phase5a_bench','single',CONFIG]),
                ('single_analysis',[PLOT,'scripts/analyze_phase5b.py','--stage','single']),
                ('concurrent',['build/faiss_phase5a_bench','concurrent',CONFIG]),
                ('final_analysis',[PLOT,'scripts/analyze_phase5b.py','--stage','final'])]
    with (logs/(a.stage+'_commands.json')).open('x') as f:json.dump(dict(commands=commands,environment={k:env[k] for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']}),f,indent=2)
    for name,cmd in commands:
        print('START',name,' '.join(cmd),flush=True);start=time.monotonic();last=start
        with (logs/(name+'.log')).open('x') as f:
            proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env)
            while proc.poll() is None:
                time.sleep(5)
                if time.monotonic()-last>=60:
                    print('RUNNING',name,'seconds',round(time.monotonic()-start),flush=True);last=time.monotonic()
        print('END',name,'exit',proc.returncode,'seconds',round(time.monotonic()-start,2),flush=True)
        if proc.returncode:raise SystemExit('STOP '+name+' failed; raw files preserved')
    print(a.stage,'COMPLETE; no extra conditions',flush=True)


if __name__=='__main__':main()
