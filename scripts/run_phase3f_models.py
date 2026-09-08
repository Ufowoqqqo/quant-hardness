#!/usr/bin/env python3
"""Noninteractive fifteen-model runner, gated by completed invariance audit."""
import json
import os
import subprocess
from pathlib import Path
from analyze_phase3b_fixed_candidate_ranking import digest
from analyze_phase3d_quantizer_seed_stability import verify_hashes

run=Path('runs/phase3f_rotation_stability_v1');config=Path('configs/indexes/phase3f_rotation_stability.conf')
v=json.loads((run/'invariance.json').read_text())
assert v['status']=='PASS' and len(v['rotations'])==5 and v['config_sha256']==digest(config)
verify_hashes(json.loads((run/'input_provenance.json').read_text())['input_sha256'])
expected='status=PASS\nconfig_sha256='+digest(config)+'\n'
for r in v['rotations']:
    for key,file in (('R_sha256','R.f64'),('base_sha256','base.f32'),('queries_sha256','queries.f32')):
        assert digest(run/f"rotation_{r['rotation']}"/file)==r[key]
        expected+=f"R{r['rotation']}_{key}={r[key]}\n"
gate=run/'invariance_gate.conf'
if gate.exists():assert gate.read_text()==expected
else:
    with gate.open('x') as f:f.write(expected)
for model in range(15):
    name=f'R{model//3}-I{model%3+1}'
    assert not (run/name).exists(),'refusing partial rerun/overwrite'
    command=['./build/faiss_rotation_stability','rotation-scores',str(config),str(run),str(model)]
    print(' '.join(command),flush=True)
    with (run/f'{name}_execution.log').open('x') as log:
        subprocess.run(command,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'24'},stdout=log,stderr=subprocess.STDOUT,check=True)
    print(name+' complete',flush=True)
print('all15 models complete',flush=True)
