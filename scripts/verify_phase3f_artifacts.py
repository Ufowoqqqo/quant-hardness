#!/usr/bin/env python3
"""Read-only frozen-input, model and independent-analysis reproduction audit."""
import argparse
import json
from pathlib import Path
import numpy as np
from analyze_phase3b_fixed_candidate_ranking import digest, dump
from analyze_phase3d_quantizer_seed_stability import verify_hashes
from analyze_phase3f_rotation_stability import validate, NAMES
from phase3f_rotation_validation import orthogonal, rotate, xvec
from analyze_phase3a_sift1m import read_conf


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('reproduction',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists()
    cfg=read_conf(a.run/'resolved_config.conf');original=read_conf(Path(cfg['phase3a_config']))
    manifests=validate(a.run/'resolved_config.conf',a.run)
    for name,m in zip(NAMES,manifests):
        assert digest('build/faiss_rotation_stability')==m['binary_sha256']
        assert digest('src/experiments/faiss_sift1m.cpp')==m['source_sha256']
        assert digest(a.run/name/'resolved_config.conf')==m['config_sha256']
    v=json.loads((a.run/'invariance.json').read_text())
    base=xvec(original['base_path'],1000000,128);query=xvec(original['query_path'],10000,128)
    for r in v['rotations']:
        root=a.run/f"rotation_{r['rotation']}";matrix=np.fromfile(root/'R.f64',dtype='<f8').reshape(128,128)
        reference=np.eye(128) if r['rotation']==0 else orthogonal(128,r['rotation_seed'])
        assert np.array_equal(matrix,reference)
        for key,file in (('R_sha256','R.f64'),('base_sha256','base.f32'),('queries_sha256','queries.f32')):assert digest(root/file)==r[key]
        # Independent transform regeneration over all database and query vectors;
        # keep historical transformed arrays; do not duplicate multi-GB caches.
        rotated=np.memmap(root/'base.f32',dtype='<f4',mode='r',shape=(1000000,128))
        for start in range(0,len(base),65536):assert np.array_equal(rotate(base[start:start+65536],matrix),rotated[start:start+65536])
        assert np.array_equal(rotate(query,matrix),np.fromfile(root/'queries.f32',dtype='<f4').reshape(10000,128))
    checked=[]
    for folder in ('analysis','ensemble'):
        for original in sorted((a.run/folder).iterdir()):
            if original.is_file():
                copy=a.reproduction/folder/original.name
                assert copy.exists() and digest(original)==digest(copy),str(original)
                checked.append(str(original))
    for folder in ('tables','figures'):
        for original in sorted((Path('results')/folder).glob('phase3f_*')):
            copy=a.reproduction/folder/original.name
            assert copy.exists() and digest(original)==digest(copy),str(original)
            checked.append(str(original))
    verify_hashes(json.loads((a.run/'input_provenance.json').read_text())['input_sha256'])
    paths=[p for p in a.run.rglob('*') if p.is_file()]+list(Path('results/tables').glob('phase3f_*'))+list(Path('results/figures').glob('phase3f_*'))
    paths+=list(Path('scripts').glob('*phase3f*'))+list(Path('tests').glob('*phase3f*'))
    paths += [Path('docs/phase3f_rotation_stability.md'),Path('configs/indexes/phase3f_rotation_stability.conf')]
    dump(a.output,{'status':'PASS','reproduced_files':checked,'full_rotated_arrays_reproduced':True,
        'fifteen_models_verified':True,'identity_3E_scores_reproduced':True,'frozen_inputs_unchanged':True,
        'sha256':{str(p):digest(p) for p in sorted(paths)}})
    print(f'PASS {len(checked)} reproduced files; {len(paths)} artifact hashes',flush=True)


if __name__=='__main__':main()
