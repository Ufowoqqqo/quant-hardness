#!/usr/bin/env python3
"""Reconstruct the committed normalized arrays without retrieval or tuning."""
import hashlib
import json
import os
import platform
import subprocess
import shutil
import tempfile
import argparse
from pathlib import Path
import numpy as np
import pyarrow as pa
from phase5a_dataset import normalize_fp32_once
from audit_phase5a_amendment import main as audit


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--finalize-existing',type=Path)
    args=parser.parse_args()
    audit()
    root = Path('runs/phase5a_highdim_external_validity_v1')
    manifest = json.loads((root/'amendment/dataset_manifest.json').read_text())
    c = manifest['config']
    if args.finalize_existing:
        out=root/'prepared';scratch=args.finalize_existing
        assert not (out/'provenance.json').exists()
        hashes={}
        for name,count in [('base',990000),('query',10000),('training',65536),('warmup',512)]:
            path=out/(name+'.f32');assert path.stat().st_size==count*1536*4
            hashes[path.name]=digest(path)
            assert hashes[path.name]==digest(scratch/path.name)
            print('matching archived/scratch hash',name,flush=True)
        name='gt_validation_query_ids.i64';hashes[name]=digest(out/name)
        assert hashes[name]==digest(scratch/name)
        assert (out/name).stat().st_size==100*8
        error=0.
        for name,count in [('base',990000),('query',10000),('training',65536)]:
            arr=np.memmap(out/(name+'.f32'),dtype='<f4',mode='r',shape=(count,1536))
            for start in range(0,count,4096):
                a=np.asarray(arr[start:start+4096],dtype='f8');assert np.isfinite(a).all()
                error=max(error,float(np.max(np.abs(np.linalg.norm(a,axis=1)-1))))
            print('finite/norm recheck',name,flush=True)
        assert error<=2e-6
        base=np.memmap(out/'base.f32',dtype='<f4',mode='r',shape=(990000,1536))
        training=np.memmap(out/'training.f32',dtype='<f4',mode='r',shape=(65536,1536))
        inverse=np.empty(1000000,dtype=np.int64)
        bid=np.fromfile(manifest['ids']['base']['path'],dtype='<i8')
        inverse[bid]=np.arange(990000)
        tid=np.fromfile(manifest['ids']['pq_training']['path'],dtype='<i8')
        for start in range(0,len(tid),256):np.testing.assert_array_equal(training[start:start+256],base[inverse[tid[start:start+256]]])
        data=dict(status='PASS',git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True,timeout=20).strip(),
                  amendment_commit='fbdf4784a505dfbf201be9e78f788e67d31fd4b2',
                  faiss_commit=subprocess.check_output(['git','--git-dir=third_party/faiss/.git','rev-parse','HEAD'],text=True,timeout=20).strip(),
                  config=c,input_manifest_sha256=digest(root/'amendment/dataset_manifest.json'),
                  preprocessing_sha256=digest('scripts/phase5a_dataset.py'),preparer_sha256=digest(__file__),
                  sha256=hashes,shapes={'base':[990000,1536],'query':[10000,1536],'training':[65536,1536]},
                  max_norm_error=error,gt_validation_queries=100,reconstruction_scratch=str(scratch),
                  finalization_recovery='All26 shards hashed/normalized by completed reconstruction; Git -C metadata call stalled. '
                  'Finalization checks all archived/scratch bytes, all output vectors, and all training/base ID alignment.',
                  machine=platform.uname()._asdict(),lscpu=subprocess.check_output(['lscpu'],text=True,timeout=20),
                  memory=Path('/proc/meminfo').read_text(),numpy=np.__version__,pyarrow=pa.__version__,affinity=sorted(os.sched_getaffinity(0)))
        assert data['faiss_commit']=='20f14b31a6d54e243a3d1de6ae193fc4c3ec18ed'
        with (out/'provenance.json').open('x') as f:json.dump(data,f,indent=2)
        print('PREPARED PASS (verified finalization recovery)',flush=True)
        return
    destination = root/'prepared'
    assert not destination.exists()
    # Random permutation writes go to local NVMe; final archived arrays are
    # copied sequentially. This changes I/O only, not a single vector value.
    out = Path(tempfile.mkdtemp(prefix='phase5a-normalization-',dir='/tmp'))
    print('local reconstruction scratch',out,flush=True)
    ids = {k: np.fromfile(v['path'], dtype='<i8') for k,v in manifest['ids'].items()}
    dest = {}
    mapping = {}
    for role in ['base','query']:
        mapping[role] = np.full(1000000, -1, dtype=np.int64)
        mapping[role][ids[role]] = np.arange(len(ids[role]))
        dest[role] = np.memmap(out/(role+'.f32'), mode='w+', dtype='<f4', shape=(len(ids[role]),1536))
    max_norm_error = 0.0
    for rec in manifest['shards']:
        assert digest(rec['path']) == rec['sha256'], rec['path']
        offset = rec['row_start']
        with pa.memory_map(rec['path'],'r') as source:
            for batch in pa.ipc.open_stream(source):
                col = batch.column(batch.schema.get_field_index('text-embedding-3-large-1536-embedding'))
                offsets = col.offsets.to_numpy()
                assert col.null_count == 0 and np.all(np.diff(offsets)==1536)
                values = col.values.slice(int(offsets[0]), int(offsets[-1]-offsets[0])).to_numpy().reshape(-1,1536)
                data = normalize_fp32_once(values)
                max_norm_error = max(max_norm_error,float(np.abs(np.linalg.norm(data.astype('f8'),axis=1)-1).max()))
                for role in dest:
                    target = mapping[role][offset:offset+len(batch)]
                    mask = target >= 0
                    dest[role][target[mask]] = data[mask]
                offset += len(batch)
        assert offset == rec['row_end_exclusive']
        print('normalized source rows',offset,flush=True)
    for a in dest.values(): a.flush()
    positions = mapping['base'][ids['pq_training']]
    assert np.all(positions>=0)
    with (out/'training.f32').open('xb') as f:
        for start in range(0,len(positions),256): dest['base'][positions[start:start+256]].tofile(f)
    # Warmup rows drawn exclusively from the already frozen training list.
    with (out/'warmup.f32').open('xb') as f: dest['base'][positions[:512]].tofile(f)
    rng = np.random.Generator(np.random.PCG64(int(c['gt_validation_seed'])))
    # Explicit latest user request increases the prior 32 checks to 100.
    rng.choice(10000,100,replace=False).astype('<i8').tofile(out/'gt_validation_query_ids.i64')
    destination.mkdir(exist_ok=False)
    scratch = str(out)
    for path in out.iterdir():
        if path.is_file(): shutil.copyfile(path,destination/path.name)
    out = destination
    data = dict(status='PASS', git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                amendment_commit='fbdf4784a505dfbf201be9e78f788e67d31fd4b2',
                faiss_commit=subprocess.check_output(['git','--git-dir=third_party/faiss/.git','rev-parse','HEAD'],text=True,timeout=20).strip(),
                config=c, input_manifest_sha256=digest(root/'amendment/dataset_manifest.json'),
                preprocessing_sha256=digest('scripts/phase5a_dataset.py'),
                preparer_sha256=digest(__file__), sha256={p.name:digest(p) for p in out.iterdir() if p.is_file()},
                shapes={'base':[990000,1536],'query':[10000,1536],'training':[65536,1536]},
                max_norm_error=max_norm_error, gt_validation_queries=100,
                reconstruction_scratch=scratch,
                machine=platform.uname()._asdict(), lscpu=subprocess.check_output(['lscpu'],text=True),
                memory=Path('/proc/meminfo').read_text(), numpy=np.__version__, pyarrow=pa.__version__,
                affinity=sorted(os.sched_getaffinity(0)))
    with (out/'provenance.json').open('x') as f: json.dump(data,f,indent=2)
    print('PREPARED PASS',flush=True)


if __name__=='__main__': main()
