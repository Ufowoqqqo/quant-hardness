#!/usr/bin/env python3
"""Lossless Git-sized transport of the five validated Phase 3F base arrays."""
import argparse
import hashlib
import json
from pathlib import Path

RUN=Path('runs/phase3f_rotation_stability_v1')
CHUNK=48*1024*1024


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(4*1024*1024),b''):h.update(data)
    return h.hexdigest()


def pack(run):
    archive=run/'git_storage';archive.mkdir(exist_ok=False)
    audit=json.loads((run/'final_verification.json').read_text())['sha256']
    records=[]
    for rotation in range(5):
        relative=f'rotation_{rotation}/base.f32';source=run/relative
        expected=audit[str(source)];assert sha(source)==expected
        root=archive/f'rotation_{rotation}';root.mkdir();parts=[];whole=hashlib.sha256();size=0
        with source.open('rb') as f:
            for i,data in enumerate(iter(lambda:f.read(CHUNK),b'')):
                file=root/f'part-{i:03d}.bin'
                with file.open('xb') as target:target.write(data)
                whole.update(data);size+=len(data)
                parts.append({'file':str(file.relative_to(archive)),'size':len(data),'sha256':hashlib.sha256(data).hexdigest()})
        assert whole.hexdigest()==expected and size==source.stat().st_size
        records.append({'target':relative,'size':size,'sha256':expected,'parts':parts})
        print(f'packed {relative}; original retained',flush=True)
    with (archive/'manifest.json').open('x') as f:json.dump({'chunk_bytes':CHUNK,'files':records},f,indent=2);f.write('\n')


def verify_or_restore(run,restore=False,destination=None):
    archive=run/'git_storage';manifest=json.loads((archive/'manifest.json').read_text());dest=destination or run
    for record in manifest['files']:
        h=hashlib.sha256();size=0
        # Check every chunk and the ordered reconstructed stream BEFORE writes.
        for part in record['parts']:
            file=archive/part['file'];assert file.stat().st_size==part['size'] and sha(file)==part['sha256']
            with file.open('rb') as f:
                for data in iter(lambda:f.read(4*1024*1024),b''):h.update(data);size+=len(data)
        assert size==record['size'] and h.hexdigest()==record['sha256']
        target=dest/record['target']
        if target.exists():assert target.stat().st_size==size and sha(target)==record['sha256']
        elif restore:
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as f:
                for part in record['parts']:
                    with (archive/part['file']).open('rb') as source:
                        for data in iter(lambda:source.read(4*1024*1024),b''):f.write(data)
            assert sha(target)==record['sha256']
        print(f"verified {'restored ' if restore else ''}{record['target']}",flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['pack','verify','restore']);p.add_argument('--run',type=Path,default=RUN)
    p.add_argument('--destination',type=Path);a=p.parse_args()
    if a.mode=='pack':pack(a.run)
    else:verify_or_restore(a.run,a.mode=='restore',a.destination)
