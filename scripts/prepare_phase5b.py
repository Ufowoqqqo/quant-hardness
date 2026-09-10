"""Verify committed Phase5A bytes, then link immutable inputs; no reconstruction."""
import datetime
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np
from phase5a_dataset import split_ids


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def main():
    old=Path('runs/phase5a_highdim_external_validity_v1');root=Path('runs/phase5b_compression_robustness_v1')
    assert not (root/'input_verification.json').exists()
    parent='541a0eb8c812bf9926f0bc5de3699d908dfa20a4'
    verified={}
    def committed(path):
        data=subprocess.check_output(['git','show',parent+':'+str(path)],timeout=30)
        assert path.read_bytes()==data,('committed metadata differs',path)
    for rel in ['amendment/dataset_manifest.json','prepared/provenance.json','graph.json','gt_validation.json','pq.json','recall_audit.json']:
        committed(old/rel)
    prepared=json.loads((old/'prepared/provenance.json').read_text());assert prepared['status']=='PASS'
    for name,digest in prepared['sha256'].items():
        path=old/'prepared'/name
        assert sha(path)==digest, path;verified[str(path)]=digest;print('PASS',path,flush=True)
    manifest=old/'amendment/dataset_manifest.json';assert sha(manifest)==prepared['input_manifest_sha256']
    assert sha('scripts/phase5a_dataset.py')==prepared['preprocessing_sha256']
    for name,ids in zip(['base','query','pq_training'],split_ids(1000000,10000,65536,20260909,95000011)):
        path=old/'amendment'/(name+'_source_ids.i64');committed(path)
        assert path.read_bytes()==ids.tobytes();verified[str(path)]=sha(path)
    graph=json.loads((old/'graph.json').read_text())
    for name,expected in [('graph.index',graph['file_sha256']),('pq.index',json.loads((old/'pq.json').read_text())['file_sha256'])]:
        path=old/name;assert sha(path)==expected;verified[str(path)]=expected;print('PASS',path,flush=True)
    gt=json.loads((old/'gt_validation.json').read_text());assert gt['status']=='PASS' and gt['queries']>=100
    for name,digest in gt['sha256'].items():
        path=old/name;assert sha(path)==digest;verified[str(path)]=digest
    # Verify exact controls and historical measured references, not regenerate them.
    for ef in (32,64,128):
        for name in ['exact_ids.i64','exact_oracle_ids.i64','per_query.csv','complete.json']:
            path=old/f'recall_ef{ef}'/name;committed(path);verified[str(path)]=sha(path)
    for path in Path('results/tables').glob('phase5a_*'):
        committed(path);verified[str(path)]=sha(path)
    links=['prepared','graph.index','graph.json','gt_ids.i64','gt_distances.f64','gt_validation.json']
    for name in links:
        target=root/name;assert not target.exists();target.symlink_to(Path('../'+old.name)/name,target_is_directory=name=='prepared')
    result=dict(status='PASS',timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        phase5a_commit=parent,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True,timeout=30).strip(),
        sha256=verified,graph_fingerprint=graph['fingerprint'],reused_without_rebuilding=True,
        counts=dict(base=990000,query=10000,training=65536,dimension=1536),
        training_id_sha256=verified[str(old/'amendment/pq_training_source_ids.i64')],
        preregistration_sha256=sha(root/'preregistration.md'),
        config_sha256=sha('configs/indexes/phase5b_compression_robustness.conf'))
    with (root/'input_verification.json').open('x') as f:json.dump(result,f,indent=2)
    print('ALL FROZEN INPUT CHECKSUMS PASS; links reference original arrays/graph')


if __name__=='__main__':main()
