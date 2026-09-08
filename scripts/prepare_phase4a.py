#!/usr/bin/env python3
"""Held-out split and provenance only: no query outcomes or policy labels."""
import hashlib
import json
import platform
import subprocess
from pathlib import Path
import numpy as np
from analyze_phase3a_sift1m import read_conf
from analyze_phase3b_fixed_candidate_ranking import digest, dump


def fvecs(path):
    a = np.memmap(path, dtype='<i4', mode='r')
    assert len(a) % 129 == 0
    a = a.reshape(-1, 129)
    assert np.all(a[:, 0] == 128)
    x = a[:, 1:].view('<f4')
    assert np.isfinite(x).all()
    return x


def vector_hashes(x):
    return [hashlib.sha256(row.tobytes()).digest() for row in x]


def main():
    cp = Path('configs/indexes/phase4a_gap_guided_refinement.conf')
    c = read_conf(cp); root = Path(c['run'])
    assert not (root/'split.json').exists(), 'refusing overwrite'
    assert digest(c['learn_path']) == c['learn_sha256']
    assert digest(c['base_path']) == '21f66e2975057b5728ba56de1c825bac4f4d89d596609ae985741c6242631816'
    learn = fvecs(c['learn_path']); base = fvecs(c['base_path'])
    old = fvecs(c['old_query_path'])
    assert learn.shape == (100000,128) and base.shape == (1000000,128)
    bh = set(vector_hashes(base)); oh = set(vector_hashes(old))
    seen = set(); eligible = []; excluded = []
    for idx, h in enumerate(vector_hashes(learn)):
        reason = 'base_overlap' if h in bh else 'old_query_overlap' if h in oh else 'duplicate_learn' if h in seen else None
        if reason: excluded.append({'learn_id':idx,'reason':reason})
        else: eligible.append(idx)
        seen.add(h)
    rng = np.random.Generator(np.random.PCG64(int(c['split_seed'])))
    ids = rng.permutation(np.array(eligible,dtype='<i8'))
    nt, nq = int(c['training_count']), int(c['query_count'])
    assert len(ids)>=nt+nq, 'insufficient disjoint learning vectors'
    train, queries = ids[:nt], ids[nt:nt+nq]
    assert not set(train)&set(queries)
    for name, ix in [('training',train),('queries',queries)]:
        np.asarray(ix,dtype='<i8').tofile(root/f'{name}_learn_ids.i64')
        np.asarray(learn[ix],dtype='<f4').tofile(root/f'{name}.f32')
    dump(root/'excluded_learning_ids.json',excluded)
    dump(root/'split.json',{'eligible':len(eligible),'excluded':len(excluded),
        'excluded_by_reason':{r:sum(v['reason']==r for v in excluded) for r in ('base_overlap','old_query_overlap','duplicate_learn')},
        'train_count':nt,'query_count':nq,'split_seed':int(c['split_seed']),
        'query_vs_training_content_overlap':0,'query_vs_old_queries_content_overlap':0,
        'query_vs_base_content_overlap':0})
    files = [cp,root/'preregistration.md',Path(c['base_path']),Path(c['graph_path']),Path(c['old_query_path'])]
    files += sorted(root.glob('*.f32'))+sorted(root.glob('*.i64'))+[Path(c['learn_path'])]
    dump(root/'provenance.json',{'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'git_status':subprocess.check_output(['git','status','--porcelain'],text=True),
        'faiss_commit':subprocess.check_output(['git','-C','third_party/faiss','rev-parse','HEAD'],text=True).strip(),
        'machine':platform.uname()._asdict(),'cpu':Path('/proc/cpuinfo').read_text(),
        'numpy':np.__version__,'source_url':c['learn_source'],
        'sha256':{str(p):digest(p) for p in files},'config':c})
    print(json.dumps(json.loads((root/'split.json').read_text()),indent=2))


if __name__=='__main__':main()
