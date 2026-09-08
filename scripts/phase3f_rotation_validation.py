#!/usr/bin/env python3
"""Pre-training orthogonal invariance gate; no PQ or graph search."""
import argparse
import json
from pathlib import Path
import numpy as np
from analyze_phase3a_sift1m import read_conf
from analyze_phase3b_fixed_candidate_ranking import digest, dump
from analyze_phase3c_precision_transition import read_candidates, jsonl, gz_writer
from analyze_phase3d_quantizer_seed_stability import prepare, verify_hashes


def orthogonal(d, seed):
    q, r = np.linalg.qr(np.random.default_rng(seed).standard_normal((d, d)))
    return q * np.where(np.diag(r) < 0, -1., 1.)


def rotate(x, r):
    return (np.asarray(x, dtype=np.float64) @ r.T).astype('<f4')


def close_distances(reference, actual, atol, rtol):
    ref=np.asarray(reference, dtype=float);value=np.asarray(actual, dtype=float)
    error=np.abs(value-ref);bound=atol+rtol*np.abs(ref)
    assert np.isfinite(value).all() and np.all(error<=bound), (float(error.max()),float((error/bound).max()))
    return {'max_absolute_error':float(error.max()),'max_relative_error':float((error/np.maximum(np.abs(ref),1e-12)).max()),
            'max_tolerance_fraction':float((error/bound).max())}


def rank_gate(reference, actual, k, atol, rtol):
    """Reject reordered non-ties; retain raw tie changes for the audit."""
    ref=np.asarray(reference);actual=np.asarray(actual)
    close_distances(ref,actual,atol,rtol)
    old=np.argsort(ref,kind='stable');new=np.argsort(actual,kind='stable')
    backwards=-np.diff(ref[new]);tolerance=2*atol+rtol*(np.abs(ref[new[:-1]])+np.abs(ref[new[1:]]))
    assert np.all(backwards<=tolerance), 'non-tolerance ranking reversal'
    return {'raw_order_changed':bool(np.any(old!=new)),
            'raw_topk_set_changed':set(old[:k])!=set(new[:k]),
            'raw_topk_order_changed':bool(np.any(old[:k]!=new[:k])),
            'reference_boundary_tie':bool(ref[old[k-1]]==ref[old[k]]),
            'strict_reference_reversals':int((backwards>0).sum())},old,new


def xvec(path,n,d,integer=False):
    a=np.memmap(path,dtype='<i4',mode='r').reshape(n,d+1)
    assert np.all(a[:,0]==d)
    return a[:,1:] if integer else a[:,1:].view('<f4')


def exhaustive(x,q):
    # Direct norm/dot identity in float64; original SIFT integer arithmetic
    # is exact at these magnitudes. Validate against direct differences below.
    x=np.asarray(x,dtype=np.float64);q=np.asarray(q,dtype=np.float64)
    return (x*x).sum(1)[None,:]+(q*q).sum(1)[:,None]-2*(q@x.T)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=Path('configs/indexes/phase3f_rotation_stability.conf'))
    p.add_argument('--run',type=Path,default=Path('runs/phase3f_rotation_stability_v1'));a=p.parse_args()
    cfg=read_conf(a.config);c=read_conf(Path(cfg['phase3a_config']));run=a.run
    assert not (run/'invariance.json').exists()
    if not (run/'input_provenance.json').exists():prepare(a.config,run)
    verify_hashes(json.loads((run/'input_provenance.json').read_text())['input_sha256'])
    for key in ('base','query','ground_truth'):
        assert digest(c[key+'_path'])==c[key+'_sha256']
    x=xvec(c['base_path'],1000000,128);q=xvec(c['query_path'],10000,128)
    gt=xvec(c['ground_truth_path'],10000,100,True)
    rng=np.random.default_rng(int(cfg['validation_seed']));n=int(cfg['validation_pairs'])
    ia=rng.integers(0,len(x),n);ib=rng.integers(0,len(x),n);iq=rng.integers(0,len(q),n)
    qids=np.sort(rng.choice(len(q),int(cfg['gt_validation_queries']),replace=False))
    dump(run/'validation_sample_ids.json',{'base_a':ia.tolist(),'base_b':ib.tolist(),'query_pairs':iq.tolist(),'exhaustive_queries':qids.tolist()})
    xa=x[ia].astype(float);xb=x[ib].astype(float);qa=q[iq].astype(float)
    refs=[(xa*xa).sum(1),((xa-xb)**2).sum(1),((qa-xa)**2).sum(1)]
    tol=(float(cfg['fp32_distance_atol']),float(cfg['fp32_distance_rtol']))
    rotations=[np.eye(128)]+[orthogonal(128,int(s)) for s in cfg['rotation_seeds'].split(',')]
    # Exhaustive original GT is generated once, independently of saved pools.
    truthdir=run/'validation_ground_truth';truthdir.mkdir(exist_ok=False)
    for start in range(0,len(qids),10):
        d=exhaustive(x,q[qids[start:start+10]])
        for j,qid in enumerate(qids[start:start+10]):
            dr=d[j];provided=gt[qid,:10];cut=np.partition(dr,9)[9]
            assert np.all(dr[provided]<=cut) and set(np.flatnonzero(dr<cut))<=set(provided)
            np.asarray(dr,dtype='<f8').tofile(truthdir/f'q{qid}.f64')
        print(f'original exhaustive GT {start+10}/{len(qids)} PASS',flush=True)
    results=[]
    for index,r in enumerate(rotations):
        root=run/f'rotation_{index}';root.mkdir(exist_ok=False)
        r.astype('<f8').tofile(root/'R.f64')
        orth_error=float(np.abs(r.T@r-np.eye(128)).max());assert orth_error<=float(cfg['orthogonality_atol'])
        v=[xa@r.T,xb@r.T,qa@r.T]
        transformed=[(v[0]**2).sum(1),((v[0]-v[1])**2).sum(1),((v[2]-v[0])**2).sum(1)]
        checks64=[close_distances(ref,z,float(cfg['fp64_atol']),float(cfg['fp64_rtol'])) for ref,z in zip(refs,transformed)]
        bx=np.memmap(root/'base.f32',dtype='<f4',mode='w+',shape=x.shape)
        for start in range(0,len(x),65536):bx[start:start+65536]=rotate(x[start:start+65536],r)
        bx.flush();bq=rotate(q,r);bq.tofile(root/'queries.f32')
        v=[bx[ia].astype(float),bx[ib].astype(float),bq[iq].astype(float)]
        transformed=[(v[0]**2).sum(1),((v[0]-v[1])**2).sum(1),((v[2]-v[0])**2).sum(1)]
        checks32=[close_distances(ref,z,*tol) for ref,z in zip(refs,transformed)]
        vf=[z.astype(np.float32) for z in v]
        accum=[(vf[0]**2).sum(1),((vf[0]-vf[1])**2).sum(1),((vf[2]-vf[0])**2).sum(1)]
        checksacc=[close_distances(ref,z,tol[0],float(cfg['fp32_accumulation_rtol'])) for ref,z in zip(refs,accum)]
        counters={'raw_order_changed':0,'raw_topk_set_changed':0,'raw_topk_order_changed':0,'reference_boundary_tie':0,'strict_reference_reversals':0}
        maxerr=0.;maxrel=0.;oracle=0.;raw_oracle=0.;count=0
        with gz_writer(root/'candidate_validation.jsonl.gz') as out:
            for meta,pool in read_candidates(Path(cfg['candidate_source']),jsonl(Path(cfg['candidate_source'])/'queries.jsonl')):
                qid=meta['query_id'];d=((bx[pool['id']].astype(float)-bq[qid].astype(float))**2).sum(1)
                errors=close_distances(pool['exact'],d,*tol);maxerr=max(maxerr,errors['max_absolute_error']);maxrel=max(maxrel,errors['max_relative_error'])
                rr,old,new=rank_gate(pool['exact'],d,10,*tol)
                # Integer SIFT exact gaps have no distinct values inside tolerance:
                # raw changes must be exact ties, not merely near ties.
                assert rr['strict_reference_reversals']==0
                for key in counters:counters[key]+=int(rr[key])
                ids=pool['id'];truth=set(meta['ground_truth_ids'])
                rec=len(set(ids[old[:10]])&truth)/10;raw=len(set(ids[new[:10]])&truth)/10
                oracle+=rec;raw_oracle+=raw;count+=len(pool)
                out.write(json.dumps({'query_id':qid,**rr,'max_distance_error':errors['max_absolute_error'],
                    'original_topk_ids':ids[old[:10]].tolist(),'rotated_raw_topk_ids':ids[new[:10]].tolist(),
                    'canonical_oracle_recall':rec,'rotated_raw_oracle_recall':raw,'candidate_order_sha256':meta['candidate_order_sha256']})+'\n')
        gtrows=[]
        for start in range(0,len(qids),10):
            distances=exhaustive(bx,bq[qids[start:start+10]])
            for j,qid in enumerate(qids[start:start+10]):
                original=np.fromfile(truthdir/f'q{qid}.f64',dtype='<f8');actual=distances[j]
                err=close_distances(original,actual,*tol)
                chosen=np.argpartition(actual,9)[:10];chosen=chosen[np.argsort(actual[chosen],kind='stable')]
                cutoff=np.partition(original,9)[9]
                assert np.all(original[chosen]<=cutoff) and set(np.flatnonzero(original<cutoff))<=set(chosen)
                # Validate norm/dot distance against independent direct formula.
                direct=((bx[chosen].astype(float)-bq[qid].astype(float))**2).sum(1)
                assert np.allclose(actual[chosen],direct,atol=1e-8,rtol=1e-12)
                gtrows.append({'query_id':int(qid),'provided_ids':gt[qid,:10].tolist(),'rotated_ids':chosen.tolist(),
                    'provided_set_changed':set(gt[qid,:10])!=set(chosen),'boundary_tied':int((original==cutoff).sum())>1,**err})
            print(f'R{index} exhaustive GT {start+10}/{len(qids)} PASS',flush=True)
        dump(root/'ground_truth_validation.json',gtrows)
        result={'rotation':index,'rotation_seed':None if index==0 else int(cfg['rotation_seeds'].split(',')[index-1]),
            'orthogonality_max_abs':orth_error,'fp64_checks':checks64,'fp32_storage_checks':checks32,'fp32_accumulation_checks':checksacc,
            'sample_max_l2_norm_absolute_error':float(np.abs(np.sqrt(transformed[0])-np.sqrt(refs[0])).max()),
            'candidate_validation':counters,'candidates':count,'queries':10000,'max_candidate_distance_error':maxerr,
            'max_candidate_relative_distance_error':maxrel,'canonical_oracle_recall':oracle/10000,'raw_rotated_oracle_recall':raw_oracle/10000,
            'exhaustive_queries':len(qids),'gt_membership_changes_exact_ties_only':sum(g['provided_set_changed'] for g in gtrows),
            'R_sha256':digest(root/'R.f64'),'base_sha256':digest(root/'base.f32'),'queries_sha256':digest(root/'queries.f32')}
        assert abs(oracle/10000-.95315)<1e-12 and count==10353047
        dump(root/'validation.json',result);results.append(result)
        print(json.dumps(result),flush=True)
    verify_hashes(json.loads((run/'input_provenance.json').read_text())['input_sha256'])
    dump(run/'invariance.json',{'status':'PASS','all_before_training':True,'rotations':results,'config_sha256':digest(a.config),
        'validation_source_sha256':digest(__file__),'numpy':np.__version__})
    with (run/'invariance_gate.conf').open('x') as gate:
        gate.write('status=PASS\nconfig_sha256='+digest(a.config)+'\n')
        for r in results:
            for key in ('R_sha256','base_sha256','queries_sha256'):
                gate.write(f"R{r['rotation']}_{key}={r[key]}\n")
    print('ALL FIVE ROTATIONS INVARIANCE GATE PASS; PQ training permitted',flush=True)


if __name__=='__main__':main()
