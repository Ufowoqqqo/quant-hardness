#!/usr/bin/env python3
"""Score-only negative control using immutable historical candidate/PQ files."""
import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
import numpy as np
from analyze_phase3a_sift1m import read_conf, spearman
from analyze_phase3b_fixed_candidate_ranking import digest, dump
from analyze_phase3c_precision_transition import read_candidates, jsonl, gz_writer
from phase3g_metrics import evaluate, permutations, summary, empirical_rank, pair_diagnostics, scalar_reference


def initialize(config, run):
    cfg = read_conf(config); source = Path(cfg['model_source']); old = json.loads((source/'final_verification.json').read_text())['sha256']
    inputs = list(Path(cfg['candidate_source']).glob('*')) + [Path(cfg['geometry_source']), Path(cfg['observed_reference'])]
    frozen = json.loads((source/'input_provenance.json').read_text())['input_sha256']
    graph = next(Path(p) for p in frozen if p.endswith('hnsw_fp32.index'))
    inputs += [graph, source/'rotation_0/base.f32', source/'input_provenance.json', source/'final_verification.json']
    manifests = {}
    for name in cfg['models'].split(','):
        root = source/name; m = json.loads((root/'manifest.json').read_text()); manifests[name] = m
        inputs += [root/'manifest.json', root/'scores.f32', root/'pq64.index']
        for file, key in [('scores.f32', 'scores_sha256'), ('pq64.index', 'model_file_sha256')]:
            assert digest(root/file) == m[key]
    hashes = {}
    for p in inputs:
        if not p.is_file(): continue
        h = digest(p); hashes[str(p)] = h
        expected = old.get(str(p), frozen.get(str(p)))
        if expected is not None: assert h == expected, p
    assert hashes[str(graph)] == frozen[str(graph)]
    assert hashes[str(source/'rotation_0/base.f32')] == manifests['R0-I1']['rotated_base_sha256']
    assert len({m['graph_fingerprint'] for m in manifests.values()}) == 1
    code = [config, run/'preregistration.md', Path(__file__), Path(__file__).with_name('phase3g_metrics.py'), Path('tests/test_phase3g_metrics.py')]
    dump(run/'provenance.json', {'git_commit': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        'git_status': subprocess.check_output(['git','status','--short'], text=True), 'utc_start': time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'machine': platform.uname()._asdict(), 'cpu': subprocess.check_output(['lscpu'], text=True),
        'numpy': np.__version__, 'input_sha256': hashes, 'source_sha256': {str(p):digest(p) for p in code},
        'config': cfg, 'models': manifests, 'dataset_search_config': read_conf(Path(cfg['candidate_source'])/'phase3a_config.conf'),
        'no_search_no_training': True})
    (run/'resolved_config.conf').write_bytes(config.read_bytes())


def references(cfg):
    geometry = list(jsonl(Path(cfg['geometry_source'])))
    names = cfg['models'].split(','); observed = {name:[] for name in names}
    for row in jsonl(Path(cfg['observed_reference'])):
        for name in names:
            m = next(v for v in row['models'] if v['model'] == name)
            observed[name].append({key:m[key] for key in ('topk_ids','loss_hits','cross_boundary_inversion_count','topk_agreement','fixed_candidate_recall')})
    return geometry, observed


def derive(config, run):
    cfg = read_conf(config); assert not (run/'provenance.json').exists(); initialize(config, run)
    geometry, previous = references(cfg); nq = int(cfg['queries']); P = int(cfg['permutations']); k = int(cfg['k'])
    assert len(geometry) == nq; source = Path(cfg['model_source']); start = time.monotonic()
    # Identity vectors are existing FP32 data; norm computation does not search.
    base = np.memmap(source/'rotation_0/base.f32', dtype='<f4', mode='r').reshape(-1,128)
    norms = np.sqrt(np.einsum('ij,ij->i',base,base,dtype=np.float64))
    for mi, name in enumerate(cfg['models'].split(',')):
        out = run/name; out.mkdir(exist_ok=False); score = np.memmap(source/name/'scores.f32',dtype='<f4',mode='r')
        raw = {key:np.empty((nq,P,k) if key=='topk' else (nq,P),dtype='<i4' if key in ('topk','inversions') else '<i2')
               for key in ('topk','hits','loss_hits','agreement_hits','displaced_count','inversions')}
        samples = np.empty((nq,int(cfg['diagnostic_sample_per_query']),4),dtype='<f8')
        rank = np.empty((nq,30),dtype='<f8'); pair_totals = np.zeros((P,3)); offset = 0
        negscores = 0; scalar_checks = 0; tie_queries = 0
        with gz_writer(out/'queries.jsonl.gz') as target, gz_writer(out/'replacement_pairs.jsonl.gz') as pairout:
            for meta, a in read_candidates(Path(cfg['candidate_source']),jsonl(Path(cfg['candidate_source'])/'queries.jsonl')):
                q = meta['query_id']; g = geometry[q]; assert q == g['query_id']; n = len(a)
                ids = a['id']; d = a['exact'].astype(float); p = score[offset:offset+n].astype(float); offset += n
                assert hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest() == meta['candidate_order_sha256'] == g['candidate_order_sha256']
                assert meta['ground_truth_ids'] == g['ground_truth_ids']; truth = meta['ground_truth_ids']
                e = p-d; assert np.array_equal(d+e,p)
                order = np.argsort(d,kind='stable'); assert ids[order[:k]].tolist() == g['exact_topk_ids']
                obs = evaluate(ids,d,d+e,truth,k); old = previous[name][q]
                assert obs['topk'][0].tolist() == old['topk_ids']
                assert int(obs['loss_hits'][0]) == old['loss_hits']
                assert int(obs['inversions'][0]) == old['cross_boundary_inversion_count']
                assert obs['hits'][0]/k == old['fixed_candidate_recall'] and obs['agreement_hits'][0]/k == old['topk_agreement']
                idx = permutations(n,P,int(cfg['permutation_seed']),mi,q); shuffled = e[idx]
                # Verify the bijection itself for every permutation, including repeated residual values.
                assert np.array_equal(np.sort(idx,axis=1),np.broadcast_to(np.arange(n),idx.shape))
                scores = d+shuffled; null = evaluate(ids,d,scores,truth,k); negscores += int((scores<0).sum())
                for key in raw: raw[key][q] = null[key]
                if q % int(cfg['centered_control_query_stride']) == 0:
                    ref = scalar_reference(ids.tolist(),d.tolist(),scores[0].tolist(),truth,k)
                    for key in ref: np.testing.assert_array_equal(ref[key],null[key][0])
                    scalar_checks += 1
                tie_queries += bool(g['exact_boundary_tie'])
                loss = null['loss_hits']/k; inv = null['inversions']; observed_loss = int(obs['loss_hits'][0])/k
                repl, totals = pair_diagnostics(ids,d,e,shuffled,obs['topk'][0],null['topk'],k); pair_totals += totals
                for r in repl: pairout.write(json.dumps({'query_id':q,**r},allow_nan=False)+'\n')
                rank[q] = e[order[:30]]
                rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([int(cfg['diagnostic_seed']),q])))
                chosen = rng.choice(n,size=int(cfg['diagnostic_sample_per_query']),replace=False)
                samples[q] = np.column_stack([ids[chosen],d[chosen],e[chosen],norms[ids[chosen]]])
                regions = {}
                for label, pos in [('top10',order[:k]),('near11_20',order[k:2*k]),('far21plus',order[2*k:])]:
                    v = e[pos]; regions[label] = {'count':len(v),'mean':float(v.mean()),'mae':float(np.abs(v).mean()),'variance':float(v.var())}
                row = {'query_id':q,'candidate_order_sha256':meta['candidate_order_sha256'],'candidate_count':n,
                    'exact_topk_ids':g['exact_topk_ids'],'ground_truth_ids':truth,'oracle_recall':g['oracle_recall'],
                    'observed_topk_ids':obs['topk'][0].tolist(),'observed_loss':observed_loss,
                    'observed_recall':int(obs['hits'][0])/k,'observed_inversions':int(obs['inversions'][0]),
                    'observed_agreement':int(obs['agreement_hits'][0])/k,'observed_displaced_count':int(obs['displaced_count'][0]),
                    'null_loss':summary(loss),'null_inversions':summary(inv),
                    'excess_loss':observed_loss-float(loss.mean()),'excess_inversions':int(obs['inversions'][0])-float(inv.mean()),
                    'loss_rank':empirical_rank(int(obs['loss_hits'][0]),null['loss_hits']),
                    'inversion_rank':empirical_rank(int(obs['inversions'][0]),inv),
                    'residual_mean':float(e.mean()),'residual_variance':float(e.var()),'residual_mae':float(np.abs(e).mean()),
                    'residual_distance_spearman':spearman(d.tolist(),e.tolist()),
                    'residual_norm_spearman':spearman(norms[ids].tolist(),e.tolist()),'regions':regions}
                target.write(json.dumps(row,allow_nan=False)+'\n')
                if (q+1)%500 == 0: print(f'{name} {q+1}/{nq}; elapsed {time.monotonic()-start:.1f}s',flush=True)
        assert offset == len(score) == 10353047
        with (out/'permutations.npz').open('xb') as f: np.savez_compressed(f,**raw)
        with (out/'diagnostics.npz').open('xb') as f: np.savez_compressed(f,rank_residuals=rank,sampled_id_distance_residual_norm=samples,null_replacement_totals=pair_totals)
        dump(out/'validation.json',{'observed_scores_exactly_reconstructed':True,'all_observed_ordered_ids_and_metrics_match_3F':True,
            'all_permutations_bijective':True,'all_candidate_order_and_exact_topk_gt_match':True,'queries':nq,'permutations_per_query':P,
            'scalar_reference_checks':scalar_checks,'negative_null_scores_not_clipped':negscores,'exact_boundary_tie_queries':tie_queries,
            'score_count':offset,'model':name,'elapsed_total_seconds':time.monotonic()-start})
    # Frozen scientific input check at the end; no arrays/models/graph modified.
    provenance = json.loads((run/'provenance.json').read_text())
    for path, h in provenance['input_sha256'].items(): assert digest(path) == h, path
    dump(run/'derive_validation.json',{'status':'PASS','input_hashes_unchanged':True,'elapsed_seconds':time.monotonic()-start})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--config',type=Path,default=Path('configs/indexes/phase3g_residual_permutation.conf'))
    p.add_argument('--run',type=Path,default=Path('runs/phase3g_residual_permutation_v1'))
    args = p.parse_args(); derive(args.config,args.run)
