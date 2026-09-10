"""Independent post-run check of every timed ID and frozen implementation hash."""
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    root = Path('runs/phase5a_highdim_external_validity_v1')
    provenance = json.loads((root / 'execution_provenance.json').read_text())
    for path, expected in provenance['sha256'].items():
        assert sha(path) == expected, path
    graph = json.loads((root / 'graph.json').read_text())
    pq = json.loads((root / 'pq.json').read_text())
    assert sha(root / 'graph.index') == graph['file_sha256']
    assert sha(root / 'pq.index') == pq['file_sha256']
    cells = []
    for stage, expected_cells in [('single', 30), ('concurrent', 20)]:
        directory = root / ('benchmark_' + stage)
        assert json.loads((directory / 'complete.json').read_text())['status'] == 'PASS'
        paths = sorted(directory.glob('ef*.json'))
        assert len(paths) == expected_cells
        for path in paths:
            metadata = json.loads(path.read_text())
            assert metadata['graph_fingerprint'] == graph['fingerprint']
            assert metadata['codes_sha256'] == pq['codes_sha256']
            assert metadata['codebook_sha256'] == pq['codebook_sha256']
            assert metadata['live_threads'] == metadata['workers'] + 1
            rows = np.genfromtxt(path.with_suffix('.csv'), delimiter=',', names=True)
            assert len(rows) == metadata['queries'] == 40000
            qids = rows['query_id'].astype('i8')
            passes = rows['pass'].astype('i8')
            workers = rows['worker'].astype('i8')
            np.testing.assert_array_equal(np.sort(passes * 10000 + qids), np.arange(40000))
            np.testing.assert_array_equal(workers, qids % metadata['workers'])
            assert np.isfinite(rows['latency_us']).all() and np.all(rows['latency_us'] > 0)
            label = 'all16' if metadata['policy'] else 'native'
            reference = np.fromfile(root / f"recall_ef{metadata['ef']}" / (label + '_ids.i64'), dtype='<i8').reshape(10000, 10)
            actual = np.fromfile(path.with_name(path.stem + '_ids.i64'), dtype='<i8').reshape(40000, 10)
            np.testing.assert_array_equal(actual, reference[qids])
            cells.append(dict(stage=stage, cell=path.name, queries=len(rows)))
    result = dict(status='PASS', frozen_files_checked=len(provenance['sha256']),
                  benchmark_cells=len(cells), timed_queries=sum(c['queries'] for c in cells),
                  every_timed_ID_matches_reference=True, cells=cells)
    with (root / 'final_independent_audit.json').open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != 'cells'}, indent=2))


if __name__ == '__main__':
    main()
