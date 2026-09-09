"""Regenerate frozen IDs and verify config/code/ID checksums, no ANN."""
import hashlib
import json
from pathlib import Path
import numpy as np
from phase5a_dataset import split_ids


def main():
    path = Path('runs/phase5a_highdim_external_validity_v1/amendment/dataset_manifest.json')
    m = json.loads(path.read_text())
    c = m['config']
    expected = split_ids(sum(s['rows'] for s in m['shards']), int(c['query_count']),
                         int(c['training_vectors']), int(c['split_seed']), int(c['training_sample_seed']))
    for name, ids in zip(['base', 'query', 'pq_training'], expected):
        rec = m['ids'][name]
        blob = Path(rec['path']).read_bytes()
        assert hashlib.sha256(blob).hexdigest() == rec['sha256']
        np.testing.assert_array_equal(np.frombuffer(blob, dtype='<i8'), ids)
        print(name, rec['count'], rec['sha256'])
    for key, source in [('preprocessing_sha256', 'scripts/phase5a_dataset.py'),
                        ('freezer_sha256', 'scripts/freeze_phase5a_amendment.py'),
                        ('config_sha256', 'configs/indexes/phase5a_highdim_external_validity.conf')]:
        assert hashlib.sha256(Path(source).read_bytes()).hexdigest() == m[key]
    print('PASS: deterministic ID regeneration and immutable code/config checksums')


if __name__ == '__main__':
    main()
