#!/usr/bin/env python3
"""Read/hash all local shards, validate vectors, freeze role IDs; no retrieval."""
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pyarrow as pa
from phase5a_dataset import normalize_fp32_once, split_ids


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    config_path = Path('configs/indexes/phase5a_highdim_external_validity.conf')
    c = dict(line.split('=', 1) for line in config_path.read_text().splitlines()
             if line and not line.startswith('#'))
    root = Path(c['run']) / 'amendment'
    root.mkdir(exist_ok=False)
    cache = Path(c['dataset_cache'])
    info = json.loads((cache/'dataset_info.json').read_text())
    paths = sorted(cache.glob('*.arrow'))
    assert len(paths) == 26
    column = 'text-embedding-3-large-1536-embedding'
    total, records, max_norm_error = 0, [], 0.0
    for number, path in enumerate(paths):
        assert path.name.endswith(f'{number:05d}-of-00026.arrow')
        before = path.stat()
        fingerprint = sha(path)
        rows = 0
        with pa.memory_map(str(path), 'r') as source:
            for batch in pa.ipc.open_stream(source):
                vectors = batch.column(batch.schema.get_field_index(column))
                assert vectors.null_count == 0
                offsets = vectors.offsets.to_numpy()
                assert np.all(np.diff(offsets) == 1536)
                flat = vectors.values.slice(int(offsets[0]), int(offsets[-1]-offsets[0]))
                assert flat.null_count == 0
                values = flat.to_numpy().reshape(-1, 1536)
                normalized = normalize_fp32_once(values)
                error = np.max(np.abs(np.linalg.norm(normalized.astype(np.float64), axis=1)-1))
                assert error <= 2e-6
                max_norm_error = max(max_norm_error, float(error))
                rows += len(batch)
        assert rows == info['splits']['train']['shard_lengths'][number]
        after = path.stat()
        assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
        records.append(dict(path=str(path), sha256=fingerprint, bytes=after.st_size,
                            rows=rows, row_start=total, row_end_exclusive=total+rows))
        total += rows
        print(f'shard {number+1}/26 validated; total rows={total}', flush=True)
    assert total == 1000000
    base, query, training = split_ids(total, int(c['query_count']), int(c['training_vectors']),
                                     int(c['split_seed']), int(c['training_sample_seed']))
    assert len(np.unique(np.concatenate([base, query]))) == total
    assert np.isin(training, base).all() and not np.isin(training, query).any()
    ids = {}
    for name, values in [('base', base), ('query', query), ('pq_training', training)]:
        path = root / (name+'_source_ids.i64')
        with path.open('xb') as f:
            values.tofile(f)
        ids[name] = dict(path=str(path), count=len(values), sha256=sha(path), dtype='<i8')
    manifest = dict(status='LOCAL_SHARDS_AND_SPLIT_VALIDATED_NO_RETRIEVAL',
                    timestamp_utc=datetime.now(timezone.utc).isoformat(),
                    parent_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
                    config=c, config_sha256=sha(config_path),
                    source_repository=c['dataset'], revision=c['dataset_revision'],
                    dataset_info_sha256=sha(cache/'dataset_info.json'), shards=records,
                    row_identity='concatenate numbered Arrow shards, then record batches/rows in stored order',
                    ids=ids, source_dtype='float64', output_dtype='float32', dimension=1536,
                    max_normalized_norm_error=max_norm_error,
                    numpy=np.__version__, pyarrow=pa.__version__, machine=platform.uname()._asdict(),
                    preprocessing_sha256=sha('scripts/phase5a_dataset.py'),
                    freezer_sha256=sha(__file__),
                    validation_scope='All shards decoded, dimensions/nulls/finite/nonzero/norm checked; '
                    'local checksums are not comparison with a publisher-signed checksum. '
                    'Normalized batches were discarded, not ANN inputs; no GT or retrieval run.')
    with (root/'dataset_manifest.json').open('x') as f:
        json.dump(manifest, f, indent=2)
        f.write('\n')
    print('PASS: immutable role IDs and local source checksums recorded; no retrieval', flush=True)


if __name__ == '__main__':
    main()
