"""Frozen pre-run split/preprocessing; no ANN evaluation or model training."""
import numpy as np


def split_ids(count, query_count, training_count, split_seed, training_seed):
    if not 0 < training_count <= count - query_count < count:
        raise ValueError("invalid role sizes")
    permutation = np.random.Generator(np.random.PCG64(split_seed)).permutation(count)
    base = permutation[:count-query_count].astype('<i8')
    query = permutation[count-query_count:].astype('<i8')
    positions = np.random.Generator(np.random.PCG64(training_seed)).choice(
        len(base), size=training_count, replace=False)
    return base, query, base[positions].astype('<i8')


def normalize_fp32_once(values):
    """Cast first; FP64 reductions/division of FP32 values, final FP32 storage.

    Invoke ONLY on original source rows, then archive/reuse the result. Do not
    renormalize at graph, training, query, GT or refinement entry points.
    """
    x = np.asarray(values, dtype=np.float32)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("expected finite matrix")
    wide = x.astype(np.float64)
    norm = np.sqrt(np.sum(wide * wide, axis=1, dtype=np.float64))
    if np.any(norm == 0):
        raise ValueError("zero vector")
    return np.ascontiguousarray(wide / norm[:, None], dtype='<f4')
