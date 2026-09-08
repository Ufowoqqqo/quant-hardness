"""Reference fixed-pool residual randomization; no ANN or PQ implementation."""
import numpy as np


def permutations(n, count, seed, model, query):
    return np.array([np.random.Generator(np.random.PCG64(
        np.random.SeedSequence([seed, model, query, p]))).permutation(n)
        for p in range(count)])


def evaluate(ids, exact, scores, truth, k=10):
    ids = np.asarray(ids); d = np.asarray(exact, dtype=np.float64)
    s = np.atleast_2d(np.asarray(scores, dtype=np.float64))
    if len(ids) <= k or len(np.unique(ids)) != len(ids) or s.shape[1] != len(ids):
        raise ValueError('invalid aligned unique candidate pool')
    if not np.isfinite(d).all() or not np.isfinite(s).all():
        raise ValueError('nonfinite distances')
    order = np.argsort(d, kind='stable'); a, b = order[:k], order[k:]
    top = np.argsort(s, axis=1, kind='stable')[:, :k]
    gt = np.isin(ids, truth); inside = np.zeros(len(ids), dtype=bool); inside[a] = True
    hits = gt[top].sum(1); agree = inside[top].sum(1)
    inversion = ((d[a, None] < d[b][None, :])[None, :, :] &
                 (s[:, a, None] > s[:, None, b])).sum((1, 2))
    return {'topk': ids[top].astype('<i4'), 'hits': hits.astype('<i2'),
            'loss_hits': (gt[a].sum()-hits).astype('<i2'),
            'agreement_hits': agree.astype('<i2'), 'displaced_count': (k-agree).astype('<i2'),
            'inversions': inversion.astype('<i4')}


def scalar_reference(ids, exact, score, truth, k=10):
    order = sorted(range(len(ids)), key=lambda i: exact[i]); a = order[:k]; b = order[k:]
    top = sorted(range(len(ids)), key=lambda i: score[i])[:k]
    hits = len(set(ids[i] for i in top) & set(truth))
    ah = len(set(ids[i] for i in a) & set(truth)); agreement = len(set(top) & set(a))
    return {'topk': np.array([ids[i] for i in top], dtype='<i4'), 'hits': hits,
            'loss_hits': ah-hits, 'agreement_hits': agreement, 'displaced_count': k-agreement,
            'inversions': sum(exact[i] < exact[j] and score[i] > score[j] for i in a for j in b)}


def summary(x):
    a = np.asarray(x, dtype=float)
    return {'mean': float(a.mean()), 'std': float(a.std(ddof=1)),
            'median': float(np.median(a)), 'p05': float(np.quantile(a, .05)),
            'p95': float(np.quantile(a, .95))}


def empirical_rank(observed, null):
    x = np.asarray(null); below = int((x < observed).sum()); equal = int((x == observed).sum())
    return {'below': below, 'equal': equal, 'above': len(x)-below-equal,
            'lower': below/len(x), 'upper': (below+equal)/len(x),
            'midrank': (below+.5*equal)/len(x)}


def pair_diagnostics(ids, d, e, shuffled, observed_top, null_top, k=10):
    order = np.argsort(d, kind='stable'); inside = set(order[:k]); lookup = dict(zip(ids.tolist(), range(len(ids))))
    native = {lookup[int(v)] for v in observed_top}; displaced = sorted(inside-native); intruders = sorted(native-inside)
    rows = []
    for i in displaced:
        for j in intruders:
            diffs = shuffled[:, i]-shuffled[:, j]; margin = float(d[j]-d[i])
            rows.append({'displaced_id': int(ids[i]), 'intruder_id': int(ids[j]),
                         'exact_margin': margin, 'observed_error_difference': float(e[i]-e[j]),
                         'observed_violation': float(e[i]-e[j]-margin),
                         'null_same_pair_difference': summary(diffs),
                         'null_same_pair_strict_inversion_fraction': float(np.mean(diffs > margin)) if margin > 0 else 0.})
    selected = []
    for p, chosen in enumerate(null_top):
        s = {lookup[int(v)] for v in chosen}; left = sorted(inside-s); right = sorted(s-inside)
        diffs = [shuffled[p, i]-shuffled[p, j] for i in left for j in right]
        margins = [d[j]-d[i] for i in left for j in right]
        selected.append([len(diffs), float(sum(diffs)), float(sum(margins))])
    return rows, np.asarray(selected)
