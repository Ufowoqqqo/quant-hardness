"""Reference five-codebook measurements; no search or quantizer implementation."""
import math
import numpy as np
from phase3c_metrics import measure

GEOMETRY = ["d1", "dk", "dk_plus_1", "d20", "boundary_margin",
            "relative_boundary_margin", "boundary_gap_r1", "boundary_gap_r2",
            "boundary_gap_r5", "local_distance_concentration", "candidate_count",
            "oracle_recall"]
CATEGORIES = ["Robust", "Occasionally harmful", "Usually harmful", "Persistently harmful"]
PAIR_CLASSES = ["one-off", "minority", "majority", "persistent"]


def category(count):
    if count not in range(6): raise ValueError("five seeds required")
    return CATEGORIES[0 if count == 0 else 1 if count <= 2 else 2 if count <= 4 else 3]


def pair_class(count):
    if count not in range(1, 6): raise ValueError("nonempty five-seed mask required")
    return PAIR_CLASSES[0 if count == 1 else 1 if count == 2 else 2 if count <= 4 else 3]


def single(ids, exact, score, truth, k, seed, samples, epsilon):
    # Reuse the validated simple Phase 3C implementation without optimizing it.
    row, replacements, _, random = measure(ids, exact, score, score, truth,
                                           k, seed, samples, epsilon)
    metric = {key[:-3]: value for key, value in row.items() if key.endswith("_64")}
    geometry = {key: row[key] for key in GEOMETRY}
    geometry.update(exact_topk_ids=row["exact_topk_ids"],
                    exact_topk_set=row["exact_topk_set"],
                    exact_boundary_tie=row["exact_boundary_tie"])
    return metric, geometry, [p for p in replacements if p["quantizer"] == 64], random


def measure_seeds(ids, exact, scores, truth, k=10, seed=1, samples=512, epsilon=1e-12):
    ids = np.asarray(ids, dtype=np.int64); d = np.asarray(exact, dtype=np.float64)
    p = np.asarray(scores, dtype=np.float64)
    if p.shape != (5, len(ids)): raise ValueError("expected five aligned score arrays")
    models, replacements, geometry = [], [], None
    for s in range(5):
        metric, g, pairs, random = single(ids, d, p[s], truth, k, seed, samples, epsilon)
        if geometry is not None: assert geometry == g
        geometry = g; metric["model"] = s; models.append(metric)
        for pair in pairs:
            pair.pop("quantizer"); pair["model"] = s; replacements.append(pair)
    order = np.argsort(d, kind="stable"); inside, outside = order[:k], order[k:]
    margin = d[outside][None, :] - d[inside][:, None]
    inversions = np.array([(margin > 0) & (p[s, inside, None] > p[s, outside][None, :]) for s in range(5)])
    frequency = inversions.sum(axis=0)
    pairs = []
    for a, b in zip(*np.nonzero(frequency)):
        i, j = inside[a], outside[b]; count = int(frequency[a, b])
        mask = [bool(v) for v in inversions[:, a, b]]
        pairs.append({"inside_id": int(ids[i]), "outside_id": int(ids[j]),
                      "inside_exact_rank": int(a+1), "outside_exact_rank": int(b+k+1),
                      "exact_margin": float(margin[a, b]),
                      "violation_by_seed": (p[:, i]-p[:, j]).tolist(),
                      "mask": mask, "frequency_count": count,
                      "inversion_frequency": count/5, "category": pair_class(count)})
    lookup = {int(v): i for i, v in enumerate(ids)}
    frequency_lookup = {(r["inside_id"], r["outside_id"]): r["frequency_count"] for r in pairs}
    gt = set(truth); A = set(geometry["exact_topk_ids"])
    # Exact accounting of retrieval losses, not a causal allocation.
    for s, metric in enumerate(models):
        attribution = {key: 0.0 for key in PAIR_CLASSES+[
            "unassigned_tie_or_no_strict_overtake", "gt_intruder_credit"]}
        S = set(metric["topk_ids"])
        for i_id in (A-S) & gt:
            i = lookup[i_id]
            responsible = [j_id for j_id in S-A if d[i] < d[lookup[j_id]] and p[s, i] > p[s, lookup[j_id]]]
            if responsible:
                for j_id in responsible:
                    attribution[pair_class(frequency_lookup[i_id, j_id])] += 1/k/len(responsible)
            else: attribution["unassigned_tie_or_no_strict_overtake"] += 1/k
        attribution["gt_intruder_credit"] = -len((S-A) & gt)/k
        assert abs(sum(attribution.values())-metric["ranking_recall_loss"]) < 1e-10
        metric["loss_attribution"] = attribution
        metric["has_high_frequency_inversion"] = bool(np.any(inversions[s] & (frequency >= 3)))
        assert int(inversions[s].sum()) == metric["cross_boundary_inversion_count"]
    losses = np.array([r["ranking_recall_loss"] for r in models])
    counts = np.array([r["cross_boundary_inversion_count"] for r in models])
    harmful = int((losses > 0).sum())
    row = {**geometry, "harmful_count": harmful, "harmful_frequency": harmful/5,
           "category": category(harmful), "mean_ranking_loss": float(losses.mean()),
           "std_ranking_loss": float(losses.std(ddof=1)), "max_ranking_loss": float(losses.max()),
           "mean_critical_inversions": float(counts.mean()),
           "std_critical_inversions": float(counts.std(ddof=1)), "models": models,
           "union_inversion_count": len(pairs),
           "inversion_pair_frequency_counts": [int((frequency==i).sum()) for i in range(1,6)]}
    return row, replacements, pairs, random


def severity_overlap(a, b, fraction):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) != len(b) or not len(a): raise ValueError("unaligned losses")
    n = math.ceil(fraction*len(a)); sets, weights, inclusive, cutoffs, ties = [], [], [], [], []
    for x in (a, b):
        order = np.lexsort((np.arange(len(x)), -x)); cutoff = x[order[n-1]]
        strict = x > cutoff; tie = x == cutoff
        weights.append(strict.astype(float)+tie*(n-int(strict.sum()))/int(tie.sum()))
        sets.append(set(order[:n])); inclusive.append(set(np.where(x >= cutoff)[0]))
        cutoffs.append(float(cutoff)); ties.append(int(tie.sum()))
    return {"top_fraction": fraction, "count_per_set": n,
            "overlap_fraction": len(sets[0]&sets[1])/n,
            "independent_tie_resolution_expected_overlap": float(np.dot(*weights)/n),
            "tie_inclusive_jaccard": len(inclusive[0]&inclusive[1])/len(inclusive[0]|inclusive[1]),
            "cutoff_a": cutoffs[0], "cutoff_b": cutoffs[1],
            "cutoff_ties_a": ties[0], "cutoff_ties_b": ties[1]}
