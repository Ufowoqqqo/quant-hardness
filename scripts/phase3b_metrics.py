"""Simple fixed-candidate reference measurements; no ANN search or fitting."""
import numpy as np


def measure(ids, exact, pq, native, truth, k, seed, pair_count, epsilon=1e-12):
    ids = np.asarray(ids, dtype=np.int64)
    exact = np.asarray(exact, dtype=np.float64)
    pq = np.asarray(pq, dtype=np.float64)
    native = np.asarray(native, dtype=np.int64)
    if (len(ids) <= k or len(set(ids)) != len(ids) or len(native) != k
            or len(set(native)) != k or not set(native) <= set(ids)
            or not np.isfinite(exact).all() or not np.isfinite(pq).all()):
        raise ValueError("invalid fixed candidate inputs")
    order = np.argsort(exact, kind="stable")
    po = np.argsort(pq, kind="stable")
    top, outside = order[:k], order[k:]
    t_exact, t_pq = ids[top], ids[po[:k]]
    locations = {int(v): i for i, v in enumerate(ids)}
    nscores = pq[[locations[int(v)] for v in native]]
    tie_equivalent = np.array_equal(nscores, pq[po[:k]])
    set_equal = set(native) == set(t_pq)
    list_equal = np.array_equal(native, t_pq)
    if tie_equivalent and not set_equal:
        exchanged = set(native) ^ set(t_pq)
        tie_equivalent = all(pq[locations[int(v)]] == pq[po[k-1]] for v in exchanged)
    truth = set(truth)
    hits_exact, hits_pq = len(set(t_exact) & truth), len(set(t_pq) & truth)
    loss_hits = hits_exact - hits_pq
    agreement = len(set(t_exact) & set(t_pq)) / k
    category = ("Stable" if agreement == 1 else "Harmful" if loss_hits > 0
                else "Lucky" if loss_hits < 0 else "Benign")
    error = pq - exact
    abs_error = np.abs(error)
    sd = exact[order]
    margin = float(sd[k] - sd[k-1])
    window = order[max(0, k-6):min(len(ids), k+5)]
    scale = float(abs_error[window].mean())
    advantage = float(error[order[k-1]] - error[order[k]])
    margins = exact[outside][None, :] - exact[top][:, None]
    advantages = error[top][:, None] - error[outside][None, :]
    violation = advantages - margins
    inversions = (margins > 0) & (pq[top][:, None] > pq[outside][None, :])
    overtakes = inversions.sum(axis=1)
    # Distinct items within each draw; draws may repeat. Persist these indices.
    rng = np.random.Generator(np.random.PCG64(seed))
    first = rng.integers(0, len(ids), pair_count)
    second = rng.integers(0, len(ids)-1, pair_count)
    second += second >= first
    inv = ((exact[first] < exact[second]) & (pq[first] > pq[second])) | (
        (exact[first] > exact[second]) & (pq[first] < pq[second]))
    row = {
        "candidate_count": len(ids), "native_set_equal": bool(set_equal),
        "native_list_equal": bool(list_equal),
        "native_order_equal_modulo_pq_ties": bool(tie_equivalent),
        "pq_topk_score_tie": bool(np.any(np.diff(pq[po[:k+1]]) == 0)),
        "exact_boundary_tie": bool(margin == 0),
        "pq_boundary_tie": bool(pq[po[k]] == pq[po[k-1]]),
        "topk_agreement": agreement, "ranking_disagreement": 1-agreement,
        "oracle_recall": hits_exact/k, "pq_fixed_recall": hits_pq/k,
        "native_recall": len(set(native) & truth)/k,
        "ranking_recall_loss": loss_hits/k, "class": category,
        "d1": float(sd[0]), "dk": float(sd[k-1]), "dk_plus_1": float(sd[k]),
        "d20": float(sd[19]) if len(ids) >= 20 else None,
        "boundary_margin": margin,
        "relative_boundary_margin": margin / max(float(sd[k-1]), epsilon),
        "candidate_mae": float(abs_error.mean()),
        "candidate_median_absolute_error": float(np.median(abs_error)),
        "candidate_p95_absolute_error": float(np.quantile(abs_error, .95)),
        "boundary_local_mae": scale,
        "boundary_local_max_absolute_error": float(abs_error[window].max()),
        "boundary_window_ranks": list(range(max(1,k-5), min(len(ids),k+5)+1)),
        "boundary_window_absolute_errors": abs_error[window].tolist(),
        "error_at_k": float(error[order[k-1]]),
        "error_at_k_plus_1": float(error[order[k]]),
        "boundary_error_advantage": advantage,
        "boundary_overtaken": bool(advantage > margin),
        "error_scale_over_boundary_margin": scale/margin if margin > 0 else None,
        "cross_boundary_inversion_count": int(inversions.sum()),
        "cross_boundary_inversion_fraction": float(inversions.mean()),
        "cross_boundary_exact_tie_count": int((margins == 0).sum()),
        "topk_items_overtaken": int((overtakes > 0).sum()),
        "max_outside_overtaking_one_topk": int(overtakes.max()),
        "maximum_violation": float(violation.max()),
        "random_pair_count": pair_count, "random_inversion_count": int(inv.sum()),
        "random_inversion_rate": float(inv.mean()),
        "random_exact_ties": int((exact[first] == exact[second]).sum()),
        "random_pq_ties": int((pq[first] == pq[second]).sum()),
        "exact_topk_ids": t_exact.tolist(), "pq_global_topk_ids": t_pq.tolist(),
        "native_ids": native.tolist(),
        "displaced_ids": sorted(set(t_exact.tolist())-set(t_pq.tolist())),
        "intruder_ids": sorted(set(t_pq.tolist())-set(t_exact.tolist())),
    }
    for r in (1,2,5):
        row[f"boundary_gap_r{r}"] = (float(sd[k+r-1]-sd[k-r])
                                      if k >= r and len(ids) >= k+r else None)
    pair_rows = []
    if category == "Harmful":
        for displaced in row["displaced_ids"]:
            for intruder in row["intruder_ids"]:
                i, j = locations[displaced], locations[intruder]
                m, a = float(exact[j]-exact[i]), float(error[i]-error[j])
                pair_rows.append({"displaced_id": displaced, "intruder_id": intruder,
                                  "displaced_is_gt": displaced in truth,
                                  "intruder_is_gt": intruder in truth,
                                  "exact_margin": m, "error_difference": a,
                                  "violation": a-m, "strict_inversion": bool(m>0 and pq[i]>pq[j])})
    return row, pair_rows, np.column_stack([first, second]).astype("<i4")
