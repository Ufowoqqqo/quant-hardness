#!/usr/bin/env python3
"""Validate and analyze the pre-registered Phase 3A SIFT1M experiment."""

import argparse
import csv
import html
import json
import math
import statistics
from pathlib import Path


DELTAS = ("delta_total", "delta_discovery", "delta_ranking")
DESCRIPTORS = (
    "query_l2_norm",
    "exact_1nn_distance",
    "exact_10nn_distance",
    "nn_margin",
    "normalized_nn_margin",
    "local_distance_concentration",
)


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def percentile(values, probability):
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def recall(ids, truth):
    if len(ids) != 10 or len(set(ids)) != 10:
        raise ValueError("result IDs are not a unique top-10")
    return len(set(ids) & set(truth)) / 10


def ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    output = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        average = (start + stop - 1) / 2
        for index in order[start:stop]:
            output[index] = average
        start = stop
    return output


def spearman(xs, ys):
    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    numerator = sum((x - mx) * (y - my) for x, y in zip(rx, ry))
    denominator = math.sqrt(
        sum((x - mx) ** 2 for x in rx) * sum((y - my) ** 2 for y in ry)
    )
    return numerator / denominator if denominator else None


def read_conf(path):
    values = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0]
        if line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def positive_concentration(values, target):
    positive = sorted((value for value in values if value > 1e-12), reverse=True)
    if not positive:
        return None
    threshold = target * sum(positive)
    cumulative = 0.0
    for count, value in enumerate(positive, 1):
        cumulative += value
        if cumulative + 1e-12 >= threshold:
            return count / len(positive)
    return 1.0


def validate_rows(rows, ef, pq_m, fingerprint, sample_count=100):
    if len(rows) != 10000 or [row["query_id"] for row in rows] != list(range(10000)):
        raise ValueError("decomposition query count/order mismatch")
    for row in rows:
        if row["ef_search"] != ef or row["pq_m"] != pq_m or row["graph_fingerprint"] != fingerprint:
            raise ValueError("decomposition condition metadata mismatch")
        truth = row["ground_truth_ids"]
        mapping = {
            "recall_exact_native": "exact_native_result_ids",
            "recall_pq_native": "pq_native_result_ids",
            "recall_exact_L0_oracle": "exact_L0_oracle_result_ids",
            "recall_pq_L0_oracle": "pq_L0_oracle_result_ids",
        }
        for metric, ids in mapping.items():
            if not math.isclose(row[metric], recall(row[ids], truth), abs_tol=1e-12):
                raise ValueError(f"{metric} does not match IDs")
        expected = {
            "delta_total": row["recall_exact_native"] - row["recall_pq_native"],
            "delta_discovery": row["recall_exact_L0_oracle"] - row["recall_pq_L0_oracle"],
            "delta_ranking": row["recall_pq_L0_oracle"] - row["recall_pq_native"],
            "delta_exact_control": row["recall_exact_L0_oracle"] - row["recall_exact_native"],
        }
        for field, value in expected.items():
            if not math.isclose(row[field], value, abs_tol=1e-12):
                raise ValueError(f"{field} identity mismatch")
        if row["delta_exact_control"] != 0:
            raise ValueError("nonzero exact L0 control")
        exact_mismatch = row["recall_exact_L0_oracle"] != row["exact_L0_candidate_coverage"]
        pq_mismatch = row["recall_pq_L0_oracle"] != row["pq_L0_candidate_coverage"]
        if exact_mismatch != row["exact_oracle_coverage_tie_mismatch"]:
            raise ValueError("exact oracle/coverage tie flag mismatch")
        if pq_mismatch != row["pq_oracle_coverage_tie_mismatch"]:
            raise ValueError("PQ oracle/coverage tie flag mismatch")
        sampled = row["query_id"] < sample_count
        fields = (
            "exact_L0_evaluated_ids", "pq_L0_evaluated_ids",
            "exact_upper_only_evaluated_ids", "pq_upper_only_evaluated_ids",
        )
        if sampled != all(row[field] is not None for field in fields):
            raise ValueError("candidate-set sampling mismatch")
        if sampled:
            for field in fields:
                if row[field] != sorted(set(row[field])):
                    raise ValueError("candidate set is not sorted unique")
            exact, pq = row[fields[0]], row[fields[1]]
            if len(exact) != row["exact_L0_distance_evaluations"] or len(pq) != row["pq_L0_distance_evaluations"]:
                raise ValueError("candidate-set size mismatch")
            if set(exact) & set(row[fields[2]]) or set(pq) & set(row[fields[3]]):
                raise ValueError("upper-only IDs leaked into V_L0")
            intersection = len(set(exact) & set(pq))
            union = len(set(exact) | set(pq))
            if intersection != row["evaluated_L0_intersection_size"]:
                raise ValueError("intersection mismatch")
            if not math.isclose(intersection / union, row["evaluated_L0_jaccard"], abs_tol=1e-12):
                raise ValueError("Jaccard mismatch")


def summarize(ef, rows):
    mean = lambda field: statistics.fmean(row[field] for row in rows)
    exact = mean("recall_exact_native")
    pq_native = mean("recall_pq_native")
    pq_oracle = mean("recall_pq_L0_oracle")
    denominator = exact - pq_native
    output = {
        "ef_search": ef,
        "query_count": len(rows),
        "mean_recall_exact_native": exact,
        "mean_recall_pq_native": pq_native,
        "mean_recall_exact_L0_oracle": mean("recall_exact_L0_oracle"),
        "mean_recall_pq_L0_oracle": pq_oracle,
        "rerank_recovery": (pq_oracle - pq_native) / denominator if abs(denominator) > 1e-12 else None,
        "mean_exact_L0_candidate_coverage": mean("exact_L0_candidate_coverage"),
        "mean_pq_L0_candidate_coverage": mean("pq_L0_candidate_coverage"),
        "mean_exact_L0_distance_evaluations": mean("exact_L0_distance_evaluations"),
        "mean_pq_L0_distance_evaluations": mean("pq_L0_distance_evaluations"),
        "mean_evaluated_L0_jaccard": mean("evaluated_L0_jaccard"),
        "max_abs_delta_exact_control": max(abs(row["delta_exact_control"]) for row in rows),
        "exact_oracle_coverage_tie_mismatch_count": sum(row["exact_oracle_coverage_tie_mismatch"] for row in rows),
        "pq_oracle_coverage_tie_mismatch_count": sum(row["pq_oracle_coverage_tie_mismatch"] for row in rows),
    }
    for field in DELTAS:
        values = [row[field] for row in rows]
        output[f"mean_{field}"] = statistics.fmean(values)
        output[f"median_{field}"] = statistics.median(values)
        for label, probability in (("p90", .90), ("p95", .95), ("p99", .99)):
            output[f"{label}_{field}"] = percentile(values, probability)
    discovery = [row["delta_discovery"] for row in rows]
    output.update({
        "fraction_delta_discovery_positive": sum(x > 1e-12 for x in discovery) / len(rows),
        "fraction_delta_discovery_ge_0p1": sum(x >= .1 - 1e-12 for x in discovery) / len(rows),
        "fraction_delta_discovery_ge_0p2": sum(x >= .2 - 1e-12 for x in discovery) / len(rows),
        "fraction_delta_discovery_zero": sum(abs(x) <= 1e-12 for x in discovery) / len(rows),
        "fraction_delta_discovery_negative": sum(x < -1e-12 for x in discovery) / len(rows),
        "positive_loss_query_fraction_for_50pct": positive_concentration(discovery, .50),
        "positive_loss_query_fraction_for_80pct": positive_concentration(discovery, .80),
        "positive_loss_query_fraction_for_90pct": positive_concentration(discovery, .90),
    })
    return output


def quality_summary(run_root, pq_m):
    distances = read_jsonl(run_root / f"pq_m{pq_m}_distance_samples.jsonl")
    orders = read_jsonl(run_root / f"pq_m{pq_m}_order_samples.jsonl")
    if len(distances) != 10000 or len(orders) != 10000:
        raise ValueError("PQ quality sample count mismatch")
    absolute = []
    relative = []
    for row in distances:
        error = abs(row["pq_distance"] - row["exact_distance"])
        if not math.isclose(error, row["absolute_error"], rel_tol=1e-6, abs_tol=1e-4):
            raise ValueError("distance error mismatch")
        absolute.append(row["absolute_error"])
        relative.append(row["absolute_relative_error"])
    for row in orders:
        inversion = ((row["exact_first"] < row["exact_second"] and row["pq_first"] > row["pq_second"]) or
                     (row["exact_first"] > row["exact_second"] and row["pq_first"] < row["pq_second"]))
        if inversion != row["strict_inversion"]:
            raise ValueError("order inversion mismatch")
    return {
        "pq_m": pq_m,
        "pq_nbits": 8,
        "distance_sample_count": len(distances),
        "order_pair_sample_count": len(orders),
        "mean_absolute_distance_error": statistics.fmean(absolute),
        "median_absolute_distance_error": statistics.median(absolute),
        "p95_absolute_distance_error": percentile(absolute, .95),
        "mean_absolute_relative_distance_error": statistics.fmean(relative),
        "median_absolute_relative_distance_error": statistics.median(relative),
        "pairwise_strict_inversion_rate": statistics.fmean(row["strict_inversion"] for row in orders),
        "pairwise_tie_rate": statistics.fmean(row["pq_tie"] for row in orders),
    }


def svg_line(path, title, series, x_label, y_label):
    width, height, left, right, top, bottom = 840, 500, 82, 28, 48, 62
    all_points = [point for _, points, _ in series for point in points]
    xmin, xmax = min(x for x, _ in all_points), max(x for x, _ in all_points)
    ymin, ymax = min(y for _, y in all_points), max(y for _, y in all_points)
    ypad = max((ymax - ymin) * .12, .002)
    ymin, ymax = ymin - ypad, ymax + ypad
    pw, ph = width - left - right, height - top - bottom
    sx = lambda x: left + (x - xmin) * pw / (xmax - xmin or 1)
    sy = lambda y: top + (ymax - y) * ph / (ymax - ymin or 1)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="{width/2}" y="25" text-anchor="middle" font-family="sans-serif" font-size="17">{html.escape(title)}</text>',
             f'<line x1="{left}" y1="{top+ph}" x2="{left+pw}" y2="{top+ph}" stroke="black"/>',
             f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+ph}" stroke="black"/>',
             f'<text x="{width/2}" y="{height-10}" text-anchor="middle" font-family="sans-serif" font-size="13">{html.escape(x_label)}</text>',
             f'<text x="18" y="{top+ph/2}" transform="rotate(-90 18 {top+ph/2})" text-anchor="middle" font-family="sans-serif" font-size="13">{html.escape(y_label)}</text>']
    for index, (label, points, color) in enumerate(series):
        coords = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in points)
        parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<text x="{left+12}" y="{top+18+index*18}" font-family="sans-serif" font-size="12" fill="{color}">{html.escape(label)}</text>')
        for x, y in points:
            parts.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="3" fill="{color}"/>')
    parts.append("</svg>\n")
    path.write_text("".join(parts), encoding="utf-8")


def svg_scatter(path, title, rows, x_field, y_field, x_label, y_label):
    width, height, left, right, top, bottom = 840, 500, 82, 28, 48, 62
    points = [(row[x_field], row[y_field]) for row in rows]
    xmin, xmax = min(x for x, _ in points), max(x for x, _ in points)
    ymin, ymax = min(y for _, y in points), max(y for _, y in points)
    xpad = max((xmax - xmin) * .03, 1e-6); ypad = max((ymax - ymin) * .08, .01)
    xmin -= xpad; xmax += xpad; ymin -= ypad; ymax += ypad
    pw, ph = width-left-right, height-top-bottom
    sx = lambda x: left+(x-xmin)*pw/(xmax-xmin)
    sy = lambda y: top+(ymax-y)*ph/(ymax-ymin)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">', '<rect width="100%" height="100%" fill="white"/>', f'<text x="{width/2}" y="25" text-anchor="middle" font-family="sans-serif" font-size="17">{html.escape(title)}</text>', f'<line x1="{left}" y1="{top+ph}" x2="{left+pw}" y2="{top+ph}" stroke="black"/>', f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+ph}" stroke="black"/>', f'<text x="{width/2}" y="{height-10}" text-anchor="middle" font-family="sans-serif" font-size="13">{html.escape(x_label)}</text>', f'<text x="18" y="{top+ph/2}" transform="rotate(-90 18 {top+ph/2})" text-anchor="middle" font-family="sans-serif" font-size="13">{html.escape(y_label)}</text>']
    for x, y in points:
        parts.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="1.4" fill="#3366aa" opacity="0.22"/>')
    parts.append("</svg>\n")
    path.write_text("".join(parts), encoding="utf-8")


def svg_ecdf(path, title, series):
    rendered = []
    for label, values, color in series:
        ordered = sorted(values)
        rendered.append((label, [(value, (i + 1) / len(ordered)) for i, value in enumerate(ordered)], color))
    svg_line(path, title, rendered, "signed recall difference", "empirical CDF")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("tables", type=Path)
    parser.add_argument("figures", type=Path)
    args = parser.parse_args()
    args.tables.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)
    selection = read_conf(args.run_root / "selection.conf")
    efs = [int(selection[key]) for key in ("low_ef", "central_ef", "very_high_ef")]
    pq_m = int(selection["primary_pq_m"])
    fingerprint = selection["graph_fingerprint"]
    all_rows, summaries, correlations = {}, [], []
    for ef in efs:
        rows = read_jsonl(args.run_root / f"decomposition_ef{ef}.jsonl")
        validate_rows(rows, ef, pq_m, fingerprint)
        all_rows[ef] = rows
        summaries.append(summarize(ef, rows))
        for descriptor in DESCRIPTORS:
            for delta in ("delta_discovery", "delta_ranking"):
                correlations.append({"ef_search": ef, "descriptor": descriptor,
                                     "outcome": delta,
                                     "spearman": spearman([row[descriptor] for row in rows],
                                                           [row[delta] for row in rows])})
    calibration_native = read_jsonl(args.run_root / f"pq_calibration_queries_m{pq_m}.jsonl")
    if len(calibration_native) != 10000:
        raise ValueError("primary PQ calibration query count mismatch")
    central_rows = all_rows[efs[1]]
    for calibration_row, decomposition_row in zip(calibration_native, central_rows):
        if calibration_row["query_id"] != decomposition_row["query_id"] or calibration_row["ef_search"] != efs[1]:
            raise ValueError("calibration/decomposition query alignment mismatch")
        if calibration_row["pq_result_ids"] != decomposition_row["pq_native_result_ids"]:
            raise ValueError("instrumented PQ IDs differ from calibration IDs")
    exact_calibration = read_csv(args.run_root / "exact_calibration.csv")
    pq_calibration = read_csv(args.run_root / "pq_calibration.csv")
    calibration_rows = []
    for row in exact_calibration:
        if row["graph_fingerprint"] != fingerprint:
            raise ValueError("exact calibration graph fingerprint mismatch")
        calibration_rows.append({
            "calibration_type": "exact_ef_sweep", "ef_search": int(row["ef_search"]),
            "pq_m": "", "pq_nbits": "", "mean_exact_recall": float(row["mean_exact_recall"]),
            "mean_pq_recall": "", "mean_delta": "",
        })
    for row in pq_calibration:
        calibration_rows.append({
            "calibration_type": "pq_rate_sweep", "ef_search": int(row["ef_search"]),
            "pq_m": int(row["pq_m"]), "pq_nbits": int(row["pq_nbits"]),
            "mean_exact_recall": float(row["mean_exact_recall"]),
            "mean_pq_recall": float(row["mean_pq_recall"]),
            "mean_delta": float(row["mean_delta"]),
        })
    qualities = [quality_summary(args.run_root, m) for m in (16, 32, 64)]
    write_csv(args.tables / "phase3a_sift1m_decomposition.csv", summaries)
    write_csv(args.tables / "phase3a_sift1m_pq_quality.csv", qualities)
    write_csv(args.tables / "phase3a_sift1m_spearman.csv", correlations)
    write_csv(args.tables / "phase3a_sift1m_calibration.csv", calibration_rows)
    (args.tables / "phase3a_sift1m_decomposition.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    (args.tables / "phase3a_sift1m_pq_quality.json").write_text(json.dumps(qualities, indent=2) + "\n", encoding="utf-8")
    (args.tables / "phase3a_sift1m_spearman.json").write_text(json.dumps(correlations, indent=2) + "\n", encoding="utf-8")
    (args.tables / "phase3a_sift1m_calibration.json").write_text(json.dumps(calibration_rows, indent=2) + "\n", encoding="utf-8")
    qualified = [row for row in summaries if row["mean_recall_exact_native"] >= .90]
    no_signal = bool(qualified) and all(
        abs(row["mean_delta_discovery"]) <= .01 and
        row["rerank_recovery"] >= .90 for row in qualified)
    emerging = any(row["mean_recall_exact_native"] >= .90 and
                   row["mean_delta_discovery"] >= .03 and
                   row["rerank_recovery"] <= .80 for row in summaries)
    decision = {"no_reproducible_real_data_discovery_signal": no_signal,
                "emerging_real_data_discovery_signal": emerging,
                "strong_real_data_discovery_signal": sum(
                    row["mean_recall_exact_native"] >= .90 and
                    row["mean_delta_discovery"] >= .03 and
                    row["rerank_recovery"] <= .80 for row in summaries) >= 2}
    (args.tables / "phase3a_sift1m_gatekeeper.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    svg_line(args.figures / "phase3a_sift1m_recall_vs_efsearch.svg", "SIFT1M recall versus efSearch", [
        ("exact native", [(r["ef_search"], r["mean_recall_exact_native"]) for r in summaries], "#2255aa"),
        ("PQ native", [(r["ef_search"], r["mean_recall_pq_native"]) for r in summaries], "#bb4422"),
        ("PQ L0 oracle", [(r["ef_search"], r["mean_recall_pq_L0_oracle"]) for r in summaries], "#228844")], "efSearch", "mean Recall@10")
    svg_line(args.figures / "phase3a_sift1m_rerank_recovery.svg", "SIFT1M rerank recovery", [("recovery", [(r["ef_search"], r["rerank_recovery"]) for r in summaries], "#7744aa")], "efSearch", "rerank recovery")
    svg_ecdf(args.figures / "phase3a_sift1m_delta_distributions.svg", "SIFT1M discovery and ranking deltas", [(f"discovery ef={ef}", [r["delta_discovery"] for r in all_rows[ef]], color) for ef, color in zip(efs, ("#2255aa", "#228844", "#8844aa"))] + [(f"ranking ef={efs[1]}", [r["delta_ranking"] for r in all_rows[efs[1]]], "#bb4422")])
    central = central_rows
    svg_scatter(args.figures / "phase3a_sift1m_jaccard_vs_discovery.svg", "L0 evaluated-set overlap versus discovery delta", central, "evaluated_L0_jaccard", "delta_discovery", "Jaccard(V_L0 exact, V_L0 PQ)", "delta discovery")
    svg_scatter(args.figures / "phase3a_sift1m_nn_margin_vs_discovery.svg", "NN margin versus discovery delta", central, "normalized_nn_margin", "delta_discovery", "normalized NN margin", "delta discovery")
    svg_scatter(args.figures / "phase3a_sift1m_query_norm_vs_discovery.svg", "Query norm versus discovery delta", central, "query_l2_norm", "delta_discovery", "query L2 norm", "delta discovery")
    print(json.dumps(decision, sort_keys=True))
    print("status=PASS")


if __name__ == "__main__":
    main()
