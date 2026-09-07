#!/usr/bin/env python3
"""Validate and summarize Phase 1 candidate-discovery/ranking decomposition."""

import argparse
import csv
import html
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


COLORS = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#b279a2", "#72b7b2"]


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def recall(result_ids, truth_ids):
    if len(result_ids) != 10 or len(set(result_ids)) != 10:
        raise ValueError("result IDs are not a unique top-10")
    if len(truth_ids) != 10 or len(set(truth_ids)) != 10:
        raise ValueError("ground truth is not a unique top-10")
    return len(set(result_ids) & set(truth_ids)) / 10


def percentile(values, probability):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


def pearson(first, second):
    mean_first = statistics.fmean(first)
    mean_second = statistics.fmean(second)
    numerator = sum((x - mean_first) * (y - mean_second) for x, y in zip(first, second))
    denominator = math.sqrt(sum((x - mean_first) ** 2 for x in first) * sum((y - mean_second) ** 2 for y in second))
    return numerator / denominator if denominator else None


def validate(rows, calibration_rows, manifest):
    expected_ef = manifest["search"]["ef_search_values"]
    query_count = manifest["dataset"]["queries"]
    reference = {(row["ef_search"], row["query_id"]): row for row in calibration_rows if row["pq_id"] == "pq_m32_nbits8" and row["ef_search"] in expected_ef}
    groups = defaultdict(list)
    for row in rows:
        ef = row["ef_search"]
        query_id = row["query_id"]
        if row["query_order"] != query_id:
            raise ValueError(f"query order mismatch: ef={ef}, query={query_id}")
        exact_set = row["exact_evaluated_ids"]
        pq_set = row["pq_evaluated_ids"]
        if exact_set != sorted(set(exact_set)) or pq_set != sorted(set(pq_set)):
            raise ValueError(f"evaluated IDs are not sorted unique sets: {ef}, {query_id}")
        if len(exact_set) != row["exact_unique_distance_evaluations"] or len(pq_set) != row["pq_unique_distance_evaluations"]:
            raise ValueError(f"evaluation count mismatch: {ef}, {query_id}")
        intersection = len(set(exact_set) & set(pq_set))
        union = len(set(exact_set) | set(pq_set))
        if intersection != row["evaluated_intersection_size"] or not math.isclose(intersection / union, row["evaluated_jaccard"], abs_tol=1e-12):
            raise ValueError(f"set overlap mismatch: {ef}, {query_id}")

        truth = row["ground_truth_ids"]
        metrics = {
            "recall_exact_native": recall(row["exact_native_result_ids"], truth),
            "recall_pq_native": recall(row["pq_native_result_ids"], truth),
            "recall_exact_oracle": recall(row["exact_oracle_result_ids"], truth),
            "recall_pq_oracle": recall(row["pq_oracle_result_ids"], truth),
            "coverage_exact": len(set(exact_set) & set(truth)) / 10,
            "coverage_pq": len(set(pq_set) & set(truth)) / 10,
        }
        for field, value in metrics.items():
            if not math.isclose(value, row[field], abs_tol=1e-12):
                raise ValueError(f"{field} mismatch: {ef}, {query_id}")
        if not math.isclose(row["recall_exact_oracle"], row["coverage_exact"], abs_tol=1e-12) or not math.isclose(row["recall_pq_oracle"], row["coverage_pq"], abs_tol=1e-12):
            raise ValueError(f"oracle/coverage equality failed: {ef}, {query_id}")
        expected_deltas = {
            "delta_total": row["recall_exact_native"] - row["recall_pq_native"],
            "delta_discovery": row["recall_exact_oracle"] - row["recall_pq_oracle"],
            "delta_ranking": row["recall_pq_oracle"] - row["recall_pq_native"],
            "delta_exact_control": row["recall_exact_oracle"] - row["recall_exact_native"],
        }
        for field, value in expected_deltas.items():
            if not math.isclose(value, row[field], abs_tol=1e-12):
                raise ValueError(f"{field} mismatch: {ef}, {query_id}")
        if not math.isclose(row["delta_total"], row["delta_discovery"] + row["delta_ranking"] - row["delta_exact_control"], abs_tol=1e-12):
            raise ValueError(f"decomposition identity failed: {ef}, {query_id}")

        previous = reference.get((ef, query_id))
        if previous is None:
            raise ValueError(f"missing calibration reference: {ef}, {query_id}")
        if row["exact_native_result_ids"] != previous["exact_result_ids"] or row["pq_native_result_ids"] != previous["pq_result_ids"] or truth != previous["ground_truth_ids"]:
            raise ValueError(f"instrumented native IDs differ from calibration: {ef}, {query_id}")
        groups[ef].append(row)
    for ef in expected_ef:
        if [row["query_id"] for row in groups[ef]] != list(range(query_count)):
            raise ValueError(f"incomplete query sequence at efSearch={ef}")
    return groups


def delta_summary(rows, field):
    values = [row[field] for row in rows]
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "fraction_equal_zero": sum(value == 0 for value in values) / len(values),
        "fraction_greater_zero": sum(value > 0 for value in values) / len(values),
        "fraction_less_zero": sum(value < 0 for value in values) / len(values),
    }


def summarize(groups):
    summaries = []
    for ef, rows in sorted(groups.items()):
        mean = lambda field: statistics.fmean(row[field] for row in rows)
        denominator = mean("recall_exact_native") - mean("recall_pq_native")
        recovery = (mean("recall_pq_oracle") - mean("recall_pq_native")) / denominator if abs(denominator) > 1e-12 else None
        total = [row["delta_total"] for row in rows]
        discovery = [row["delta_discovery"] for row in rows]
        ranking = [row["delta_ranking"] for row in rows]
        summaries.append({
            "ef_search": ef,
            "mean_recall_exact_native": mean("recall_exact_native"),
            "mean_recall_pq_native": mean("recall_pq_native"),
            "mean_recall_exact_oracle": mean("recall_exact_oracle"),
            "mean_recall_pq_oracle": mean("recall_pq_oracle"),
            "mean_coverage_exact": mean("coverage_exact"),
            "mean_coverage_pq": mean("coverage_pq"),
            "mean_exact_unique_distance_evaluations": mean("exact_unique_distance_evaluations"),
            "mean_pq_unique_distance_evaluations": mean("pq_unique_distance_evaluations"),
            "mean_evaluated_jaccard": mean("evaluated_jaccard"),
            "rerank_recovery": recovery,
            "max_absolute_delta_exact_control": max(abs(row["delta_exact_control"]) for row in rows),
            "fraction_pq_candidate_coverage_better": sum(row["delta_discovery"] < 0 for row in rows) / len(rows),
            "fraction_exact_candidate_coverage_better": sum(row["delta_discovery"] > 0 for row in rows) / len(rows),
            "fraction_candidate_coverage_equal": sum(row["delta_discovery"] == 0 for row in rows) / len(rows),
            "correlation_total_with_discovery": pearson(total, discovery),
            "correlation_total_with_ranking": pearson(total, ranking),
            "delta_total": delta_summary(rows, "delta_total"),
            "delta_discovery": delta_summary(rows, "delta_discovery"),
            "delta_ranking": delta_summary(rows, "delta_ranking"),
        })
    return summaries


def canvas(title, x_label, y_label, x_min, x_max, y_min, y_max):
    if x_min == x_max: x_min, x_max = x_min - 0.05, x_max + 0.05
    if y_min == y_max: y_min, y_max = y_min - 0.05, y_max + 0.05
    width, height, left, right, top, bottom = 800, 500, 76, 24, 42, 64
    plot_w, plot_h = width - left - right, height - top - bottom
    sx = lambda x: left + (x - x_min) * plot_w / (x_max - x_min)
    sy = lambda y: top + (y_max - y) * plot_h / (y_max - y_min)
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="800" height="500" viewBox="0 0 800 500">', '<rect width="100%" height="100%" fill="white"/>', f'<text x="400" y="24" text-anchor="middle" font-family="sans-serif" font-size="17">{html.escape(title)}</text>', f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="black"/>', f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="black"/>', f'<text x="{left + plot_w / 2}" y="486" text-anchor="middle" font-family="sans-serif" font-size="13">{html.escape(x_label)}</text>', f'<text x="18" y="{top + plot_h / 2}" text-anchor="middle" transform="rotate(-90 18 {top + plot_h / 2})" font-family="sans-serif" font-size="13">{html.escape(y_label)}</text>']
    for i in range(6):
        x = x_min + (x_max - x_min) * i / 5
        y = y_min + (y_max - y_min) * i / 5
        parts.extend([f'<text x="{sx(x):.2f}" y="{top + plot_h + 20}" text-anchor="middle" font-family="sans-serif" font-size="11">{x:.3g}</text>', f'<text x="{left - 8}" y="{sy(y) + 4:.2f}" text-anchor="end" font-family="sans-serif" font-size="11">{y:.3g}</text>'])
    return parts, sx, sy


def add_scatter(parts, sx, sy, points, color):
    for x, y in points:
        parts.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="3" fill="{color}" fill-opacity="0.35"/>')


def add_line(parts, sx, sy, points, color, label, legend_index):
    encoded = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in points)
    parts.append(f'<polyline points="{encoded}" fill="none" stroke="{color}" stroke-width="2"/>')
    for x, y in points: parts.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="3" fill="{color}"/>')
    y = 54 + legend_index * 18
    parts.extend([f'<line x1="570" y1="{y}" x2="595" y2="{y}" stroke="{color}" stroke-width="2"/>', f'<text x="601" y="{y + 4}" font-family="sans-serif" font-size="11">{html.escape(label)}</text>'])


def write_svg(path, parts):
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def figures(directory, groups, summaries):
    directory.mkdir(parents=True, exist_ok=True)
    paths = [directory / f"phase1_decomposition_discovery_vs_ranking_ef{ef}.svg" for ef in sorted(groups)]
    paths += [directory / "phase1_decomposition_oracle_vs_native.svg", directory / "phase1_decomposition_jaccard_vs_total.svg", directory / "phase1_decomposition_mean_recall_vs_efsearch.svg", directory / "phase1_decomposition_delta_cdfs.svg"]
    for path in paths:
        if path.exists(): raise FileExistsError(f"refusing to overwrite {path}")

    for path, (ef, rows) in zip(paths[:3], sorted(groups.items())):
        points = [(row["delta_discovery"], row["delta_ranking"]) for row in rows]
        x = [p[0] for p in points]; y = [p[1] for p in points]
        parts, sx, sy = canvas(f"Discovery versus ranking delta, efSearch={ef}", "delta_discovery", "delta_ranking", min(x) - 0.02, max(x) + 0.02, min(y) - 0.02, max(y) + 0.02)
        add_scatter(parts, sx, sy, points, COLORS[0]); write_svg(path, parts)

    all_rows = [row for rows in groups.values() for row in rows]
    parts, sx, sy = canvas("PQ oracle recall versus PQ native recall", "recall_pq_oracle", "recall_pq_native", 0.55, 1.01, 0.35, 1.01)
    for i, (ef, rows) in enumerate(sorted(groups.items())): add_scatter(parts, sx, sy, [(row["recall_pq_oracle"], row["recall_pq_native"]) for row in rows], COLORS[i])
    write_svg(paths[3], parts)

    x = [row["evaluated_jaccard"] for row in all_rows]; y = [row["delta_total"] for row in all_rows]
    parts, sx, sy = canvas("Evaluated-set overlap versus total delta", "Jaccard(V_E, V_P)", "delta_total", min(x) - 0.01, max(x) + 0.01, min(y) - 0.02, max(y) + 0.02)
    for i, (ef, rows) in enumerate(sorted(groups.items())): add_scatter(parts, sx, sy, [(row["evaluated_jaccard"], row["delta_total"]) for row in rows], COLORS[i])
    write_svg(paths[4], parts)

    ef_values = [row["ef_search"] for row in summaries]
    parts, sx, sy = canvas("Mean recall versus efSearch", "efSearch", "mean Recall@10", min(ef_values), max(ef_values), 0.75, 1.0)
    for i, (field, label) in enumerate((("mean_recall_exact_native", "exact native"), ("mean_recall_pq_native", "PQ native"), ("mean_recall_pq_oracle", "PQ traversal + exact rerank"))):
        add_line(parts, sx, sy, [(row["ef_search"], row[field]) for row in summaries], COLORS[i], label, i)
    write_svg(paths[5], parts)

    values = [row[field] for rows in groups.values() for row in rows for field in ("delta_discovery", "delta_ranking")]
    parts, sx, sy = canvas("CDFs of discovery and ranking delta", "delta", "fraction of queries at or below x", min(values) - 0.02, max(values) + 0.02, 0, 1)
    legend = 0
    for ef, rows in sorted(groups.items()):
        for field, suffix in (("delta_discovery", "discovery"), ("delta_ranking", "ranking")):
            ordered = sorted(row[field] for row in rows)
            add_line(parts, sx, sy, [(value, (rank + 1) / len(ordered)) for rank, value in enumerate(ordered)], COLORS[legend], f"ef={ef} {suffix}", legend)
            legend += 1
    write_svg(paths[6], parts)


def flatten_summary(summary):
    row = {key: value for key, value in summary.items() if not isinstance(value, dict)}
    for section in ("delta_total", "delta_discovery", "delta_ranking"):
        for key, value in summary[section].items(): row[f"{section}_{key}"] = value
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("calibration_queries", type=Path)
    parser.add_argument("table_directory", type=Path)
    parser.add_argument("figure_directory", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.run_directory / "manifest.json").read_text(encoding="utf-8"))
    groups = validate(read_jsonl(args.run_directory / "queries.jsonl"), read_jsonl(args.calibration_queries), manifest)
    summaries = summarize(groups)

    args.table_directory.mkdir(parents=True, exist_ok=True)
    json_path = args.table_directory / "phase1_decomposition.json"
    csv_path = args.table_directory / "phase1_decomposition.csv"
    for path in (json_path, csv_path):
        if path.exists(): raise FileExistsError(f"refusing to overwrite {path}")
    payload = {"schema_version": 1, "source_run": str(args.run_directory), "calibration_reference": str(args.calibration_queries), "graph_fingerprint": manifest["graph_fingerprint"], "native_rows_matching_calibration": len(groups) * manifest["dataset"]["queries"], "summaries": summaries}
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    flattened = [flatten_summary(summary) for summary in summaries]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flattened[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(flattened)
    figures(args.figure_directory, groups, summaries)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
