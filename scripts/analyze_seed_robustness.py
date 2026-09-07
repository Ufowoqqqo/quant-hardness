#!/usr/bin/env python3
"""Validate and aggregate the pre-registered Phase 1 seed robustness run."""

import argparse
import csv
import html
import json
import math
import statistics
from pathlib import Path


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def recall(result_ids, truth_ids):
    if len(result_ids) != 10 or len(set(result_ids)) != 10:
        raise ValueError("result IDs are not a unique top-10")
    return len(set(result_ids) & set(truth_ids)) / 10


def validate_replicate(rows, manifest, expected_query_count):
    if manifest["search"]["ef_search_values"] != [256]:
        raise ValueError("replicate did not use efSearch=256")
    if manifest["pq"]["M"] != 32 or manifest["pq"]["nbits"] != 8:
        raise ValueError("replicate did not use PQ32x8")
    if [row["query_id"] for row in rows] != list(range(expected_query_count)):
        raise ValueError("replicate query order is incomplete")
    for row in rows:
        truth = row["ground_truth_ids"]
        exact_set = row["exact_evaluated_ids"]
        pq_set = row["pq_evaluated_ids"]
        if exact_set != sorted(set(exact_set)) or pq_set != sorted(set(pq_set)):
            raise ValueError("evaluated candidate IDs are not sorted unique sets")
        if len(exact_set) != row["exact_unique_distance_evaluations"] or len(pq_set) != row["pq_unique_distance_evaluations"]:
            raise ValueError("unique distance-evaluation count mismatch")
        values = {
            "recall_exact_native": recall(row["exact_native_result_ids"], truth),
            "recall_pq_native": recall(row["pq_native_result_ids"], truth),
            "recall_exact_oracle": recall(row["exact_oracle_result_ids"], truth),
            "recall_pq_oracle": recall(row["pq_oracle_result_ids"], truth),
            "coverage_exact": len(set(exact_set) & set(truth)) / 10,
            "coverage_pq": len(set(pq_set) & set(truth)) / 10,
        }
        for field, value in values.items():
            if not math.isclose(value, row[field], abs_tol=1e-12):
                raise ValueError(f"{field} mismatch")
        if row["recall_exact_oracle"] != row["coverage_exact"] or row["recall_pq_oracle"] != row["coverage_pq"]:
            raise ValueError("oracle recall differs from candidate coverage")
        deltas = {
            "delta_total": row["recall_exact_native"] - row["recall_pq_native"],
            "delta_discovery": row["recall_exact_oracle"] - row["recall_pq_oracle"],
            "delta_ranking": row["recall_pq_oracle"] - row["recall_pq_native"],
            "delta_exact_control": row["recall_exact_oracle"] - row["recall_exact_native"],
        }
        for field, value in deltas.items():
            if not math.isclose(value, row[field], abs_tol=1e-12):
                raise ValueError(f"{field} mismatch")
        intersection = len(set(exact_set) & set(pq_set))
        union = len(set(exact_set) | set(pq_set))
        if intersection != row["evaluated_intersection_size"] or not math.isclose(intersection / union, row["evaluated_jaccard"], abs_tol=1e-12):
            raise ValueError("evaluated-set overlap mismatch")


def summarize_replicate(replicate_id, rows, manifest, criterion):
    mean = lambda field: statistics.fmean(row[field] for row in rows)
    exact = mean("recall_exact_native")
    pq_native = mean("recall_pq_native")
    pq_oracle = mean("recall_pq_oracle")
    denominator = exact - pq_native
    recovery = (pq_oracle - pq_native) / denominator if abs(denominator) > 1e-12 else None
    summary = {
        "replicate_id": replicate_id,
        "database_seed": manifest["dataset"]["database_seed"],
        "query_seed": manifest["dataset"]["query_seed"],
        "graph_seed": manifest["graph"]["seed"],
        "pq_seed": manifest["pq"]["seed"],
        "graph_fingerprint": manifest["graph_fingerprint"],
        "mean_recall_exact_native": exact,
        "mean_recall_pq_native": pq_native,
        "mean_recall_pq_oracle": pq_oracle,
        "mean_delta_total": mean("delta_total"),
        "mean_delta_discovery": mean("delta_discovery"),
        "mean_delta_ranking": mean("delta_ranking"),
        "rerank_recovery": recovery,
        "mean_candidate_coverage_exact": mean("coverage_exact"),
        "mean_candidate_coverage_pq": mean("coverage_pq"),
        "mean_evaluated_set_jaccard": mean("evaluated_jaccard"),
        "max_absolute_delta_exact_control": max(abs(row["delta_exact_control"]) for row in rows),
    }
    summary["criterion_abs_discovery_pass"] = abs(summary["mean_delta_discovery"]) <= criterion["max_abs_mean_delta_discovery"]
    summary["criterion_recovery_pass"] = recovery is not None and recovery >= criterion["min_rerank_recovery"]
    summary["criterion_replicate_pass"] = summary["criterion_abs_discovery_pass"] and summary["criterion_recovery_pass"]
    return summary


def aggregate(replicates):
    fields = [
        "mean_recall_exact_native", "mean_recall_pq_native",
        "mean_recall_pq_oracle", "mean_delta_total", "mean_delta_discovery",
        "mean_delta_ranking", "rerank_recovery",
        "mean_candidate_coverage_exact", "mean_candidate_coverage_pq",
        "mean_evaluated_set_jaccard",
    ]
    output = {}
    for field in fields:
        values = [replicate[field] for replicate in replicates]
        output[field] = {
            "mean_across_replicates": statistics.fmean(values),
            "sample_standard_deviation": statistics.stdev(values),
            "minimum": min(values),
            "maximum": max(values),
        }
    return output


def five_numbers(values):
    quartiles = statistics.quantiles(values, n=4, method="inclusive")
    return min(values), quartiles[0], statistics.median(values), quartiles[2], max(values)


def boxplot_svg(path, title, labels_and_values, y_label):
    width, height, left, right, top, bottom = 900, 500, 78, 24, 42, 78
    all_values = [value for _, values, _ in labels_and_values for value in values]
    y_min, y_max = min(all_values), max(all_values)
    padding = max((y_max - y_min) * 0.08, 0.005)
    y_min -= padding; y_max += padding
    plot_w, plot_h = width - left - right, height - top - bottom
    sy = lambda value: top + (y_max - value) * plot_h / (y_max - y_min)
    spacing = plot_w / len(labels_and_values)
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="500" viewBox="0 0 900 500">', '<rect width="100%" height="100%" fill="white"/>', f'<text x="450" y="24" text-anchor="middle" font-family="sans-serif" font-size="17">{html.escape(title)}</text>', f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="black"/>', f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="black"/>', f'<text x="18" y="{top + plot_h / 2}" text-anchor="middle" transform="rotate(-90 18 {top + plot_h / 2})" font-family="sans-serif" font-size="13">{html.escape(y_label)}</text>']
    for tick in range(6):
        value = y_min + (y_max - y_min) * tick / 5
        parts.append(f'<text x="{left - 8}" y="{sy(value) + 4:.2f}" text-anchor="end" font-family="sans-serif" font-size="11">{value:.3g}</text>')
    for index, (label, values, color) in enumerate(labels_and_values):
        low, q1, median, q3, high = five_numbers(values)
        x = left + spacing * (index + 0.5)
        box_width = min(32, spacing * 0.55)
        parts.extend([
            f'<line x1="{x:.2f}" y1="{sy(low):.2f}" x2="{x:.2f}" y2="{sy(high):.2f}" stroke="{color}"/>',
            f'<line x1="{x - box_width / 3:.2f}" y1="{sy(low):.2f}" x2="{x + box_width / 3:.2f}" y2="{sy(low):.2f}" stroke="{color}"/>',
            f'<line x1="{x - box_width / 3:.2f}" y1="{sy(high):.2f}" x2="{x + box_width / 3:.2f}" y2="{sy(high):.2f}" stroke="{color}"/>',
            f'<rect x="{x - box_width / 2:.2f}" y="{sy(q3):.2f}" width="{box_width:.2f}" height="{sy(q1) - sy(q3):.2f}" fill="{color}" fill-opacity="0.3" stroke="{color}"/>',
            f'<line x1="{x - box_width / 2:.2f}" y1="{sy(median):.2f}" x2="{x + box_width / 2:.2f}" y2="{sy(median):.2f}" stroke="{color}" stroke-width="2"/>',
            f'<text x="{x:.2f}" y="{top + plot_h + 22}" text-anchor="middle" font-family="sans-serif" font-size="10" transform="rotate(30 {x:.2f} {top + plot_h + 22})">{html.escape(label)}</text>',
        ])
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("table_directory", type=Path)
    parser.add_argument("figure_directory", type=Path)
    args = parser.parse_args()
    root_manifest = json.loads((args.run_directory / "manifest.json").read_text(encoding="utf-8"))
    criterion = root_manifest["pre_registered_criterion"]
    query_count = root_manifest["fixed_parameters"]["queries"]
    replicates = []
    per_query = []
    fingerprints = set()
    for replicate_id in range(root_manifest["replicate_count"]):
        directory = args.run_directory / f"replicate_{replicate_id:02d}"
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        rows = read_jsonl(directory / "queries.jsonl")
        validate_replicate(rows, manifest, query_count)
        if manifest["graph_fingerprint"] in fingerprints:
            raise ValueError("replicates unexpectedly share a graph fingerprint")
        fingerprints.add(manifest["graph_fingerprint"])
        replicates.append(summarize_replicate(replicate_id, rows, manifest, criterion))
        per_query.append(rows)
    passing = sum(replicate["criterion_replicate_pass"] for replicate in replicates)
    criterion_pass = passing >= criterion["required_passing_replicates"]
    payload = {
        "schema_version": 1,
        "source_run": str(args.run_directory),
        "pre_registered_criterion": criterion,
        "passing_replicates": passing,
        "robustness_criterion_pass": criterion_pass,
        "replicates": replicates,
        "aggregate": aggregate(replicates),
    }

    args.table_directory.mkdir(parents=True, exist_ok=True)
    args.figure_directory.mkdir(parents=True, exist_ok=True)
    json_path = args.table_directory / "phase1_seed_robustness.json"
    csv_path = args.table_directory / "phase1_seed_robustness.csv"
    delta_figure = args.figure_directory / "phase1_seed_robustness_delta_boxplots.svg"
    recovery_figure = args.figure_directory / "phase1_seed_robustness_recovery_boxplot.svg"
    for path in (json_path, csv_path, delta_figure, recovery_figure):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(replicates[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(replicates)
    delta_boxes = []
    for replicate_id, rows in enumerate(per_query):
        delta_boxes.append((f"R{replicate_id} D", [row["delta_discovery"] for row in rows], "#4c78a8"))
        delta_boxes.append((f"R{replicate_id} R", [row["delta_ranking"] for row in rows], "#f58518"))
    boxplot_svg(delta_figure, "Per-query discovery (D) and ranking (R) deltas", delta_boxes, "delta recall@10")
    boxplot_svg(recovery_figure, "Rerank recovery across seed replicates", [("recovery", [replicate["rerank_recovery"] for replicate in replicates], "#54a24b")], "rerank recovery")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
