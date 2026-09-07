#!/usr/bin/env python3
"""Validate and summarize the fixed-topology Phase 1 calibration sweep."""

import argparse
import csv
import html
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


COLORS = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#b279a2"]


def percentile(values, probability):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


def concentration(positive, target):
    ordered = sorted(positive, reverse=True)
    threshold = target * sum(ordered)
    running = 0.0
    for count, value in enumerate(ordered, 1):
        running += value
        if running + 1e-12 >= threshold:
            return count, count / len(ordered)
    raise RuntimeError("positive-loss threshold was not reached")


def recall(result_ids, truth_ids):
    if len(result_ids) != 10 or len(set(result_ids)) != 10:
        raise ValueError("result IDs are not a unique top-10")
    if len(truth_ids) != 10 or len(set(truth_ids)) != 10:
        raise ValueError("ground-truth IDs are not a unique top-10")
    return len(set(result_ids) & set(truth_ids)) / 10


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def validate(manifest, exact_rows, paired_rows, distance_rows, order_rows):
    query_count = manifest["dataset"]["queries"]
    ef_values = manifest["search"]["ef_search_values"]
    pq_ids = [item["id"] for item in manifest["pq_configs"]]
    exact = {}
    exact_groups = defaultdict(list)
    for row in exact_rows:
        key = (row["ef_search"], row["query_id"])
        if key in exact or row["query_order"] != row["query_id"]:
            raise ValueError(f"duplicate or reordered exact query: {key}")
        measured = recall(row["exact_result_ids"], row["ground_truth_ids"])
        if not math.isclose(measured, row["recall_exact_at_10"], abs_tol=1e-12):
            raise ValueError(f"exact recall mismatch: {key}")
        exact[key] = row
        exact_groups[row["ef_search"]].append(row)
    for ef in ef_values:
        if sorted(row["query_id"] for row in exact_groups[ef]) != list(range(query_count)):
            raise ValueError(f"incomplete exact query order for efSearch={ef}")

    paired_groups = defaultdict(list)
    for row in paired_rows:
        key = (row["ef_search"], row["query_id"])
        reference = exact.get(key)
        if reference is None or row["query_order"] != row["query_id"]:
            raise ValueError(f"unpaired query: {key}")
        if row["exact_result_ids"] != reference["exact_result_ids"] or row["ground_truth_ids"] != reference["ground_truth_ids"]:
            raise ValueError(f"exact result or ground truth changed: {key}")
        exact_recall = recall(row["exact_result_ids"], row["ground_truth_ids"])
        pq_recall = recall(row["pq_result_ids"], row["ground_truth_ids"])
        delta = exact_recall - pq_recall
        if not math.isclose(exact_recall, row["recall_exact_at_10"], abs_tol=1e-12) or not math.isclose(pq_recall, row["recall_pq_at_10"], abs_tol=1e-12) or not math.isclose(delta, row["delta_recall_at_10"], abs_tol=1e-12):
            raise ValueError(f"paired metric mismatch: {key}, {row['pq_id']}")
        paired_groups[(row["ef_search"], row["pq_id"])].append(row)
    for ef in ef_values:
        for pq_id in pq_ids:
            group = paired_groups[(ef, pq_id)]
            if [row["query_id"] for row in group] != list(range(query_count)):
                raise ValueError(f"incomplete paired query order: {ef}, {pq_id}")

    distance_groups = defaultdict(list)
    for row in distance_rows:
        expected = abs(row["pq_distance"] - row["exact_distance"])
        if not math.isclose(expected, row["absolute_error"], rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("distance absolute-error mismatch")
        if row["absolute_relative_error"] is not None:
            relative = expected / row["exact_distance"]
            if not math.isclose(relative, row["absolute_relative_error"], rel_tol=1e-6, abs_tol=1e-8):
                raise ValueError("distance relative-error mismatch")
        distance_groups[row["pq_id"]].append(row)

    order_groups = defaultdict(list)
    for row in order_rows:
        inversion = ((row["exact_first"] < row["exact_second"] and row["pq_first"] > row["pq_second"]) or
                     (row["exact_first"] > row["exact_second"] and row["pq_first"] < row["pq_second"]))
        if inversion != row["strict_inversion"] or (row["pq_first"] == row["pq_second"]) != row["pq_tie"]:
            raise ValueError("pairwise-order label mismatch")
        order_groups[row["pq_id"]].append(row)
    expected_distance = manifest["distance_sampling"]["query_selection"].split()[1]
    expected_distance = int(expected_distance) * manifest["distance_sampling"]["distance_pairs_per_query"]
    expected_order = int(manifest["distance_sampling"]["query_selection"].split()[1]) * manifest["distance_sampling"]["order_pairs_per_query"]
    for pq_id in pq_ids:
        if len(distance_groups[pq_id]) != expected_distance or len(order_groups[pq_id]) != expected_order:
            raise ValueError(f"incomplete distance samples for {pq_id}")
    return exact_groups, paired_groups, distance_groups, order_groups


def summarize(exact_groups, paired_groups, distance_groups, order_groups, manifest):
    exact_summary = []
    for ef, rows in sorted(exact_groups.items()):
        recalls = [row["recall_exact_at_10"] for row in rows]
        exact_summary.append({"ef_search": ef, "mean_recall_exact_at_10": statistics.fmean(recalls), "median_recall_exact_at_10": statistics.median(recalls)})

    paired_summary = []
    for (ef, pq_id), rows in sorted(paired_groups.items()):
        exact = [row["recall_exact_at_10"] for row in rows]
        pq = [row["recall_pq_at_10"] for row in rows]
        delta = [row["delta_recall_at_10"] for row in rows]
        positive = [value for value in delta if value > 0]
        item = {
            "ef_search": ef,
            "pq_id": pq_id,
            "mean_recall_exact_at_10": statistics.fmean(exact),
            "mean_recall_pq_at_10": statistics.fmean(pq),
            "mean_delta_recall_at_10": statistics.fmean(delta),
            "median_delta_recall_at_10": statistics.median(delta),
            "p90_delta_recall_at_10": percentile(delta, 0.90),
            "p95_delta_recall_at_10": percentile(delta, 0.95),
            "p99_delta_recall_at_10": percentile(delta, 0.99),
            "fraction_delta_equal_zero": sum(value == 0 for value in delta) / len(delta),
            "fraction_delta_greater_zero": len(positive) / len(delta),
            "fraction_delta_less_zero": sum(value < 0 for value in delta) / len(delta),
            "positive_loss_query_count": len(positive),
            "total_positive_loss": sum(positive),
        }
        for label, target in (("50", 0.5), ("80", 0.8), ("90", 0.9)):
            if positive:
                count, fraction = concentration(positive, target)
                item[f"positive_query_count_for_{label}_percent_loss"] = count
                item[f"positive_query_fraction_for_{label}_percent_loss"] = fraction
            else:
                item[f"positive_query_count_for_{label}_percent_loss"] = 0
                item[f"positive_query_fraction_for_{label}_percent_loss"] = None
        paired_summary.append(item)

    pq_metadata = {item["id"]: item for item in manifest["pq_configs"]}
    distance_summary = []
    for pq_id, rows in sorted(distance_groups.items()):
        absolute = [row["absolute_error"] for row in rows]
        relative = [row["absolute_relative_error"] for row in rows if row["absolute_relative_error"] is not None]
        order = order_groups[pq_id]
        distance_summary.append({
            "pq_id": pq_id,
            "code_bytes": pq_metadata[pq_id]["code_bytes"],
            "distance_pair_count": len(rows),
            "mean_absolute_distance_error": statistics.fmean(absolute),
            "median_absolute_distance_error": statistics.median(absolute),
            "p95_absolute_distance_error": percentile(absolute, 0.95),
            "relative_distance_pair_count": len(relative),
            "mean_absolute_relative_distance_error": statistics.fmean(relative),
            "median_absolute_relative_distance_error": statistics.median(relative),
            "p95_absolute_relative_distance_error": percentile(relative, 0.95),
            "order_pair_count": len(order),
            "strict_order_inversion_rate": sum(row["strict_inversion"] for row in order) / len(order),
            "pq_distance_tie_rate": sum(row["pq_tie"] for row in order) / len(order),
        })
    return exact_summary, paired_summary, distance_summary


def canvas(title, x_label, y_label, x_min, x_max, y_min, y_max):
    width, height, left, right, top, bottom = 820, 500, 76, 24, 42, 64
    plot_w, plot_h = width - left - right, height - top - bottom
    def sx(x): return left + (x - x_min) * plot_w / (x_max - x_min)
    def sy(y): return top + (y_max - y) * plot_h / (y_max - y_min)
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="820" height="500" viewBox="0 0 820 500">', '<rect width="100%" height="100%" fill="white"/>', f'<text x="410" y="24" text-anchor="middle" font-family="sans-serif" font-size="17">{html.escape(title)}</text>', f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="black"/>', f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="black"/>', f'<text x="{left + plot_w / 2}" y="486" text-anchor="middle" font-family="sans-serif" font-size="13">{html.escape(x_label)}</text>', f'<text x="18" y="{top + plot_h / 2}" text-anchor="middle" transform="rotate(-90 18 {top + plot_h / 2})" font-family="sans-serif" font-size="13">{html.escape(y_label)}</text>']
    for i in range(6):
        x = x_min + (x_max - x_min) * i / 5
        y = y_min + (y_max - y_min) * i / 5
        parts.extend([f'<text x="{sx(x):.2f}" y="{top + plot_h + 20}" text-anchor="middle" font-family="sans-serif" font-size="11">{x:.3g}</text>', f'<text x="{left - 8}" y="{sy(y) + 4:.2f}" text-anchor="end" font-family="sans-serif" font-size="11">{y:.2f}</text>'])
    return parts, sx, sy


def polyline(parts, sx, sy, points, color, label, legend_index, dashed=False):
    encoded = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in points)
    dash = ' stroke-dasharray="6 4"' if dashed else ""
    parts.append(f'<polyline points="{encoded}" fill="none" stroke="{color}" stroke-width="2"{dash}/>')
    for x, y in points:
        parts.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="3" fill="{color}"/>')
    y = 52 + legend_index * 18
    parts.extend([f'<line x1="580" y1="{y}" x2="605" y2="{y}" stroke="{color}" stroke-width="2"{dash}/>', f'<text x="611" y="{y + 4}" font-family="sans-serif" font-size="11">{html.escape(label)}</text>'])


def write_figures(directory, exact_summary, paired_summary, distance_summary, paired_groups, selected):
    directory.mkdir(parents=True, exist_ok=True)
    paths = [directory / "phase1_calibration_recall_vs_efsearch.svg", directory / "phase1_calibration_recall_vs_compression.svg", directory / "phase1_calibration_selected_delta_cdfs.svg"]
    for path in paths:
        if path.exists(): raise FileExistsError(f"refusing to overwrite {path}")

    ef_values = [row["ef_search"] for row in exact_summary]
    parts, sx, sy = canvas("Recall@10 versus efSearch", "efSearch", "mean Recall@10", min(ef_values), max(ef_values), 0, 1)
    polyline(parts, sx, sy, [(row["ef_search"], row["mean_recall_exact_at_10"]) for row in exact_summary], COLORS[0], "exact FP32", 0)
    pq_ids = sorted({row["pq_id"] for row in paired_summary})
    for i, pq_id in enumerate(pq_ids, 1):
        rows = sorted((row for row in paired_summary if row["pq_id"] == pq_id), key=lambda row: row["ef_search"])
        polyline(parts, sx, sy, [(row["ef_search"], row["mean_recall_pq_at_10"]) for row in rows], COLORS[i % len(COLORS)], pq_id, i)
    parts.append("</svg>"); paths[0].write_text("\n".join(parts) + "\n", encoding="utf-8")

    code_bytes = {row["pq_id"]: row["code_bytes"] for row in distance_summary}
    parts, sx, sy = canvas("Recall@10 versus PQ code size", "PQ code bytes per vector", "mean Recall@10", min(code_bytes.values()), max(code_bytes.values()), 0, 1)
    for i, (ef, _) in enumerate(selected):
        rows = sorted((row for row in paired_summary if row["ef_search"] == ef), key=lambda row: code_bytes[row["pq_id"]])
        polyline(parts, sx, sy, [(code_bytes[row["pq_id"]], row["mean_recall_pq_at_10"]) for row in rows], COLORS[i], f"PQ, ef={ef}", i)
        exact_value = next(row["mean_recall_exact_at_10"] for row in exact_summary if row["ef_search"] == ef)
        polyline(parts, sx, sy, [(min(code_bytes.values()), exact_value), (max(code_bytes.values()), exact_value)], COLORS[i], f"exact, ef={ef}", i + len(selected), dashed=True)
    parts.append("</svg>"); paths[1].write_text("\n".join(parts) + "\n", encoding="utf-8")

    selected_values = [[row["delta_recall_at_10"] for row in paired_groups[key]] for key in selected]
    minimum = min(min(values) for values in selected_values) - 0.02
    maximum = max(max(values) for values in selected_values) + 0.02
    parts, sx, sy = canvas("Selected operating-point delta CDFs", "delta_recall@10", "fraction of queries at or below x", minimum, maximum, 0, 1)
    for i, (key, values) in enumerate(zip(selected, selected_values)):
        ordered = sorted(values)
        polyline(parts, sx, sy, [(value, (rank + 1) / len(ordered)) for rank, value in enumerate(ordered)], COLORS[i], f"ef={key[0]}, {key[1]}", i)
    parts.append("</svg>"); paths[2].write_text("\n".join(parts) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("table_directory", type=Path)
    parser.add_argument("figure_directory", type=Path)
    parser.add_argument("--selected", required=True, help="comma-separated ef:pq_id operating points")
    args = parser.parse_args()
    selected = [(int(item.split(":", 1)[0]), item.split(":", 1)[1]) for item in args.selected.split(",")]
    if not 2 <= len(selected) <= 3: raise ValueError("select two or three operating points")

    manifest = json.loads((args.run_directory / "manifest.json").read_text(encoding="utf-8"))
    groups = validate(manifest, read_jsonl(args.run_directory / "exact_queries.jsonl"), read_jsonl(args.run_directory / "paired_queries.jsonl"), read_jsonl(args.run_directory / "distance_samples.jsonl"), read_jsonl(args.run_directory / "order_samples.jsonl"))
    exact_summary, paired_summary, distance_summary = summarize(*groups, manifest)
    for key in selected:
        if key not in groups[1]: raise ValueError(f"selected point does not exist: {key}")

    args.table_directory.mkdir(parents=True, exist_ok=True)
    json_path = args.table_directory / "phase1_calibration.json"
    csv_path = args.table_directory / "phase1_calibration.csv"
    distance_path = args.table_directory / "phase1_calibration_distance_quality.csv"
    selected_path = args.table_directory / "phase1_calibration_selected_points.json"
    for path in (json_path, csv_path, distance_path, selected_path):
        if path.exists(): raise FileExistsError(f"refusing to overwrite {path}")
    payload = {"schema_version": 1, "source_run": str(args.run_directory), "graph_fingerprint": manifest["graph_fingerprint"], "exact": exact_summary, "paired": paired_summary, "distance_quality": distance_summary}
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_summary[0]), lineterminator="\n"); writer.writeheader(); writer.writerows(paired_summary)
    with distance_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(distance_summary[0]), lineterminator="\n"); writer.writeheader(); writer.writerows(distance_summary)
    selected_rows = [next(row for row in paired_summary if (row["ef_search"], row["pq_id"]) == key) for key in selected]
    selected_path.write_text(json.dumps(selected_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_figures(args.figure_directory, exact_summary, paired_summary, distance_summary, groups[1], selected)
    print(json.dumps({"exact": exact_summary, "distance_quality": distance_summary, "selected": selected_rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
