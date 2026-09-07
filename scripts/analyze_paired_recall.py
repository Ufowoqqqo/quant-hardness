#!/usr/bin/env python3
"""Validate and summarize an immutable paired-recall JSONL run."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def percentile(values, probability):
    """Empirical nearest-rank percentile (smallest value with ECDF >= p)."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


def concentration(positive, target):
    ordered = sorted(positive, reverse=True)
    threshold = target * sum(ordered)
    running = 0.0
    for count, value in enumerate(ordered, 1):
        running += value
        if running + 1e-12 >= threshold:
            return count / len(ordered), count
    raise RuntimeError("positive-loss concentration threshold was not reached")


def validate_rows(path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            expected = len(rows)
            if row["query_id"] != expected or row["query_order"] != expected:
                raise ValueError(f"non-identical query ordering at line {line_number}")
            for field in ("exact_result_ids", "pq_result_ids", "ground_truth_ids"):
                if len(row[field]) != 10 or len(set(row[field])) != 10:
                    raise ValueError(f"invalid top-10 IDs in {field} at line {line_number}")
            ground_truth = set(row["ground_truth_ids"])
            exact_recall = len(set(row["exact_result_ids"]) & ground_truth) / 10
            pq_recall = len(set(row["pq_result_ids"]) & ground_truth) / 10
            if not math.isclose(exact_recall, row["recall_exact_at_10"], abs_tol=1e-12):
                raise ValueError(f"exact recall mismatch at line {line_number}")
            if not math.isclose(pq_recall, row["recall_pq_at_10"], abs_tol=1e-12):
                raise ValueError(f"PQ recall mismatch at line {line_number}")
            delta = row["recall_exact_at_10"] - row["recall_pq_at_10"]
            if not math.isclose(delta, row["delta_recall_at_10"], abs_tol=1e-12):
                raise ValueError(f"delta sign/value mismatch at line {line_number}")
            rows.append(row)
    if not rows:
        raise ValueError("input contains no rows")
    return rows


def svg_start(title, x_label, y_label, x_min, x_max, y_min, y_max):
    width, height = 760, 480
    left, right, top, bottom = 78, 24, 42, 66
    plot_w, plot_h = width - left - right, height - top - bottom
    if x_max == x_min:
        x_min, x_max = x_min - 0.5, x_max + 0.5
    if y_max == y_min:
        y_min, y_max = y_min - 0.5, y_max + 0.5

    def sx(x):
        return left + (x - x_min) * plot_w / (x_max - x_min)

    def sy(y):
        return top + (y_max - y) * plot_h / (y_max - y_min)

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="760" height="480" viewBox="0 0 760 480">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="380" y="24" text-anchor="middle" font-family="sans-serif" font-size="17">{title}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="black"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="black"/>',
        f'<text x="{left + plot_w / 2}" y="464" text-anchor="middle" font-family="sans-serif" font-size="13">{x_label}</text>',
        f'<text x="18" y="{top + plot_h / 2}" text-anchor="middle" transform="rotate(-90 18 {top + plot_h / 2})" font-family="sans-serif" font-size="13">{y_label}</text>',
    ]
    for i in range(6):
        x = x_min + (x_max - x_min) * i / 5
        px = sx(x)
        parts.extend([
            f'<line x1="{px:.2f}" y1="{top + plot_h}" x2="{px:.2f}" y2="{top + plot_h + 5}" stroke="black"/>',
            f'<text x="{px:.2f}" y="{top + plot_h + 21}" text-anchor="middle" font-family="sans-serif" font-size="11">{x:.2g}</text>',
        ])
        y = y_min + (y_max - y_min) * i / 5
        py = sy(y)
        parts.extend([
            f'<line x1="{left - 5}" y1="{py:.2f}" x2="{left}" y2="{py:.2f}" stroke="black"/>',
            f'<text x="{left - 9}" y="{py + 4:.2f}" text-anchor="end" font-family="sans-serif" font-size="11">{y:.2g}</text>',
        ])
    return parts, sx, sy


def write_svg(path, parts):
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def make_figures(rows, directory, prefix):
    deltas = [row["delta_recall_at_10"] for row in rows]
    exact = [row["recall_exact_at_10"] for row in rows]

    # Discrete Recall@10 differences have step 0.1; use centered 0.1 bins.
    low = math.floor(min(deltas) * 10) / 10
    high = math.ceil(max(deltas) * 10) / 10
    centers = [round(low + 0.1 * i, 10) for i in range(round((high - low) * 10) + 1)]
    counts = [sum(math.isclose(value, center, abs_tol=1e-9) for value in deltas) for center in centers]
    parts, sx, sy = svg_start("Histogram of delta recall", "delta_recall@10", "query count", low - 0.06, high + 0.06, 0, max(counts) * 1.08)
    bar_width = abs(sx(low + 0.08) - sx(low))
    for center, count in zip(centers, counts):
        parts.append(f'<rect x="{sx(center) - bar_width / 2:.2f}" y="{sy(count):.2f}" width="{bar_width:.2f}" height="{sy(0) - sy(count):.2f}" fill="#4c78a8"/>')
    write_svg(directory / f"{prefix}_histogram.svg", parts)

    ordered = sorted(deltas)
    parts, sx, sy = svg_start("Empirical CDF of delta recall", "delta_recall@10", "fraction of queries at or below x", min(ordered) - 0.02, max(ordered) + 0.02, 0, 1)
    points = " ".join(f"{sx(value):.2f},{sy((i + 1) / len(ordered)):.2f}" for i, value in enumerate(ordered))
    parts.append(f'<polyline points="{points}" fill="none" stroke="#4c78a8" stroke-width="2"/>')
    write_svg(directory / f"{prefix}_ecdf.svg", parts)

    parts, sx, sy = svg_start("Delta recall sorted by query", "sorted query rank", "delta_recall@10", 1, len(ordered), min(ordered) - 0.02, max(ordered) + 0.02)
    points = " ".join(f"{sx(i + 1):.2f},{sy(value):.2f}" for i, value in enumerate(ordered))
    parts.append(f'<polyline points="{points}" fill="none" stroke="#f58518" stroke-width="2"/>')
    write_svg(directory / f"{prefix}_sorted.svg", parts)

    parts, sx, sy = svg_start("Exact recall versus delta recall", "recall_exact@10", "delta_recall@10", min(exact) - 0.02, max(exact) + 0.02, min(deltas) - 0.02, max(deltas) + 0.02)
    for x, y in zip(exact, deltas):
        parts.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="3" fill="#54a24b" fill-opacity="0.45"/>')
    write_svg(directory / f"{prefix}_scatter.svg", parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_jsonl", type=Path)
    parser.add_argument("table_directory", type=Path)
    parser.add_argument("figure_directory", type=Path)
    parser.add_argument("--prefix", default="phase1_synthetic_paired_recall_v1")
    args = parser.parse_args()

    rows = validate_rows(args.input_jsonl)
    deltas = [row["delta_recall_at_10"] for row in rows]
    positive = [value for value in deltas if value > 0]
    n = len(deltas)
    summary = {
        "schema_version": 1,
        "query_count": n,
        "delta_definition": "recall_exact_at_10 - recall_pq_at_10",
        "percentile_definition": "empirical nearest-rank",
        "delta_recall_at_10": {
            "mean": statistics.fmean(deltas),
            "median": statistics.median(deltas),
            "min": min(deltas),
            "max": max(deltas),
            "p90": percentile(deltas, 0.90),
            "p95": percentile(deltas, 0.95),
            "p99": percentile(deltas, 0.99),
            "fraction_equal_zero": sum(value == 0 for value in deltas) / n,
            "fraction_greater_than_zero": len(positive) / n,
            "fraction_less_than_zero": sum(value < 0 for value in deltas) / n,
        },
        "positive_recall_loss": {
            "query_count": len(positive),
            "total": sum(positive),
        },
    }
    if positive:
        for label, target in (("50_percent", 0.5), ("80_percent", 0.8), ("90_percent", 0.9)):
            fraction, count = concentration(positive, target)
            summary["positive_recall_loss"][f"query_fraction_for_{label}"] = fraction
            summary["positive_recall_loss"][f"query_count_for_{label}"] = count

    args.table_directory.mkdir(parents=True, exist_ok=True)
    args.figure_directory.mkdir(parents=True, exist_ok=True)
    summary_path = args.table_directory / f"{args.prefix}_summary.json"
    csv_path = args.table_directory / f"{args.prefix}_delta_sorted.csv"
    figure_paths = [args.figure_directory / f"{args.prefix}_{suffix}.svg" for suffix in ("histogram", "ecdf", "sorted", "scatter")]
    for path in [summary_path, csv_path, *figure_paths]:
        if path.exists():
            raise FileExistsError(f"refusing to overwrite derived output: {path}")

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["sorted_rank", "query_id", "delta_recall_at_10"])
        for rank, row in enumerate(sorted(rows, key=lambda item: (item["delta_recall_at_10"], item["query_id"])), 1):
            writer.writerow([rank, row["query_id"], row["delta_recall_at_10"]])
    make_figures(rows, args.figure_directory, args.prefix)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
