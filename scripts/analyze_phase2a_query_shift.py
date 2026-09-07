#!/usr/bin/env python3
"""Validate, aggregate, and plot the Phase 2A query-shift experiment."""

import argparse
import csv
import html
import json
import math
import statistics
from pathlib import Path


DESCRIPTORS = [
    "query_l2_norm",
    "distance_to_database_centroid",
    "exact_nn_distance_squared",
    "exact_10th_distance_squared",
    "nn_margin_squared",
    "normalized_nn_margin_by_10th",
    "nearest_to_10th_distance_ratio",
]
DELTA_FIELDS = ("delta_total", "delta_discovery", "delta_ranking")


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def recall(ids, truth):
    if len(ids) != 10 or len(set(ids)) != 10:
        raise ValueError("native/oracle result is not a unique top-10")
    return len(set(ids) & set(truth)) / 10


def percentile(values, probability):
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def rank(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    output = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        average_rank = (start + stop - 1) / 2
        for index in order[start:stop]:
            output[index] = average_rank
        start = stop
    return output


def spearman(xs, ys):
    rx, ry = rank(xs), rank(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    numerator = sum((x - mx) * (y - my) for x, y in zip(rx, ry))
    denominator = math.sqrt(sum((x - mx) ** 2 for x in rx) * sum((y - my) ** 2 for y in ry))
    return numerator / denominator if denominator else None


def validate_rows(rows, query_count, phase, ef_search):
    if len(rows) != query_count:
        raise ValueError(f"expected {query_count} rows, found {len(rows)}")
    if [row["query_id"] for row in rows] != list(range(query_count)):
        raise ValueError("query order is incomplete or changed")
    for row in rows:
        if row["phase"] != phase or row["ef_search"] != ef_search:
            raise ValueError("phase or efSearch mismatch")
        truth = row["ground_truth_ids"]
        for field in ("exact_native_result_ids", "pq_native_result_ids", "exact_oracle_result_ids", "pq_oracle_result_ids"):
            expected = recall(row[field], truth)
            metric = {
                "exact_native_result_ids": "recall_exact_native",
                "pq_native_result_ids": "recall_pq_native",
                "exact_oracle_result_ids": "recall_exact_oracle",
                "pq_oracle_result_ids": "recall_pq_oracle",
            }[field]
            if not math.isclose(expected, row[metric], abs_tol=1e-12):
                raise ValueError(f"{metric} mismatch")
        expected_deltas = {
            "delta_total": row["recall_exact_native"] - row["recall_pq_native"],
            "delta_discovery": row["recall_exact_oracle"] - row["recall_pq_oracle"],
            "delta_ranking": row["recall_pq_oracle"] - row["recall_pq_native"],
            "delta_exact_control": row["recall_exact_oracle"] - row["recall_exact_native"],
        }
        for field, value in expected_deltas.items():
            if not math.isclose(value, row[field], abs_tol=1e-12):
                raise ValueError(f"{field} mismatch")
        if row["recall_exact_oracle"] != row["coverage_exact"] or row["recall_pq_oracle"] != row["coverage_pq"]:
            raise ValueError("candidate coverage/oracle equality failed")
        exact_ids, pq_ids = row["exact_evaluated_ids"], row["pq_evaluated_ids"]
        should_store = row["query_id"] < 25
        if should_store != (exact_ids is not None and pq_ids is not None):
            raise ValueError("candidate-set sampling policy mismatch")
        if should_store:
            if exact_ids != sorted(set(exact_ids)) or pq_ids != sorted(set(pq_ids)):
                raise ValueError("sampled candidate IDs are not sorted unique")
            if len(exact_ids) != row["exact_unique_distance_evaluations"] or len(pq_ids) != row["pq_unique_distance_evaluations"]:
                raise ValueError("candidate count mismatch")
            intersection = len(set(exact_ids) & set(pq_ids))
            union = len(set(exact_ids) | set(pq_ids))
            if intersection != row["evaluated_intersection_size"] or not math.isclose(intersection / union, row["evaluated_jaccard"], abs_tol=1e-12):
                raise ValueError("candidate overlap mismatch")
            if len(set(exact_ids) & set(truth)) / 10 != row["coverage_exact"] or len(set(pq_ids) & set(truth)) / 10 != row["coverage_pq"]:
                raise ValueError("sampled candidate coverage mismatch")


def summarize(rows, replicate_id, phase):
    mean = lambda field: statistics.fmean(row[field] for row in rows)
    exact = mean("recall_exact_native")
    pq_native = mean("recall_pq_native")
    pq_oracle = mean("recall_pq_oracle")
    denominator = exact - pq_native
    output = {
        "phase": phase,
        "replicate_id": replicate_id,
        "condition_id": rows[0]["condition_id"],
        "shift_family": rows[0]["shift_family"],
        "shift_severity": rows[0]["shift_severity"],
        "ef_search": rows[0]["ef_search"],
        "mean_recall_exact_native": exact,
        "mean_recall_pq_native": pq_native,
        "mean_recall_pq_oracle": pq_oracle,
        "rerank_recovery": ((pq_oracle - pq_native) / denominator if abs(denominator) > 1e-12 else None),
        "mean_candidate_coverage_exact": mean("coverage_exact"),
        "mean_candidate_coverage_pq": mean("coverage_pq"),
        "mean_evaluated_set_jaccard": mean("evaluated_jaccard"),
        "mean_exact_unique_distance_evaluations": mean("exact_unique_distance_evaluations"),
        "mean_pq_unique_distance_evaluations": mean("pq_unique_distance_evaluations"),
        "max_absolute_delta_exact_control": max(abs(row["delta_exact_control"]) for row in rows),
    }
    for field in DELTA_FIELDS:
        values = [row[field] for row in rows]
        output[f"mean_{field}"] = statistics.fmean(values)
        output[f"median_{field}"] = statistics.median(values)
        for label, probability in (("p90", .90), ("p95", .95), ("p99", .99)):
            output[f"{label}_{field}"] = percentile(values, probability)
    discovery = [row["delta_discovery"] for row in rows]
    output["fraction_delta_discovery_positive"] = sum(value > 1e-12 for value in discovery) / len(discovery)
    output["fraction_delta_discovery_zero"] = sum(abs(value) <= 1e-12 for value in discovery) / len(discovery)
    output["fraction_delta_discovery_negative"] = sum(value < -1e-12 for value in discovery) / len(discovery)
    return output


def aggregate(summaries):
    identity = {"phase", "replicate_id", "condition_id", "shift_family", "shift_severity", "ef_search"}
    numeric = [key for key, value in summaries[0].items() if key not in identity and value is not None]
    groups = {}
    for row in summaries:
        key = (row["phase"], row["condition_id"], row["shift_family"], row["shift_severity"], row["ef_search"])
        groups.setdefault(key, []).append(row)
    output = []
    for key, rows in sorted(groups.items()):
        record = dict(zip(("phase", "condition_id", "shift_family", "shift_severity", "ef_search"), key))
        record["replicate_count"] = len(rows)
        for field in numeric:
            values = [row[field] for row in rows if row[field] is not None]
            record[f"{field}_across_replicates_mean"] = statistics.fmean(values)
            record[f"{field}_across_replicates_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        output.append(record)
    return output


def expanded_family(rows, family):
    selected = []
    for row in rows:
        if row["shift_family"] == family:
            selected.append(row)
        elif row["shift_family"] == "iid":
            copy = dict(row)
            copy["shift_family"] = family
            copy["shift_severity"] = 0.0 if family == "mean" else 1.0
            selected.append(copy)
    return sorted(selected, key=lambda row: (row["shift_severity"], row["replicate_id"]))


def line_svg(path, title, series, y_label, x_label):
    width, height, left, right, top, bottom = 840, 500, 82, 28, 48, 66
    points = [(x, y) for _, values, _ in series for x, y in values]
    x_min, x_max = min(x for x, _ in points), max(x for x, _ in points)
    y_min, y_max = min(y for _, y in points), max(y for _, y in points)
    padding = max((y_max - y_min) * .12, .003)
    y_min -= padding; y_max += padding
    pw, ph = width - left - right, height - top - bottom
    sx = lambda x: left + (x - x_min) * pw / (x_max - x_min)
    sy = lambda y: top + (y_max - y) * ph / (y_max - y_min)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>', f'<text x="{width/2}" y="25" text-anchor="middle" font-family="sans-serif" font-size="17">{html.escape(title)}</text>', f'<line x1="{left}" y1="{top+ph}" x2="{left+pw}" y2="{top+ph}" stroke="black"/>', f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+ph}" stroke="black"/>']
    for tick in range(6):
        y = y_min + (y_max-y_min)*tick/5
        parts.append(f'<text x="{left-8}" y="{sy(y)+4:.2f}" text-anchor="end" font-family="sans-serif" font-size="11">{y:.3f}</text>')
    for x in sorted(set(x for x, _ in points)):
        parts.append(f'<text x="{sx(x):.2f}" y="{top+ph+20}" text-anchor="middle" font-family="sans-serif" font-size="11">{x:g}</text>')
    parts.extend([f'<text x="{width/2}" y="{height-12}" text-anchor="middle" font-family="sans-serif" font-size="13">{html.escape(x_label)}</text>', f'<text x="18" y="{top+ph/2}" text-anchor="middle" transform="rotate(-90 18 {top+ph/2})" font-family="sans-serif" font-size="13">{html.escape(y_label)}</text>'])
    for index, (label, values, color) in enumerate(series):
        coords = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in values)
        parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>')
        for x, y in values:
            parts.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="3" fill="{color}"/>')
        parts.append(f'<text x="{left+10}" y="{top+16+index*17}" font-family="sans-serif" font-size="11" fill="{color}">{html.escape(label)}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def scatter_svg(path, title, rows, descriptor):
    values = [(row[descriptor], row["delta_discovery"]) for row in rows]
    width, height, left, right, top, bottom = 840, 500, 82, 28, 48, 66
    xmin, xmax = min(x for x, _ in values), max(x for x, _ in values)
    ymin, ymax = min(y for _, y in values), max(y for _, y in values)
    if xmin == xmax: xmax += 1
    if ymin == ymax: ymax += .1
    pw, ph = width-left-right, height-top-bottom
    sx = lambda x: left+(x-xmin)*pw/(xmax-xmin)
    sy = lambda y: top+(ymax-y)*ph/(ymax-ymin)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>', f'<text x="{width/2}" y="25" text-anchor="middle" font-family="sans-serif" font-size="17">{html.escape(title)}</text>', f'<line x1="{left}" y1="{top+ph}" x2="{left+pw}" y2="{top+ph}" stroke="black"/>', f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+ph}" stroke="black"/>', f'<text x="{width/2}" y="{height-12}" text-anchor="middle" font-family="sans-serif" font-size="13">{html.escape(descriptor)}</text>', f'<text x="18" y="{top+ph/2}" text-anchor="middle" transform="rotate(-90 18 {top+ph/2})" font-family="sans-serif" font-size="13">delta_discovery</text>']
    for x, y in values:
        parts.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="2" fill="#4c78a8" fill-opacity="0.3"/>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("table_directory", type=Path)
    parser.add_argument("figure_directory", type=Path)
    parser.add_argument("--phase1-run", type=Path, default=Path("runs/phase1_seed_robustness_v1"))
    args = parser.parse_args()
    root = json.loads((args.run_directory / "manifest.json").read_text(encoding="utf-8"))
    query_count = root["fixed_parameters"]["queries"]
    fixed_summaries, matched_summaries, all_fixed_rows = [], [], []
    fingerprints, codebook_hashes, code_hashes = set(), set(), set()
    iid_reproduced = True
    for replicate_id in range(root["replicate_count"]):
        directory = args.run_directory / f"replicate_{replicate_id:02d}"
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        fingerprints.add(manifest["graph_fingerprint"])
        codebook_hashes.add(manifest["pq_codebook_sha256"])
        code_hashes.add(manifest["pq_codes_sha256"])
        fixed_files = sorted((directory / "fixed").glob("*.jsonl"))
        if len(fixed_files) != 8:
            raise ValueError("expected eight unique fixed query conditions")
        for path in fixed_files:
            rows = read_jsonl(path)
            validate_rows(rows, query_count, "fixed", 256)
            fixed_summaries.append(summarize(rows, replicate_id, "fixed"))
            all_fixed_rows.extend(rows)
            if path.stem == "iid":
                previous = read_jsonl(args.phase1_run / f"replicate_{replicate_id:02d}" / "queries.jsonl")
                keys = ("ground_truth_ids", "exact_native_result_ids", "pq_native_result_ids", "exact_oracle_result_ids", "pq_oracle_result_ids", "recall_exact_native", "recall_pq_native", "recall_pq_oracle", "delta_discovery", "evaluated_jaccard")
                iid_reproduced &= all(all(row[key] == old[key] for key in keys) for row, old in zip(rows, previous))
        for path in sorted((directory / "matched").glob("*.jsonl")):
            rows = read_jsonl(path)
            ef = rows[0]["ef_search"]
            validate_rows(rows, query_count, "matched", ef)
            matched_summaries.append(summarize(rows, replicate_id, "matched"))
    if len(fingerprints) != 4 or len(codebook_hashes) != 4 or len(code_hashes) != 4:
        raise ValueError("replicates must have independent graph/PQ hashes")
    if not iid_reproduced:
        raise ValueError("IID control did not reproduce Phase 1 rows")

    fixed_aggregate = aggregate(fixed_summaries)
    matched_aggregate = aggregate(matched_summaries) if matched_summaries else []
    thresholds = root["pre_registered_thresholds"]
    criterion_rows = []
    for summary in fixed_summaries:
        if summary["shift_family"] == "iid":
            continue
        eligible = summary["mean_recall_exact_native"] >= .90
        criterion_rows.append({
            "replicate_id": summary["replicate_id"],
            "condition_id": summary["condition_id"],
            "eligible_exact_recall_at_least_0p90": eligible,
            "no_evidence_pass": eligible and abs(summary["mean_delta_discovery"]) <= thresholds["no_evidence_max_abs_mean_delta_discovery"] and summary["rerank_recovery"] >= thresholds["no_evidence_min_rerank_recovery"],
            "emerging_evidence_pass": eligible and summary["mean_delta_discovery"] >= thresholds["emerging_min_mean_delta_discovery"] and summary["rerank_recovery"] <= thresholds["emerging_max_rerank_recovery"],
        })
    emerging_counts = {}
    for row in criterion_rows:
        emerging_counts[row["condition_id"]] = emerging_counts.get(row["condition_id"], 0) + row["emerging_evidence_pass"]
    emerging_conditions = sorted(key for key, count in emerging_counts.items() if count >= thresholds["emerging_required_replicates"])
    eligible = [row for row in criterion_rows if row["eligible_exact_recall_at_least_0p90"]]
    no_evidence_all = bool(eligible) and all(row["no_evidence_pass"] for row in eligible)

    correlation_rows = []
    for family in ("mean", "radial"):
        family_rows = [row for row in all_fixed_rows if row["shift_family"] in (family, "iid")]
        for severity in sorted(set((0.0 if family == "mean" else 1.0) if row["shift_family"] == "iid" else row["shift_severity"] for row in family_rows)):
            condition_rows = [row for row in family_rows if ((row["shift_family"] == "iid" and severity == (0.0 if family == "mean" else 1.0)) or (row["shift_family"] == family and row["shift_severity"] == severity))]
            descriptors = DESCRIPTORS + (["projection_onto_mean_shift_direction"] if family == "mean" else [])
            for descriptor in descriptors:
                correlation_rows.append({"shift_family": family, "shift_severity": severity, "descriptor": descriptor, "spearman_rho": spearman([row[descriptor] for row in condition_rows], [row["delta_discovery"] for row in condition_rows]), "query_count": len(condition_rows)})

    with (args.run_directory / "exact_recall_calibration.csv").open(encoding="utf-8") as handle:
        calibration_rows = list(csv.DictReader(handle))
    for row in calibration_rows:
        row["replicate_id"] = int(row["replicate_id"])
        row["ef_search"] = int(row["ef_search"])
        row["mean_exact_recall"] = float(row["mean_exact_recall"])
        row["selected_matched_ef"] = int(row["selected_matched_ef"])

    fixed_nonzero_controls = sum(abs(row["delta_exact_control"]) > 1e-12 for row in all_fixed_rows)
    fixed_max_control = max(abs(row["delta_exact_control"]) for row in all_fixed_rows)
    fixed_mean_control = statistics.fmean(row["delta_exact_control"] for row in all_fixed_rows)
    payload = {
        "schema_version": 1,
        "source_run": str(args.run_directory),
        "pre_registered_thresholds": thresholds,
        "validation": {"four_unique_graph_fingerprints": True, "four_unique_pq_codebook_hashes": True, "four_unique_pq_code_hashes": True, "all_delta_exact_control_zero": fixed_nonzero_controls == 0, "nonzero_delta_exact_control_rows": fixed_nonzero_controls, "mean_delta_exact_control": fixed_mean_control, "max_absolute_delta_exact_control": fixed_max_control, "iid_control_exactly_reproduces_phase1_replicates_0_to_3": iid_reproduced, "sampled_candidate_sets_validated": True},
        "criteria": {"A_no_evidence_holds_for_all_eligible_fixed_rows": no_evidence_all, "B_emerging_evidence_conditions": emerging_conditions, "B_emerging_evidence_holds": bool(emerging_conditions), "per_replicate_condition": criterion_rows},
        "replicate_summaries": fixed_summaries,
        "across_replicate_summaries": fixed_aggregate,
    }
    matched_payload = {"schema_version": 1, "source_run": str(args.run_directory), "replicate_summaries": matched_summaries, "across_replicate_summaries": matched_aggregate}

    args.table_directory.mkdir(parents=True, exist_ok=True)
    args.figure_directory.mkdir(parents=True, exist_ok=True)
    outputs = [
        args.table_directory / "phase2a_query_shift_fixed.csv",
        args.table_directory / "phase2a_query_shift_fixed.json",
        args.table_directory / "phase2a_query_shift_matched.csv",
        args.table_directory / "phase2a_query_shift_matched.json",
        args.table_directory / "phase2a_query_shift_spearman.csv",
        args.table_directory / "phase2a_query_shift_spearman.json",
        args.table_directory / "phase2a_query_shift_ef_calibration.csv",
        args.table_directory / "phase2a_query_shift_ef_calibration.json",
        args.table_directory / "phase2a_query_shift_fixed_aggregate.csv",
        args.table_directory / "phase2a_query_shift_matched_aggregate.csv",
    ]
    for path in outputs:
        if path.exists(): raise FileExistsError(f"refusing to overwrite {path}")
    write_csv(outputs[0], fixed_summaries)
    outputs[1].write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(outputs[2], matched_summaries)
    outputs[3].write_text(json.dumps(matched_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(outputs[4], correlation_rows)
    outputs[5].write_text(json.dumps(correlation_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(outputs[6], calibration_rows)
    outputs[7].write_text(json.dumps(calibration_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(outputs[8], fixed_aggregate)
    write_csv(outputs[9], matched_aggregate)

    colors = ["#4c78a8", "#f58518", "#54a24b"]
    plot_specs = [
        ("delta_discovery", "mean_delta_discovery", "Mean discovery delta"),
        ("delta_ranking", "mean_delta_ranking", "Mean ranking delta"),
        ("rerank_recovery", "rerank_recovery", "Rerank recovery"),
        ("jaccard", "mean_evaluated_set_jaccard", "Evaluated-set Jaccard"),
    ]
    for family in ("mean", "radial"):
        family_summaries = expanded_family(fixed_summaries, family)
        xlabel = "alpha" if family == "mean" else "sigma"
        for stem, field, ylabel in plot_specs:
            series = []
            for replicate_id in range(4):
                rows = [row for row in family_summaries if row["replicate_id"] == replicate_id]
                series.append((f"replicate {replicate_id}", [(row["shift_severity"], row[field]) for row in rows], ["#9ecae1", "#fdae6b", "#a1d99b", "#bcbddc"][replicate_id]))
            line_svg(args.figure_directory / f"phase2a_{family}_{stem}.svg", f"{family.title()} shift: {ylabel}", series, ylabel, xlabel)
        recall_series = []
        for label, field, color in zip(("exact native", "PQ native", "PQ exact-rerank"), ("mean_recall_exact_native", "mean_recall_pq_native", "mean_recall_pq_oracle"), colors):
            severities = sorted(set(row["shift_severity"] for row in family_summaries))
            values = [(severity, statistics.fmean(row[field] for row in family_summaries if row["shift_severity"] == severity)) for severity in severities]
            recall_series.append((label, values, color))
        line_svg(args.figure_directory / f"phase2a_{family}_recall.svg", f"{family.title()} shift: mean recall@10", recall_series, "Mean recall@10", xlabel)
        corr = [row for row in correlation_rows if row["shift_family"] == family and row["shift_severity"] != (0.0 if family == "mean" else 1.0) and row["spearman_rho"] is not None]
        strongest = max(corr, key=lambda row: abs(row["spearman_rho"]))["descriptor"]
        shifted_rows = [row for row in all_fixed_rows if row["shift_family"] == family]
        scatter_svg(args.figure_directory / f"phase2a_{family}_descriptor_scatter.svg", f"{family.title()} shifts: strongest exploratory descriptor", shifted_rows, strongest)
    print(json.dumps({"fixed_replicate_rows": len(fixed_summaries), "matched_replicate_rows": len(matched_summaries), "no_evidence_all": no_evidence_all, "emerging_conditions": emerging_conditions}, indent=2))


if __name__ == "__main__":
    main()
