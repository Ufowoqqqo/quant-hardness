#!/usr/bin/env python3
"""Validate and analyze the pre-registered Phase 2B clustered experiment."""

import argparse
import csv
import html
import json
import math
import statistics
from pathlib import Path


DELTAS = ("delta_total", "delta_discovery", "delta_ranking")
DESCRIPTORS = (
    "distance_to_generating_center",
    "distance_to_nearest_center",
    "distance_to_second_center",
    "cluster_center_margin",
    "normalized_cluster_center_margin",
    "exact_nn_distance_squared",
    "exact_10th_distance_squared",
    "nn_margin_squared",
    "normalized_nn_margin_by_10th",
    "query_l2_norm",
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
        raise ValueError("result is not a unique top-10")
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
    denominator = math.sqrt(sum((x - mx) ** 2 for x in rx) * sum((y - my) ** 2 for y in ry))
    return numerator / denominator if denominator else None


def validate_queries(rows, query_count, fingerprint):
    if len(rows) != query_count or [row["query_id"] for row in rows] != list(range(query_count)):
        raise ValueError("query rows/order mismatch")
    for row in rows:
        if row["graph_fingerprint"] != fingerprint or row["phase"] != "fixed" or row["ef_search"] != 256:
            raise ValueError("query condition metadata mismatch")
        truth = row["ground_truth_ids"]
        metric_ids = {
            "recall_exact_native": "exact_native_result_ids",
            "recall_pq_native": "pq_native_result_ids",
            "recall_exact_L0_oracle": "exact_L0_oracle_result_ids",
            "recall_pq_L0_oracle": "pq_L0_oracle_result_ids",
        }
        for metric, ids in metric_ids.items():
            if not math.isclose(row[metric], recall(row[ids], truth), abs_tol=1e-12):
                raise ValueError(f"{metric} mismatch")
        if row["recall_exact_L0_oracle"] != row["coverage_exact_L0"] or row["recall_pq_L0_oracle"] != row["coverage_pq_L0"]:
            raise ValueError("L0 oracle/coverage mismatch")
        expected = {
            "delta_total": row["recall_exact_native"] - row["recall_pq_native"],
            "delta_discovery": row["recall_exact_L0_oracle"] - row["recall_pq_L0_oracle"],
            "delta_ranking": row["recall_pq_L0_oracle"] - row["recall_pq_native"],
            "delta_exact_control": row["recall_exact_L0_oracle"] - row["recall_exact_native"],
        }
        for field, value in expected.items():
            if not math.isclose(row[field], value, abs_tol=1e-12):
                raise ValueError(f"{field} mismatch")
        if row["delta_exact_control"] != 0:
            raise ValueError("corrected L0 exact control is nonzero")
        sampled = row["query_id"] < 25
        fields = ("exact_L0_evaluated_ids", "pq_L0_evaluated_ids", "exact_upper_only_evaluated_ids", "pq_upper_only_evaluated_ids")
        if sampled != all(row[field] is not None for field in fields):
            raise ValueError("candidate-set sampling mismatch")
        if sampled:
            for field in fields:
                if row[field] != sorted(set(row[field])):
                    raise ValueError("candidate IDs are not sorted unique")
            exact_l0, pq_l0 = row[fields[0]], row[fields[1]]
            if len(exact_l0) != row["exact_L0_unique_distance_evaluations"] or len(pq_l0) != row["pq_L0_unique_distance_evaluations"]:
                raise ValueError("L0 count mismatch")
            if set(exact_l0) & set(row[fields[2]]) or set(pq_l0) & set(row[fields[3]]):
                raise ValueError("upper-only evaluation leaked into V_L0")
            intersection = len(set(exact_l0) & set(pq_l0))
            union = len(set(exact_l0) | set(pq_l0))
            if intersection != row["evaluated_L0_intersection_size"] or not math.isclose(intersection / union, row["evaluated_L0_jaccard"], abs_tol=1e-12):
                raise ValueError("L0 overlap mismatch")
            if len(set(exact_l0) & set(truth)) / 10 != row["coverage_exact_L0"] or len(set(pq_l0) & set(truth)) / 10 != row["coverage_pq_L0"]:
                raise ValueError("L0 sampled coverage mismatch")


def validate_quality(distances, orders):
    if len(distances) != 10000 or len(orders) != 10000:
        raise ValueError("PQ-quality sample count mismatch")
    for row in distances:
        expected = abs(row["pq_distance"] - row["exact_distance"])
        if not math.isclose(expected, row["absolute_error"], rel_tol=1e-6, abs_tol=1e-5):
            raise ValueError("absolute distance error mismatch")
        relative = expected / row["exact_distance"]
        if not math.isclose(relative, row["absolute_relative_error"], rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("relative distance error mismatch")
    for row in orders:
        expected = ((row["exact_first"] < row["exact_second"] and row["pq_first"] > row["pq_second"]) or
                    (row["exact_first"] > row["exact_second"] and row["pq_first"] < row["pq_second"]))
        if expected != row["strict_inversion"] or ((row["pq_first"] == row["pq_second"]) != row["pq_tie"]):
            raise ValueError("pairwise order classification mismatch")


def summarize(replicate_id, rho, rows, distances, orders):
    mean = lambda field: statistics.fmean(row[field] for row in rows)
    exact, pq_native, pq_oracle = mean("recall_exact_native"), mean("recall_pq_native"), mean("recall_pq_L0_oracle")
    denominator = exact - pq_native
    output = {
        "replicate_id": replicate_id, "rho": rho, "ef_search": 256,
        "mean_recall_exact_native": exact,
        "mean_recall_pq_native": pq_native,
        "mean_recall_pq_L0_oracle": pq_oracle,
        "rerank_recovery": (pq_oracle - pq_native) / denominator if abs(denominator) > 1e-12 else None,
        "mean_coverage_exact_L0": mean("coverage_exact_L0"),
        "mean_coverage_pq_L0": mean("coverage_pq_L0"),
        "mean_evaluated_L0_jaccard": mean("evaluated_L0_jaccard"),
        "mean_exact_L0_unique_distance_evaluations": mean("exact_L0_unique_distance_evaluations"),
        "mean_pq_L0_unique_distance_evaluations": mean("pq_L0_unique_distance_evaluations"),
        "mean_exact_upper_only_unique_distance_evaluations": mean("exact_upper_only_unique_distance_evaluations"),
        "mean_pq_upper_only_unique_distance_evaluations": mean("pq_upper_only_unique_distance_evaluations"),
        "fraction_ground_truth_spans_multiple_components": statistics.fmean(row["ground_truth_spans_multiple_components"] for row in rows),
        "max_absolute_delta_exact_control": max(abs(row["delta_exact_control"]) for row in rows),
    }
    for field in DELTAS:
        values = [row[field] for row in rows]
        output[f"mean_{field}"] = statistics.fmean(values)
        output[f"median_{field}"] = statistics.median(values)
        for label, probability in (("p90", .90), ("p95", .95), ("p99", .99)):
            output[f"{label}_{field}"] = percentile(values, probability)
    discovery = [row["delta_discovery"] for row in rows]
    output["fraction_delta_discovery_positive"] = sum(x > 1e-12 for x in discovery) / len(rows)
    output["fraction_delta_discovery_zero"] = sum(abs(x) <= 1e-12 for x in discovery) / len(rows)
    output["fraction_delta_discovery_negative"] = sum(x < -1e-12 for x in discovery) / len(rows)
    absolute = [row["absolute_error"] for row in distances]
    relative = [row["absolute_relative_error"] for row in distances]
    output.update({
        "pq_mean_absolute_distance_error": statistics.fmean(absolute),
        "pq_median_absolute_distance_error": statistics.median(absolute),
        "pq_p95_absolute_distance_error": percentile(absolute, .95),
        "pq_mean_absolute_relative_distance_error": statistics.fmean(relative),
        "pq_median_absolute_relative_distance_error": statistics.median(relative),
        "pq_pairwise_strict_inversion_rate": statistics.fmean(row["strict_inversion"] for row in orders),
        "pq_pairwise_tie_rate": statistics.fmean(row["pq_tie"] for row in orders),
    })
    for descriptor, prefix in (("cluster_center_margin", "cluster_margin"), ("nn_margin_squared", "nn_margin")):
        cutoff = percentile([row[descriptor] for row in rows], .25)
        low = [row["delta_discovery"] for row in rows if row[descriptor] <= cutoff]
        other = [row["delta_discovery"] for row in rows if row[descriptor] > cutoff]
        output[f"mean_delta_discovery_lowest_{prefix}_quartile"] = statistics.fmean(low)
        output[f"mean_delta_discovery_other_{prefix}_quartiles"] = statistics.fmean(other) if other else None
    return output


def aggregate(replicates):
    identity = {"replicate_id", "rho", "ef_search"}
    fields = [key for key, value in replicates[0].items() if key not in identity and value is not None]
    output = []
    for rho in sorted(set(row["rho"] for row in replicates)):
        rows = [row for row in replicates if row["rho"] == rho]
        record = {"rho": rho, "replicate_count": len(rows), "ef_search": 256}
        for field in fields:
            values = [row[field] for row in rows if row[field] is not None]
            record[f"{field}_across_replicates_mean"] = statistics.fmean(values)
            record[f"{field}_across_replicates_std"] = statistics.stdev(values)
        output.append(record)
    return output


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def line_svg(path, title, series, y_label):
    width, height, left, right, top, bottom = 840, 500, 82, 28, 48, 62
    points = [(x, y) for _, values, _ in series for x, y in values]
    xmin, xmax = min(x for x, _ in points), max(x for x, _ in points)
    ymin, ymax = min(y for _, y in points), max(y for _, y in points)
    pad = max((ymax - ymin) * .12, .002); ymin -= pad; ymax += pad
    pw, ph = width-left-right, height-top-bottom
    sx = lambda x: left+(x-xmin)*pw/(xmax-xmin)
    sy = lambda y: top+(ymax-y)*ph/(ymax-ymin)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>', f'<text x="{width/2}" y="25" text-anchor="middle" font-family="sans-serif" font-size="17">{html.escape(title)}</text>', f'<line x1="{left}" y1="{top+ph}" x2="{left+pw}" y2="{top+ph}" stroke="black"/>', f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+ph}" stroke="black"/>', f'<text x="{width/2}" y="{height-10}" text-anchor="middle" font-family="sans-serif" font-size="13">rho</text>', f'<text x="18" y="{top+ph/2}" text-anchor="middle" transform="rotate(-90 18 {top+ph/2})" font-family="sans-serif" font-size="13">{html.escape(y_label)}</text>']
    for tick in range(6):
        value = ymin+(ymax-ymin)*tick/5
        parts.append(f'<text x="{left-7}" y="{sy(value)+4:.2f}" text-anchor="end" font-family="sans-serif" font-size="11">{value:.3f}</text>')
    for x in sorted(set(x for x, _ in points)):
        parts.append(f'<text x="{sx(x):.2f}" y="{top+ph+20}" text-anchor="middle" font-family="sans-serif" font-size="11">{x:g}</text>')
    for index, (label, values, color) in enumerate(series):
        coords = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in values)
        parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>')
        for x, y in values: parts.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="3" fill="{color}"/>')
        parts.append(f'<text x="{left+10}" y="{top+16+index*17}" font-family="sans-serif" font-size="11" fill="{color}">{html.escape(label)}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts)+"\n", encoding="utf-8")


def scatter_svg(path, title, rows, x_field):
    width, height, left, right, top, bottom = 840, 500, 82, 28, 48, 62
    values = [(row[x_field], row["delta_discovery"], row["rho"]) for row in rows]
    xmin, xmax = min(x for x, _, _ in values), max(x for x, _, _ in values)
    ymin, ymax = min(y for _, y, _ in values), max(y for _, y, _ in values)
    pw, ph = width-left-right, height-top-bottom
    sx=lambda x:left+(x-xmin)*pw/(xmax-xmin); sy=lambda y:top+(ymax-y)*ph/(ymax-ymin)
    colors={0.0:"#999999",.1:"#4c78a8",.25:"#54a24b",.5:"#f58518",.75:"#b279a2",.9:"#e45756"}
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">','<rect width="100%" height="100%" fill="white"/>',f'<text x="{width/2}" y="25" text-anchor="middle" font-family="sans-serif" font-size="17">{html.escape(title)}</text>',f'<line x1="{left}" y1="{top+ph}" x2="{left+pw}" y2="{top+ph}" stroke="black"/>',f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+ph}" stroke="black"/>',f'<text x="{width/2}" y="{height-10}" text-anchor="middle" font-family="sans-serif" font-size="13">{html.escape(x_field)}</text>',f'<text x="18" y="{top+ph/2}" text-anchor="middle" transform="rotate(-90 18 {top+ph/2})" font-family="sans-serif" font-size="13">delta_discovery</text>']
    for x,y,rho in values: parts.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="1.5" fill="{colors[rho]}" fill-opacity="0.25"/>')
    parts.append('</svg>'); path.write_text("\n".join(parts)+"\n",encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("table_directory", type=Path)
    parser.add_argument("figure_directory", type=Path)
    parser.add_argument("--phase1-run", type=Path, default=Path("runs/phase1_seed_robustness_v1"))
    args = parser.parse_args()
    root = json.loads((args.run_directory/"manifest.json").read_text())
    query_count = root["fixed_parameters"]["queries"]
    summaries, all_rows, correlations = [], [], []
    fingerprints = set()
    rho0_native_reproduced = True
    for replicate_id in range(4):
        replicate_dir = args.run_directory/f"replicate_{replicate_id:02d}"
        condition_dirs = sorted(replicate_dir.glob("rho_*"))
        if len(condition_dirs) != 6: raise ValueError("expected six rho conditions")
        for directory in condition_dirs:
            manifest = json.loads((directory/"manifest.json").read_text())
            rho, fingerprint = manifest["rho"], manifest["graph_fingerprint"]
            fingerprints.add(fingerprint)
            rows = read_jsonl(directory/"fixed.jsonl")
            distances, orders = read_jsonl(directory/"distance_samples.jsonl"), read_jsonl(directory/"order_samples.jsonl")
            validate_queries(rows, query_count, fingerprint); validate_quality(distances, orders)
            summaries.append(summarize(replicate_id, rho, rows, distances, orders)); all_rows.extend(rows)
            if rho == 0:
                old = read_jsonl(args.phase1_run/f"replicate_{replicate_id:02d}"/"queries.jsonl")
                keys=("ground_truth_ids","exact_native_result_ids","pq_native_result_ids","recall_exact_native","recall_pq_native")
                rho0_native_reproduced &= all(all(row[key] == previous[key] for key in keys) for row,previous in zip(rows,old))
            for descriptor in DESCRIPTORS:
                correlations.append({"replicate_id":replicate_id,"rho":rho,"descriptor":descriptor,"spearman_rho":spearman([row[descriptor] for row in rows],[row["delta_discovery"] for row in rows]),"query_count":len(rows)})
    if len(fingerprints) != 24: raise ValueError("each replicate/rho must have a distinct graph")
    if not rho0_native_reproduced: raise ValueError("rho=0 did not reproduce Phase 1 native baseline")
    aggregate_rows = aggregate(summaries)
    thresholds=root["pre_registered_thresholds"]
    no_evidence=all(abs(row["mean_delta_discovery"])<=thresholds["no_evidence_max_abs_mean_delta_discovery"] and row["rerank_recovery"]>=thresholds["no_evidence_min_rerank_recovery"] for row in summaries if row["mean_recall_exact_native"]>=.90)
    emerging_counts={rho:sum(row["mean_delta_discovery"]>=thresholds["emerging_min_mean_delta_discovery"] and row["rerank_recovery"]<=thresholds["emerging_max_rerank_recovery"] and row["mean_recall_exact_native"]>=.90 for row in summaries if row["rho"]==rho) for rho in sorted(set(row["rho"] for row in summaries)) if rho>0}
    emerging=[rho for rho,count in emerging_counts.items() if count>=thresholds["emerging_required_replicates"]]
    aggregate_discovery=[row["mean_delta_discovery_across_replicates_mean"] for row in aggregate_rows]
    aggregate_inversion=[row["pq_pairwise_strict_inversion_rate_across_replicates_mean"] for row in aggregate_rows]
    payload={"schema_version":1,"source_run":str(args.run_directory),"pre_registered_thresholds":thresholds,"validation":{"all_delta_exact_control_zero":True,"upper_only_disjoint_from_L0_for_sampled_queries":True,"native_rho0_exactly_reproduces_phase1_replicates_0_to_3":rho0_native_reproduced,"unique_graph_fingerprints":24,"matched_runs":0},"criteria":{"no_discovery_breakdown":no_evidence,"emerging_rho_values":emerging,"emerging_geometry_degradation":bool(emerging),"emerging_replicate_counts":emerging_counts},"rho_level_spearman":{"mean_delta_discovery_vs_pairwise_inversion_rate":spearman(aggregate_discovery,aggregate_inversion)},"replicate_summaries":summaries,"across_replicate_summaries":aggregate_rows}
    args.table_directory.mkdir(parents=True,exist_ok=True); args.figure_directory.mkdir(parents=True,exist_ok=True)
    paths=[args.table_directory/"phase2b_clustered_replicates.csv",args.table_directory/"phase2b_clustered_summary.csv",args.table_directory/"phase2b_clustered_summary.json",args.table_directory/"phase2b_clustered_spearman.csv",args.table_directory/"phase2b_clustered_spearman.json"]
    for path in paths:
        if path.exists(): raise FileExistsError(f"refusing to overwrite {path}")
    write_csv(paths[0],summaries); write_csv(paths[1],aggregate_rows); paths[2].write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n"); write_csv(paths[3],correlations); paths[4].write_text(json.dumps(correlations,indent=2,sort_keys=True)+"\n")
    colors=["#9ecae1","#fdae6b","#a1d99b","#bcbddc"]
    specs=[("delta_discovery","mean_delta_discovery","Mean discovery delta"),("delta_ranking","mean_delta_ranking","Mean ranking delta"),("rerank_recovery","rerank_recovery","Rerank recovery"),("l0_jaccard","mean_evaluated_L0_jaccard","L0 evaluated-set Jaccard"),("pq_inversion","pq_pairwise_strict_inversion_rate","PQ strict inversion rate")]
    for stem,field,ylabel in specs:
        series=[]
        for replicate_id in range(4):
            rows=sorted((row for row in summaries if row["replicate_id"]==replicate_id),key=lambda row:row["rho"])
            series.append((f"replicate {replicate_id}",[(row["rho"],row[field]) for row in rows],colors[replicate_id]))
        line_svg(args.figure_directory/f"phase2b_{stem}.svg",ylabel+" vs clustering strength",series,ylabel)
    recall_series=[]
    for label,field,color in (("exact native","mean_recall_exact_native","#4c78a8"),("PQ native","mean_recall_pq_native","#f58518"),("PQ L0 oracle","mean_recall_pq_L0_oracle","#54a24b")):
        recall_series.append((label,[(row["rho"],row[field+"_across_replicates_mean"]) for row in aggregate_rows],color))
    line_svg(args.figure_directory/"phase2b_recall.svg","Recall vs clustering strength",recall_series,"Mean Recall@10")
    scatter_svg(args.figure_directory/"phase2b_cluster_margin_vs_discovery.svg","Cluster-center margin vs discovery delta",all_rows,"cluster_center_margin")
    scatter_svg(args.figure_directory/"phase2b_nn_margin_vs_discovery.svg","Exact NN margin vs discovery delta",all_rows,"nn_margin_squared")
    print(json.dumps({"replicate_condition_rows":len(summaries),"no_discovery_breakdown":no_evidence,"emerging_rho_values":emerging,"matched_runs":0},indent=2))


if __name__ == "__main__":
    main()
