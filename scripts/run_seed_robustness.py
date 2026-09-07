#!/usr/bin/env python3
"""Run pre-registered decomposition replicates through the shared C++ runner."""

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def parse_config(path):
    values = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0]
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key in values or not key or not value:
            raise ValueError(f"invalid or duplicate config key: {key}")
        values[key] = value
    expected = {
        "seed_tuples", "base_vectors", "queries", "dimension", "k",
        "hnsw_m", "ef_construction", "ef_search", "pq_m", "pq_nbits",
        "required_passing_replicates", "max_abs_mean_delta_discovery",
        "min_rerank_recovery",
    }
    if set(values) != expected:
        raise ValueError(f"config keys differ: {set(values) ^ expected}")
    tuples = []
    for item in values["seed_tuples"].split(","):
        seeds = tuple(int(value) for value in item.split(":"))
        if len(seeds) != 4 or any(value <= 0 for value in seeds):
            raise ValueError("each seed tuple must contain four positive integers")
        tuples.append(seeds)
    if len(tuples) != 8 or len(set(tuples)) != 8:
        raise ValueError("exactly eight distinct seed tuples are required")
    return values, tuples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("decomposition_binary", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    values, seed_tuples = parse_config(args.config)
    if args.output_directory.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_directory}")
    args.output_directory.mkdir(parents=True)
    (args.output_directory / "resolved_config.conf").write_text(
        args.config.read_text(encoding="utf-8"), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "run_id": "phase1_seed_robustness_v1",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "distribution": "independent isotropic standard-normal FP32 database and queries",
        "replicate_count": 8,
        "seed_tuple_fields": ["database_seed", "query_seed", "graph_seed", "pq_seed"],
        "seed_tuples": seed_tuples,
        "fixed_parameters": {key: int(values[key]) for key in (
            "base_vectors", "queries", "dimension", "k", "hnsw_m",
            "ef_construction", "ef_search", "pq_m", "pq_nbits")},
        "pre_registered_criterion": {
            "required_passing_replicates": int(values["required_passing_replicates"]),
            "max_abs_mean_delta_discovery": float(values["max_abs_mean_delta_discovery"]),
            "min_rerank_recovery": float(values["min_rerank_recovery"]),
        },
    }
    (args.output_directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    for replicate_id, seeds in enumerate(seed_tuples):
        database_seed, query_seed, graph_seed, pq_seed = seeds
        replicate_name = f"replicate_{replicate_id:02d}"
        generated_config = args.output_directory / f"{replicate_name}.conf"
        generated_config.write_text(
            "\n".join([
                f"database_seed={database_seed}",
                f"query_seed={query_seed}",
                f"graph_seed={graph_seed}",
                f"pq_seed={pq_seed}",
                f"base_vectors={values['base_vectors']}",
                f"queries={values['queries']}",
                f"dimension={values['dimension']}",
                f"k={values['k']}",
                f"hnsw_m={values['hnsw_m']}",
                f"ef_construction={values['ef_construction']}",
                f"ef_search_values={values['ef_search']}",
                f"pq_m={values['pq_m']}",
                f"pq_nbits={values['pq_nbits']}",
                "expected_graph_fingerprint=auto",
                f"run_id=phase1_seed_robustness_v1_{replicate_name}",
            ]) + "\n",
            encoding="utf-8",
        )
        command = [str(args.decomposition_binary), str(generated_config),
                   str(args.output_directory / replicate_name)]
        print(f"starting {replicate_name}", flush=True)
        subprocess.run(command, check=True)
    print("status=PASS", flush=True)


if __name__ == "__main__":
    main()
