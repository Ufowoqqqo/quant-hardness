# Quantization-Induced Query Hardness in Graph ANN

This repository studies when and why quantization changes the difficulty of
individual queries in graph-based approximate nearest-neighbor search. The
working hypothesis is not assumed to be true; the current phase establishes
comparable exact-vector and quantized-vector baselines and attempts to falsify
the proposed phenomenon.

## Repository boundaries

- `third_party/` contains upstream ANN implementations such as FAISS and
  DiskANN. Keep vendored code, submodules, and upstream patches here.
- `src/instrumentation/` contains repository-owned measurement hooks and data
  collection. Do not place this code inside vendored implementations.
- `src/graph/`, `src/quantization/`, and `src/metrics/` contain only
  repository-owned adapters, reference implementations, and measurements.
- `configs/` is the source of experiment parameters.
- `runs/` stores immutable raw run outputs, including per-query measurements
  and provenance.
- `results/` stores derived tables and figures; every artifact should identify
  its source run(s).
- `docs/experiment_log.md` is the chronological experiment record.

This separation is important because experiments will compare multiple graph
and quantization systems, including HNSW, Vamana, PQ, and RaBitQ, without
combining their implementations with the measurement layer.

## Current status

The repository is in the phenomenon-validation phase. No baseline result or
claim has been established yet.

