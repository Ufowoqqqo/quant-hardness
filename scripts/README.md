# Experiment entry points

Planned non-interactive entry points:

- `build_index.py`: build or import a graph index from configuration.
- `run_search.py`: run paired exact-vector and quantized-vector searches.
- `sweep.py`: execute deterministic configuration sweeps.
- `analyze.py`: derive tables and figures from immutable raw runs.

Implement these only when their input/output contracts and baseline systems are
specified. Keep measurement logic under `src/instrumentation/` and metrics under
`src/metrics/`.

Implemented Phase 1 entry point:

- `analyze_paired_recall.py`: validates the paired query order and delta sign,
  computes predefined descriptive statistics, and emits CSV/SVG derived
  artifacts without overwriting existing output.
- `analyze_calibration.py`: validates all exact/PQ calibration rows and sampled
  distance diagnostics, emits complete calibration tables, and plots the
  predeclared efSearch/PQ sweep plus explicitly selected operating points.
- `analyze_decomposition.py`: validates recorded distance-evaluation sets,
  recomputes every recall/coverage/delta, checks native IDs against calibration,
  and emits the Phase 1 discovery-versus-ranking tables and figures.
