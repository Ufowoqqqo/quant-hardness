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
- `run_seed_robustness.py`: executes eight pre-registered seed tuples through
  the shared decomposition binary, preserving each replicate separately.
- `analyze_seed_robustness.py`: validates each replicate, evaluates the
  pre-registered stability criterion, aggregates across seeds, and emits only
  the requested summary table and boxplots.
- `analyze_phase2a_query_shift.py`: validates the frozen-index query-shift
  run, checks exact Phase 1 IID reproduction, evaluates the pre-registered
  mechanism criteria, and emits fixed-ef/matched-recall tables, correlations,
  and SVG figures.
- `analyze_phase2b_clustered_geometry.py`: validates corrected L0 candidate
  sets, PQ-quality samples, and the paired clustered-mixture sweep, then emits
  replicate/aggregate tables, exploratory correlations, and SVG figures.
- `faiss_sift1m` (built C++ runner): executes Phase 3A as separate `prepare`,
  `calibrate`, and `decompose` stages so the expensive frozen HNSW/PQ indexes
  are serialized once and reused. Every stage is non-interactive; preparation
  refuses to overwrite a run or cache directory.
- `analyze_phase3a_sift1m.py`: validates all raw SIFT1M rows and regenerates
  gatekeeper tables and the limited, predeclared SVG figures.
