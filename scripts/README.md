# Experiment entry points

Planned non-interactive entry points:

- `build_index.py`: build or import a graph index from configuration.
- `run_search.py`: run paired exact-vector and quantized-vector searches.
- `sweep.py`: execute deterministic configuration sweeps.
- `analyze.py`: derive tables and figures from immutable raw runs.

Implement these only when their input/output contracts and baseline systems are
specified. Keep measurement logic under `src/instrumentation/` and metrics under
`src/metrics/`.

