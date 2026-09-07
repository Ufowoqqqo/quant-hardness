# Phase 2 plan: distribution dependence of discovery degradation

## Status and objective

This was the prepared broad Phase 2 plan. Its controlled query-shift component
has now been executed as Phase 2A; see `docs/phase2a_query_shift.md`. The
clustered component has now been executed as Phase 2B; see
`docs/phase2b_clustered_geometry.md`. The anisotropic condition remains unrun.

Phase 1 found seed-stable ranking dominance for IID isotropic Gaussian data at
`efSearch=256`, PQ32×8. Phase 2 will test whether candidate-discovery
degradation emerges under structured, anisotropic, or shifted query
distributions. The hypothesis is not assumed true.

## Fixed experimental method

Use the existing FAISS fixed-topology decomposition without trajectory logging.
Within every condition and seed:

1. generate the database and training/query sets from the declared
   distribution;
2. compute exhaustive FP32 top-10 ground truth;
3. build one HNSW graph using FP32 squared-L2 distances;
4. train one standard FAISS PQ32×8 model on all database vectors;
5. run exact and PQ traversal over the identical graph at `k=10` and
   `efSearch=256`;
6. record unique query-to-node distance-evaluated sets and exact-rerank both;
7. verify graph fingerprints and uninstrumented/instrumented native identity.

Keep database size 20,000, query count 500, dimension 64, HNSW `M=16`,
`efConstruction=80`, one thread, and ascending query order. Use three
pre-registered independent database/query/graph/PQ seed tuples per condition.
The same tuple index should be paired across conditions; seed values must be
written to configuration before execution.

Do not tune distribution parameters or search/PQ parameters in response to
delta distributions. If exact native recall differs materially across
conditions, report that as a confounder; do not silently change `efSearch`.

## Distribution conditions

All generation is FP32 and has no normalization unless stated.

### A. IID isotropic reference

- Database and queries: independent `N(0, I_64)`.
- Purpose: reproduce the Phase 1 reference inside the paired Phase 2 matrix.

### B. Clustered / mixture synthetic data

- 32 equal-probability clusters.
- For each seed, draw centers independently from `N(0, 4 I_64)`.
- Draw each database/query vector as its assigned center plus independent
  `N(0, I_64)` residual noise.
- Database and queries use independent cluster assignments but the same fixed
  center set within a replicate.
- Purpose: introduce local-density and inter-cluster structure without an OOD
  shift.

### C. Anisotropic synthetic data

- Database and queries: independent zero-mean diagonal Gaussian.
- Coordinate standard deviations are log-spaced from 1 to 10 across 64
  dimensions, fixed in coordinate order.
- No whitening or normalization.
- Purpose: test sensitivity to direction-dependent scale and distance
  concentration changes.

### D. Distribution-shifted / OOD queries

- Database and PQ training data: `N(0, I_64)`, identical in construction to
  condition A for the paired seed.
- Queries: `N(mu, I_64)`, where `mu=(2,0,...,0)`.
- Reuse condition A's database, HNSW graph, and PQ codes for the matching seed;
  only the query set and ground truth differ.
- Purpose: isolate a moderate query mean shift without changing graph or
  quantizer training distribution.

The numeric mixture separation, anisotropy condition number, and OOD shift are
predeclared here. Any later sensitivity sweep must be a separately named
experiment, not a silent modification of this primary comparison.

## Preserved metrics

For every query preserve the Phase 1 definitions and raw candidate sets:

- exact/PQ native Recall@10;
- exact/PQ candidate coverage and oracle Recall@10;
- `delta_total`, `delta_discovery`, `delta_ranking`, and
  `delta_exact_control` without truncating negative values;
- unique exact/PQ distance-evaluated node counts;
- evaluated-set intersection and Jaccard;
- ordered native/oracle/ground-truth result IDs.

For every condition and seed report the same replicate summaries as
`docs/phase1_seed_robustness.md`, plus per-query delta quantiles and sign
fractions. Aggregate across the three seeds using mean, sample standard
deviation, minimum, and maximum. Preserve raw per-query rows rather than only
aggregates.

## Primary comparisons

The primary outcome is mean `delta_discovery`; rerank recovery is the paired
mechanism check. Compare B, C, and D with A at matching seed index. Also report:

- fraction of queries with positive/zero/negative discovery delta;
- candidate coverage difference and evaluated-set Jaccard;
- whether exact native recall or unique evaluation count shifts across
  distributions;
- whether any apparent discovery effect is consistent across all three seeds.

No heavy-tail or hard-query criterion is pre-registered for Phase 2. Describe
the observed distributions and potential confounders without converting an
exploratory contrast into confirmation.

## Outputs and execution gate

Planned outputs:

```text
configs/indexes/faiss_hnsw_phase2_distributions.conf
runs/phase2_distributions_v1/<condition>/<replicate>/
results/tables/phase2_distributions.{csv,json}
results/figures/phase2_distributions_*.svg
docs/phase2_distributions.md
```

Before execution, add unit tests for each distribution generator and record
checksums or deterministic summary moments for generated database/query arrays.
Confirm disk requirements before preserving all candidate sets. Append the
exact command and resolved configuration to `docs/experiment_log.md` only when
the experiment is actually run.
