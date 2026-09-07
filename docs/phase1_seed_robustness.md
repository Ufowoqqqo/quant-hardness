# Phase 1 IID-Gaussian seed robustness check

## Purpose and pre-registration

This small robustness check asks whether the Phase 1 ranking-dominance result
was an artifact of one IID isotropic Gaussian database, query set, HNSW graph,
or PQ initialization. It is not intended to provide publication-quality
statistics.

Before examining any replicate result, the following criterion and eight new
seed tuples were written to
`configs/indexes/faiss_hnsw_seed_robustness.conf`:

> The IID-Gaussian ranking-dominance observation is stable if at least 7 of 8
> replicates have `abs(mean delta_discovery) <= 0.01` and
> `rerank recovery >= 0.90`.

The thresholds were not changed after execution. None of the eight tuples is
the earlier calibration tuple.

## Fixed and regenerated quantities

Each replicate independently regenerated the database, query set, FP32 HNSW
graph, and PQ training/codebook state. Fixed quantities were:

- 20,000 database vectors and 500 separate queries;
- dimension 64, IID isotropic standard-normal FP32, no preprocessing;
- HNSW `M=16`, `efConstruction=80`, exact FP32 graph construction;
- `efSearch=256`, `k=10`, one thread, ascending query IDs;
- FAISS PQ32×8 trained on all 20,000 vectors;
- exhaustive FP32 squared-L2 ground truth;
- the evaluated-node-set decomposition and metric definitions from
  `docs/phase1_decomposition.md`.

Within every replicate, exact and PQ search reused one graph. Eight distinct
graph fingerprints were produced. Uninstrumented and instrumented exact/PQ
native results were required to match before raw rows were written.

Execution used Git HEAD
`04e4f134cb9d7459e0ee94b4a8e84690e540cab5` with the robustness changes
uncommitted. The reused CMake cache caused the binary manifests to embed the
earlier configure-time commit `776bc6aab58856cfe346364674116c2fa6309cbb` and
`dirty_worktree=1`; both provenance values are retained rather than silently
rewriting the raw manifests.

## Replicate results

| Replicate | Exact native | PQ native | PQ exact-rerank | Total delta | Discovery delta | Ranking delta | Recovery | Coverage exact / PQ | Jaccard |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 0 | 0.9668 | 0.8238 | 0.9626 | 0.1430 | 0.0042 | 0.1388 | 0.9706 | 0.9668 / 0.9626 | 0.8515 |
| 1 | 0.9662 | 0.8218 | 0.9634 | 0.1444 | 0.0028 | 0.1416 | 0.9806 | 0.9662 / 0.9634 | 0.8500 |
| 2 | 0.9594 | 0.8230 | 0.9572 | 0.1364 | 0.0022 | 0.1342 | 0.9839 | 0.9594 / 0.9572 | 0.8496 |
| 3 | 0.9674 | 0.8326 | 0.9686 | 0.1348 | -0.0012 | 0.1360 | 1.0089 | 0.9674 / 0.9686 | 0.8519 |
| 4 | 0.9622 | 0.8288 | 0.9628 | 0.1334 | -0.0006 | 0.1340 | 1.0045 | 0.9622 / 0.9628 | 0.8489 |
| 5 | 0.9676 | 0.8328 | 0.9668 | 0.1348 | 0.0008 | 0.1340 | 0.9941 | 0.9676 / 0.9668 | 0.8492 |
| 6 | 0.9558 | 0.8202 | 0.9580 | 0.1356 | -0.0022 | 0.1378 | 1.0162 | 0.9558 / 0.9580 | 0.8520 |
| 7 | 0.9642 | 0.8256 | 0.9652 | 0.1386 | -0.0010 | 0.1396 | 1.0072 | 0.9642 / 0.9652 | 0.8516 |

Every replicate individually passes both pre-registered thresholds.
`delta_exact_control` was exactly zero for all 4,000 queries.

## Across-replicate aggregate

Standard deviation below is the sample standard deviation across eight
replicate-level means.

| Metric | Mean across seeds | Std. dev. | Minimum | Maximum |
| --- | ---: | ---: | ---: | ---: |
| Exact native recall | 0.963700 | 0.004279 | 0.9558 | 0.9676 |
| PQ native recall | 0.826075 | 0.004824 | 0.8202 | 0.8328 |
| PQ exact-rerank recall | 0.963075 | 0.003957 | 0.9572 | 0.9686 |
| Mean total delta | 0.137625 | 0.004056 | 0.1334 | 0.1444 |
| Mean discovery delta | 0.000625 | 0.002251 | -0.0022 | 0.0042 |
| Mean ranking delta | 0.137000 | 0.002894 | 0.1340 | 0.1416 |
| Rerank recovery | 0.995752 | 0.016055 | 0.9706 | 1.0162 |
| Exact candidate coverage | 0.963700 | 0.004279 | 0.9558 | 0.9676 |
| PQ candidate coverage | 0.963075 | 0.003957 | 0.9572 | 0.9686 |
| Evaluated-set Jaccard | 0.850600 | 0.001295 | 0.8489 | 0.8520 |

## Criterion result and interpretation

**The pre-registered robustness criterion passes: 8 of 8 replicates pass, more
than the required 7 of 8.**

Across these new IID-Gaussian seeds, mean discovery delta remains close to zero
and exact reranking recovers 97.1%–101.6% of the native gap. Mean ranking delta
is 0.1340–0.1416 and accounts for almost all mean total delta under the
evaluated-set definition.

This supports seed stability of the earlier observation for this one data
family and fixed operating point. It does not establish generality beyond IID
isotropic Gaussian data, prove a hard-query population, or isolate a distinct
terminal sorting phase. As before, “ranking” includes PQ-based selection and
retention after a node's distance has been evaluated.

## Outputs and validation

- Root manifest and pre-registered config:
  `runs/phase1_seed_robustness_v1/manifest.json` and `resolved_config.conf`
- Raw per-query data: 500 rows in each
  `runs/phase1_seed_robustness_v1/replicate_XX/queries.jsonl`
- Replicate table: `results/tables/phase1_seed_robustness.csv`
- Full aggregate: `results/tables/phase1_seed_robustness.json`
- Requested figures only:
  `results/figures/phase1_seed_robustness_delta_boxplots.svg` and
  `phase1_seed_robustness_recovery_boxplot.svg`

All 4,000 rows were independently checked for candidate-set uniqueness,
coverage/oracle equality, metric definitions, decomposition identity, query
order, and set-overlap calculations. The boxplot whiskers show observed minima
and maxima; boxes use inclusive quartiles.

Because the robustness criterion passed, an unexecuted Phase 2 comparison plan
is recorded in `docs/phase2_distribution_plan.md`.
