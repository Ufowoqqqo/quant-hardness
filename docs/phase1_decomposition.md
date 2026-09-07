# Phase 1 candidate-discovery versus approximate-ranking decomposition

## Scope

This experiment decomposes the calibrated PQ32×8 end-to-end Recall@10 gap at
`efSearch` 160, 256, and 384. It reuses the same generated database and query
vectors, exhaustive ground truth, FP32 construction parameters, PQ training
seed and data, query order, and HNSW structural fingerprint as the calibration.
No graph-construction change, new quantizer, optimization, full trajectory
logging, or SIFT1M run was introduced.

The result identifies mechanisms under these definitions and this synthetic
setting. Different per-query values alone are not evidence of a hard-query
phenomenon.

## Fixed configuration and identity controls

- Database: 20,000 independent standard-normal FP32 vectors, dimension 64,
  seed 1729.
- Queries: 500 separately generated vectors from the same distribution, seed
  2718, executed in ascending query-ID order.
- HNSW: one FP32 squared-L2 construction, `M=16`, `efConstruction=80`, seed
  31415.
- Search: `k=10`, `efSearch` 160/256/384, bounded queue and relative-distance
  check enabled, one thread.
- PQ: FAISS `IndexPQ`, 32 subquantizers × 8 bits, trained on all 20,000
  database vectors with seed 16180; 32-byte codes and FP32 queries with ADC.
- Ground truth: exhaustive FP32 squared-L2 top-10.
- FAISS: `v1.15.0`, commit
  `20f14b31a6d54e243a3d1de6ae193fc4c3ec18ed`.
- Graph fingerprint:
  `5eeca90445dd29413ba0ba7bceb4be18258ef5580174050fbaf087e2f57791e7`.

The runner requires the fingerprint to equal the calibration value and checks
it after PQ preparation and after every instrumented exact/PQ search. The
analysis compared all 1,500 exact and PQ native ordered result-ID lists with
`runs/phase1_calibration_v1/paired_queries.jsonl`; every list matched.

## Instrumentation location and semantics

Instrumentation is repository-owned in
`src/instrumentation/faiss_distance_recorder.cpp`; upstream FAISS is unchanged.
`RecordingIndex` returns a `RecordingDistanceComputer` which delegates to the
original `IndexFlatL2` or `IndexPQ` distance computer.

FAISS `IndexHNSW.cpp::hnsw_search()` obtains this distance computer before its
query loop, calls `set_query()`, then passes it to `HNSW::search()`. The wrapper
records each database ID passed to either:

- `DistanceComputer::operator()(id)`; or
- `DistanceComputer::distances_batch_4(id0, id1, id2, id3, ...)`.

The underlying scalar or batch-4 implementation is then called unchanged. A
record therefore means that FAISS requested a query-to-node distance during
upper-level or base-level HNSW search. `symmetric_dis`, which is a
database-to-database distance, is delegated but not recorded.

The raw output stores the unique IDs sorted by database ID. Evaluation order,
queue insertion, queue popping, expansion order, and visited-table events are
not recorded. Consequently this is a candidate-set measurement, not trajectory
logging. Reported evaluation counts are counts of unique distance-evaluated
nodes, not total distance-computer calls; duplicate evaluations across levels
are collapsed.

Unit controls confirm that instrumentation leaves ordered native IDs and
distances unchanged for both FP32 and PQ distance computers. They also check
set uniqueness, graph identity, exact reranking against hand-computed
distances, candidate coverage, and oracle/coverage equality.

## Definitions and interpretation boundary

For each query, all distance-evaluated nodes are exact-reranked with FP32 L2.
Every one of the 1,500 rows satisfies:

```text
recall_exact_oracle == coverage_exact
recall_pq_oracle    == coverage_pq
delta_total = delta_discovery + delta_ranking - delta_exact_control
```

`delta_exact_control` is exactly zero for every query.

Under the requested definition, `delta_ranking` covers failure to select or
retain a useful node after its PQ distance has already been evaluated. FAISS
does not perform a distinct exact-sized “final ranking” pass, so this quantity
includes PQ-score-based candidate/result selection throughout search after
evaluation. It should not be described as only a terminal sorting error.

## Summary results

| Measurement | `ef=160` | `ef=256` | `ef=384` |
| --- | ---: | ---: | ---: |
| Mean exact native recall | 0.9206 | 0.9572 | 0.9784 |
| Mean PQ native recall | 0.8070 | 0.8222 | 0.8328 |
| Mean exact oracle/coverage | 0.9206 | 0.9572 | 0.9784 |
| Mean PQ oracle/coverage | 0.9214 | 0.9572 | 0.9792 |
| Mean unique exact evaluations | 3036.01 | 4260.95 | 5596.96 |
| Mean unique PQ evaluations | 3032.69 | 4262.54 | 5597.67 |
| Mean Jaccard(`V_E`, `V_P`) | 0.8224 | 0.8493 | 0.8710 |
| Rerank recovery | 1.0070 | 1.0000 | 1.0055 |

Recovery slightly above one at `ef=160` and 384 is not a control failure: PQ
candidate coverage is higher on average by 0.0008 at those points. It means
exact reranking of the PQ-evaluated set marginally exceeds exact native recall
for this finite query set.

### Delta distributions

| `efSearch`, quantity | Mean | Median | p90 | p95 | p99 | Fraction = 0 | Fraction > 0 | Fraction < 0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 160, total | 0.1136 | 0.1 | 0.2 | 0.3 | 0.4 | 0.234 | 0.738 | 0.028 |
| 160, discovery | -0.0008 | 0 | 0.1 | 0.1 | 0.1 | 0.782 | 0.108 | 0.110 |
| 160, ranking | 0.1144 | 0.1 | 0.2 | 0.3 | 0.4 | 0.246 | 0.754 | 0 |
| 256, total | 0.1350 | 0.1 | 0.3 | 0.3 | 0.4 | 0.162 | 0.830 | 0.008 |
| 256, discovery | 0 | 0 | 0 | 0.1 | 0.1 | 0.862 | 0.068 | 0.070 |
| 256, ranking | 0.1350 | 0.1 | 0.2 | 0.3 | 0.4 | 0.164 | 0.836 | 0 |
| 384, total | 0.1456 | 0.1 | 0.3 | 0.3 | 0.4 | 0.128 | 0.866 | 0.006 |
| 384, discovery | -0.0008 | 0 | 0 | 0 | 0.1 | 0.922 | 0.036 | 0.042 |
| 384, ranking | 0.1464 | 0.1 | 0.3 | 0.3 | 0.4 | 0.126 | 0.874 | 0 |

`delta_discovery` was not forced non-negative. PQ traversal produced better
candidate coverage than exact traversal for 11.0%, 7.0%, and 4.2% of queries
at increasing `efSearch`; exact traversal was better for 10.8%, 6.8%, and 3.6%.
Coverage was equal for the remaining 78.2%, 86.2%, and 92.2%.

## Answers to the key questions

### 1. Which component dominates?

Under the experiment's evaluated-set definition, approximate ranking/selection
dominates. Mean discovery delta is -0.0008, 0, and -0.0008, whereas mean ranking
delta is 0.1144, 0.1350, and 0.1464. Exact reranking recovers approximately
100% of the native gap at every operating point. PQ traversal therefore did
evaluate enough ground-truth neighbors on average; PQ scores failed to promote
them into native top-10 results.

### 2. What changes with increasing `efSearch`?

PQ candidate coverage improves from 0.9214 to 0.9572 to 0.9792. Evaluated-set
Jaccard also rises from 0.8224 to 0.8710. Exact reranking continues to recover
the complete average gap, so there is no observed saturation of PQ candidate
discovery over this range.

### 3. Why did total delta increase during calibration?

From `ef=160` to 384, exact native recall rises by 0.0578 while PQ native recall
rises by only 0.0258. PQ oracle recall rises by 0.0578, essentially tracking
exact recall, but PQ native output fails to realize that candidate-set gain.
Thus the widening total gap is explained by PQ native ranking/selection
saturating relative to the improving evaluated candidate set, not by PQ
candidate coverage saturating.

### 4. Which component is associated with per-query heterogeneity?

Total delta correlates more strongly with ranking delta (Pearson 0.867, 0.905,
0.952) than with discovery delta (0.414, 0.316, 0.289). Discovery delta is zero
for 78.2–92.2% of queries, while ranking delta is positive for 75.4–87.4%.
Within this run, observed heterogeneity is therefore primarily associated with
PQ ranking/selection among evaluated nodes. This is not evidence by itself for
a distinct hard-query population.

### 5. Does PQ ever discover a better set?

Yes. PQ candidate coverage exceeds exact candidate coverage for 55/500,
35/500, and 21/500 queries at `efSearch` 160, 256, and 384. These negative
`delta_discovery` cases are retained in the raw output and all summaries.

## Implications and limitations

For this IID Gaussian dataset and PQ32×8, the proposed direction becomes less
promising if narrowly framed as quantization causing traversal to miss relevant
regions: this experiment observes almost no average discovery degradation. The
broader end-to-end quantization effect remains measurable, but here it is a
ranking/selection effect under the evaluated-set oracle definition.

This does not falsify traversal-induced hardness on other data distributions,
OOD queries, graph parameters, or compression strengths. Exact reranking of
all evaluated nodes uses roughly 3,000–5,600 candidates per query and is an
oracle diagnostic, not a practical reranking budget. The candidate sets reveal
membership only, not when or why a node was evaluated.

## Outputs

- Raw 1,500 per-query rows with all evaluated ID sets:
  `runs/phase1_decomposition_v1/queries.jsonl`
- Manifest and resolved configuration:
  `runs/phase1_decomposition_v1/manifest.json` and `resolved_config.conf`
- Summary tables: `results/tables/phase1_decomposition.json` and `.csv`
- Figures: seven `results/figures/phase1_decomposition_*.svg` files covering
  the requested per-ef scatter plots, oracle/native recall, set overlap, mean
  recall, and delta CDFs.

## Smallest next experiment with highest information gain

Repeat this same recorder-and-oracle decomposition at one central operating
point (`efSearch=256`, PQ32×8) for several independent database/query/graph/PQ
seed tuples. This is the smallest test of whether ranking dominance and
near-zero mean discovery delta are stable rather than an accidental property of
the single seed tuple. It changes no algorithm and should precede a larger
distribution-shift experiment.
