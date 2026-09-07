# Phase 1 synthetic operating-regime calibration

## Scope

This calibration searches for high-recall exact-HNSW operating regimes before
the traversal-versus-reranking decomposition. One FP32-built HNSW graph, one
database, one query set, one exhaustive ground truth, and one query order are
shared by every `efSearch` and PQ condition. Only query-time `efSearch` and the
attached distance storage vary.

This is parameter calibration on one synthetic distribution. It does not test
novelty, establish a heavy tail, attribute loss to traversal, or optimize an
algorithm. No trajectory logging or SIFT1M data was added.

## Previous experiment parameters

The destructive preceding run used:

| Parameter | Value |
| --- | --- |
| Database size | 20,000 |
| Query count | 500, generated separately from the database |
| Dimension | 64 |
| Database/query distribution | Independent standard-normal FP32; no preprocessing |
| Database/query seeds | 1729 / 2718 |
| HNSW `M` | 16 |
| HNSW `efConstruction` | 80 |
| HNSW construction seed | 31415 |
| `efSearch` | 16 |
| PQ subquantizers | 8 |
| PQ bits per subquantizer | 4 |
| PQ code size | 4 bytes/vector |
| PQ training set | All 20,000 database vectors |
| PQ seed | 16180 |
| Metric / result count | Squared L2 / `k=10` |

The calibration deliberately retains this exact condition as its lowest
`efSearch` and strongest-compression control. It reproduces mean exact recall
0.4772, mean PQ recall 0.0708, and mean delta 0.4064.

## Calibration design

The exact-HNSW sweep was fixed in advance as:

```text
efSearch = 16, 32, 64, 96, 128, 160, 192, 256, 384, 512
```

The graph was not rebuilt between values. The graph fingerprint remained:

```text
5eeca90445dd29413ba0ba7bceb4be18258ef5580174050fbaf087e2f57791e7
```

Four standard FAISS PQ configurations valid for dimension 64 were evaluated at
every `efSearch`:

| ID | Subquantizers | Bits/subquantizer | Code bytes/vector |
| --- | ---: | ---: | ---: |
| `pq_m8_nbits4` | 8 | 4 | 4 |
| `pq_m8_nbits8` | 8 | 8 | 8 |
| `pq_m16_nbits8` | 16 | 8 | 16 |
| `pq_m32_nbits8` | 32 | 8 | 32 |

Every PQ was trained on the same 20,000 database vectors with seed 16180. All
searches used `k=10`, bounded queues, relative-distance checking, one thread,
and query IDs 0 through 499 in that order.

For distance-only diagnostics, query IDs 0 through 99 were used. For every
query and PQ configuration, 100 database IDs and 100 database-ID pairs were
drawn from MT19937 seed 424242. This gives 10,000 distance samples and 10,000
ordering samples per PQ. Relative error means absolute error divided by exact
distance when exact distance exceeds `1e-12`. An inversion is a strict reversal
of an exact pairwise order; PQ ties are reported separately.

## Exact-HNSW `efSearch` diagnosis

| `efSearch` | Mean exact Recall@10 | Median exact Recall@10 |
| ---: | ---: | ---: |
| 16 | 0.4772 | 0.5 |
| 32 | 0.6346 | 0.6 |
| 64 | 0.7856 | 0.8 |
| 96 | 0.8590 | 0.9 |
| 128 | 0.8970 | 0.9 |
| 160 | 0.9206 | 0.9 |
| 192 | 0.9380 | 1.0 |
| 256 | 0.9572 | 1.0 |
| 384 | 0.9784 | 1.0 |
| 512 | 0.9864 | 1.0 |

Because this curve changes only `efSearch` on the identical graph and data,
the previous exact recall of 0.4772 was mainly an insufficient-`efSearch`
operating-point issue in this experiment. Increasing `efSearch` from 16 to 160,
256, and 384 raised mean exact recall to 0.9206, 0.9572, and 0.9784.

## Independent PQ distance quality

| PQ | Mean / median / p95 absolute error | Mean / median / p95 absolute relative error | Strict inversion rate | PQ tie rate |
| --- | --- | --- | ---: | ---: |
| 8×4 | 37.7057 / 36.6868 / 67.2499 | 0.2865 / 0.2922 / 0.4421 | 0.3265 | 0 |
| 8×8 | 19.8182 / 18.4856 / 42.3884 | 0.1494 / 0.1461 / 0.2890 | 0.2432 | 0 |
| 16×8 | 7.5121 / 6.1060 / 19.0706 | 0.0576 / 0.0494 / 0.1373 | 0.1317 | 0 |
| 32×8 | 2.1394 / 1.6945 / 5.5812 | 0.01665 / 0.01367 / 0.04265 | 0.0452 | 0 |

The previous 8×4 PQ condition has 28.65% mean absolute relative distance error
and reverses 32.65% of sampled strict pairwise orders. Its PQ Recall@10 stays
between 0.0664 and 0.0828 over the complete `efSearch` sweep. Relative to the
calibration goal, this condition was excessively destructive. The observation
does not imply that 8×4 PQ is invalid for every dataset or objective.

The 32×8 condition is materially less destructive: 1.665% mean absolute
relative error, 4.52% strict inversion rate, and PQ Recall@10 from 0.8070 to
0.8328 at the three selected exact-recall regimes.

## Complete paired results

The complete 40-row table—every combination of ten `efSearch` values and four
PQ configurations—is stored in:

- `results/tables/phase1_calibration.csv`
- `results/tables/phase1_calibration.json`

Each row reports mean exact and PQ Recall@10, mean/median/p90/p95/p99 delta,
fractions with zero/positive/negative delta, and the count and fraction of
positive-loss queries required for 50%, 80%, and 90% of total positive loss.
No combination was removed from the table.

At `efSearch=256`, where exact mean Recall@10 is 0.9572, the compression trend
is:

| PQ | Mean PQ recall | Mean / median delta | Fraction delta = 0 / > 0 / < 0 | Positive-query fraction for 50% / 80% / 90% loss |
| --- | ---: | ---: | --- | --- |
| 8×4 | 0.0718 | 0.8854 / 0.9 | 0 / 1 / 0 | 0.460 / 0.764 / 0.874 |
| 8×8 | 0.2422 | 0.7150 / 0.7 | 0 / 1 / 0 | 0.428 / 0.738 / 0.856 |
| 16×8 | 0.5070 | 0.4502 / 0.4 | 0.002 / 0.998 / 0 | 0.389 / 0.703 / 0.824 |
| 32×8 | 0.8222 | 0.1350 / 0.1 | 0.162 / 0.830 / 0.008 | 0.333 / 0.675 / 0.839 |

As quantization becomes less aggressive, the affected fraction falls and the
positive loss becomes relatively more concentrated: at this `efSearch`, the
positive-query fraction needed for 50% of loss falls from 46.0% to 33.3%.
The 90% concentration statistic changes less and is not monotonic across the
last two settings. Even for 32×8, 83.9% of positive-loss queries—69.6% of all
queries—are required to account for 90% of positive loss. The loss therefore
remains broadly distributed rather than confined to a small rare tail.

## Selected operating points

Selection used the requested recall and degradation criteria, without
inspecting whether a point produced a favorable concentration pattern. PQ32×8
was selected because its independent distance error and recall are neither
negligible nor catastrophic. The three `efSearch` values span the requested
exact-recall targets while satisfying exact mean Recall@10 above 0.90.

| `efSearch`, PQ | Mean exact / PQ recall | Mean / median delta | p90 / p95 / p99 delta | Fraction delta = 0 / > 0 / < 0 | Positive-query fraction for 50% / 80% / 90% loss |
| --- | --- | --- | --- | --- | --- |
| 160, 32×8 | 0.9206 / 0.8070 | 0.1136 / 0.1 | 0.2 / 0.3 / 0.4 | 0.234 / 0.738 / 0.028 | 0.331 / 0.686 / 0.843 |
| 256, 32×8 | 0.9572 / 0.8222 | 0.1350 / 0.1 | 0.3 / 0.3 / 0.4 | 0.162 / 0.830 / 0.008 | 0.333 / 0.675 / 0.839 |
| 384, 32×8 | 0.9784 / 0.8328 | 0.1456 / 0.1 | 0.3 / 0.3 / 0.4 | 0.128 / 0.866 / 0.006 | 0.335 / 0.663 / 0.831 |

All 500 paired per-query rows for every selected point, and for every other
swept point, are preserved in
`runs/phase1_calibration_v1/paired_queries.jsonl`.

## Answers to the calibration questions

**Was the previous low exact recall mainly an `efSearch` issue?** Yes, within
this fixed graph and synthetic dataset. Changing only `efSearch` raises exact
mean recall from 0.4772 at 16 to 0.9206 at 160 and 0.9864 at 512.

**Was the previous PQ configuration excessively destructive?** Yes for the
intended phenomenon-validation regime. Its independent distance error,
pairwise inversion rate, globally positive delta, and near-flat low PQ recall
across `efSearch` distinguish it from the milder PQ32×8 condition.

**Does concentration change with less aggressive quantization?** At fixed
`efSearch=256`, milder compression introduces zero- and negative-delta queries
and reduces the fraction of positive queries needed for 50% of total loss.
However, the majority of queries still have positive delta, and roughly 82–87%
of positive queries are needed for 90% of loss across the four code rates.

**Is there preliminary evidence of heterogeneous per-query sensitivity?** The
selected conditions contain negative, zero, and positive deltas, with p99 0.4
and median 0.1, so per-query response is observably heterogeneous. This is only
preliminary: 73.8–86.6% of queries are harmed at the selected points, and the
concentration statistics do not show degradation isolated to a rare minority.
No traversal-specific explanation is supported.

## Artifacts and correctness notes

- Raw exact sweep: `runs/phase1_calibration_v1/exact_queries.jsonl`
- Raw paired sweep: `runs/phase1_calibration_v1/paired_queries.jsonl`
- Raw distance samples: `runs/phase1_calibration_v1/distance_samples.jsonl`
- Raw ordering samples: `runs/phase1_calibration_v1/order_samples.jsonl`
- Run manifest/config: `runs/phase1_calibration_v1/manifest.json` and
  `resolved_config.conf`
- Derived tables: `results/tables/phase1_calibration.*`
- Figures: exact/PQ recall versus `efSearch`, recall versus code size, and
  selected-point delta CDFs under `results/figures/phase1_calibration_*.svg`

The analysis independently recomputes Recall@10 and delta from every result-ID
list, checks exact result/ground-truth identity across paired modes, verifies
all query orders and expected row counts, recomputes sampled errors and ordering
labels, and refuses to overwrite derived outputs. The C++ runner asserts the
graph fingerprint after each exact and PQ sweep.

Unresolved confounder: native PQ output still combines query-time traversal
decisions with approximate final ranking. This calibration does not attribute
delta to either component.
