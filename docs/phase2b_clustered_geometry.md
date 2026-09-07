# Phase 2B: controlled clustered database geometry

## Objective and pre-registration

This experiment tests whether balanced clustered geometry changes the
mechanism of PQ damage from selection/ranking among evaluated nodes to failure
to discover useful level-0 candidates. Queries and database vectors are IID
from the same distribution in every condition.

Before execution, the configuration fixed four seed tuples, `K=16`, rho
`{0,.10,.25,.50,.75,.90}`, PQ32x8, efSearch 256, and the following exploratory
criteria:

- no discovery breakdown: every condition with exact Recall@10 at least 0.90
  has `abs(mean delta_discovery) <= 0.01` and rerank recovery at least 0.90;
- emerging geometry-induced degradation: at least one rho greater than zero
  has mean discovery delta at least 0.03 and recovery at most 0.80 in at least
  three of four replicates while exact recall is high or restored;
- strong geometry effect additionally requires a systematic rho dependence
  that is not explained solely by ordinary PQ distance error.

The thresholds were not changed after observing results. They are not
statistical-significance claims.

## Candidate-set correction gate

Phase 2B uses the corrected L0-only candidate definition documented in
`docs/instrumentation_candidate_semantics.md`. Both known Phase 2A all-level
exact-control anomalies were regenerated: each changed from exactly +0.1
under the legacy definition to zero under L0 semantics. Native IDs/distances
and graph fingerprints were unchanged.

Complete 500-query regressions produced mean L0 discovery delta 0.0042 and
rerank recovery 0.970629 for IID, and approximately zero and 1.0 for mean
alpha=1. The correction did not materially change previous aggregate
conclusions, so the predeclared continuation gate allowed Phase 2B to run.

## Dataset and search method

For every replicate, 16 seeded random unit directions were fixed. Database
and query labels were independently uniform, and vectors were generated as

```
x = sqrt(1-rho) z + sqrt(rho * 64) c_j,  z ~ N(0,I_64).
```

Latent vectors, labels, and centers were paired across rho within a replicate.
Each replicate/rho nevertheless received its own FP32-built HNSW graph and PQ
model. There were 20,000 database vectors, 500 queries, dimension 64, HNSW
`M=16`, `efConstruction=80`, `k=10`, bounded queues, relative-distance
checking, and one search thread. PQ32x8 was trained on and encoded the entire
condition-specific database. Ground truth was exhaustive FP32 squared L2.

The empirical mean squared query norm ranged from 63.73 to 64.07 across rho,
consistent with the intended approximately fixed second-moment scale.

For each condition, PQ quality used the Phase 1 sampling method: the first 100
queries, 100 reproducibly sampled database IDs per query, and 100 reproducible
candidate pairs per query. Relative error is absolute error divided by exact
distance; inversion means a strict order reversal, with ties reported
separately.

The binary recorded Git HEAD
`cbc944c773ac7d38d1f514d66aa46d87341e3cc9` with the Phase 2B changes
uncommitted, FAISS commit
`20f14b31a6d54e243a3d1de6ae193fc4c3ec18ed`, and machine
`rwcpu8.cse.ust.hk` (Linux 5.14.0-687.24.1.el9_8.x86_64, GCC 11.5.0). Exact
seeds and hashes are preserved in the resolved configuration and 24 condition
manifests.

## Primary observations

Values are mean +/- sample standard deviation across four replicate-level
means.

| rho | Exact native | PQ native | PQ L0 oracle | Total delta | Discovery delta | Ranking delta | Recovery | L0 Jaccard |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | 0.9649 +/- .0037 | 0.8253 +/- .0049 | 0.9629 +/- .0047 | 0.1396 +/- .0048 | 0.0020 +/- .0023 | 0.1376 +/- .0032 | 0.9860 +/- .0163 | 0.8511 +/- .0014 |
| 0.10 | 0.9729 +/- .0022 | 0.8283 +/- .0021 | 0.9731 +/- .0021 | 0.1447 +/- .0037 | -0.0003 +/- .0008 | 0.1449 +/- .0034 | 1.0018 +/- .0056 | 0.8535 +/- .0010 |
| 0.25 | 0.9956 +/- .0005 | 0.8421 +/- .0020 | 0.9957 +/- .0007 | 0.1535 +/- .0019 | -0.0002 +/- .0006 | 0.1537 +/- .0024 | 1.0009 +/- .0039 | 0.8827 +/- .0025 |
| 0.50 | 0.9998 +/- .0002 | 0.8290 +/- .0035 | 0.9998 +/- .0002 | 0.1707 +/- .0036 | approximately 0 +/- .0002 | 0.1708 +/- .0036 | 1.0003 +/- .0011 | 0.9713 +/- .0025 |
| 0.75 | 0.9997 +/- .0003 | 0.7801 +/- .0081 | 0.9997 +/- .0003 | 0.2196 +/- .0083 | 0.0000 +/- .0000 | 0.2196 +/- .0083 | 1.0000 +/- .0000 | 0.9698 +/- .0008 |
| 0.90 | 0.9997 +/- .0003 | 0.7117 +/- .0049 | 0.9997 +/- .0002 | 0.2881 +/- .0051 | 0.0000 +/- .0002 | 0.2881 +/- .0050 | 1.0000 +/- .0006 | 0.9635 +/- .0010 |

Clustering increased total degradation, but none of that increase survived
exact reranking of the PQ L0 evaluated set. Mean discovery delta remained
between -0.0003 and 0.0020; ranking delta grew from 0.1376 to 0.2881. At rho
0.75 and 0.90, 99.9% of queries had exactly zero discovery delta.

Exact recall was at least 0.9594 in every individual replicate/condition and
approached one as clusters strengthened. No matched-ef search was triggered.
All 24 replicate/condition rows passed the pre-registered no-breakdown
criterion. No rho met the emerging or strong criterion.

Level-0 evaluation counts fell from approximately 4,214 at rho zero to 1,166
at rho 0.90 for both exact and PQ traversal. Their counts remained closely
matched. L0 Jaccard increased rather than decreased, reaching roughly 0.96--
0.97 for rho at least 0.5.

## PQ approximation-quality control

| rho | Mean / median / p95 absolute error | Mean / median absolute relative error | Strict inversion rate |
| ---: | --- | --- | ---: |
| 0.00 | 2.1316 / 1.6843 / 5.6029 | .01654 / .01348 | .04247 |
| 0.10 | 2.1284 / 1.6795 / 5.6418 | .01660 / .01348 | .04345 |
| 0.25 | 2.0756 / 1.6412 / 5.4372 | .01632 / .01331 | .03900 |
| 0.50 | 1.9394 / 1.5461 / 5.0388 | .01563 / .01282 | .03840 |
| 0.75 | 1.6701 / 1.3369 / 4.3396 | .01405 / .01144 | .03500 |
| 0.90 | 1.2939 / 1.0458 / 3.3363 | .01185 / .00921 | .03200 |

These are across-replicate means; complete uncertainties are in the summary
tables. Ordinary randomly sampled pair error and inversion rate did not worsen
with clustering: after a small rho=0.10 increase, they declined. Across the six
rho-level means, Spearman correlation between discovery delta and inversion
rate was -0.406, based on only six points and not interpreted causally.

The simultaneous increase in PQ-native ranking loss is therefore not
explained by worse error on these globally sampled pairs. A plausible
measurement limitation is that increasingly separated mixtures make random
database pairs predominantly cross-cluster, whereas top-10 selection depends
on fine within-cluster ordering. This is a confounder for explaining the
ranking trend, but it does not create evidence of discovery loss because the
PQ L0 candidate oracle remains essentially exact.

## Cluster-boundary and margin observations

The fraction of queries whose ground-truth top-10 spans multiple generating
components fell from 1.000 at rho zero to 0.997, 0.464, 0.0005, 0, and 0 as rho
increased. At rho zero, component labels are intentionally geometrically
meaningless.

Neither the lowest cluster-center-margin quartile nor the lowest exact-NN-
margin quartile showed positive discovery degradation of meaningful size.
Their mean deltas differed from the remaining queries by at most a few
thousandths and without a consistent sign. Per-replicate Spearman correlations
were small and inconsistent; several high-rho correlations are undefined
because virtually every discovery delta is exactly zero. These observations
do not support disproportionate L0 discovery failure at cluster boundaries.

## Correctness controls and limitations

- `delta_exact_control` was exactly zero for all 12,000 fixed-ef queries.
- All 24 graph fingerprints were distinct, and each graph/PQ hash remained
  unchanged through its paired run.
- Instrumented phase-separated native IDs and distances matched ordinary
  FAISS search exactly; sampled upper-only sets were disjoint from `V_L0`.
- Rho zero exactly reproduced Phase 1 replicates 0--3 for database/query
  generation, ground truth, exact/PQ native IDs, and native recall.
- All raw distance errors, order classifications, recalls, coverage values,
  deltas, and sampled set overlaps were independently recomputed by the
  analysis script.
- Candidate ID sets are retained for query IDs 0--24 per condition; every
  query retains scalar measurements and ordered ground-truth/native/oracle
  results. Raw distance/order samples are also preserved.
- The family is controlled synthetic geometry. Balanced spherical mixtures
  are not evidence about all structured or real embedding distributions.

## Kill / continue decision

1. **Balanced clustered geometry did not create reproducible L0 discovery
   degradation.** Discovery remained approximately zero in all replicates.
2. No matched control was required because exact recall always exceeded 0.95.
3. There is no discovery effect to explain through PQ error. Ordinary PQ error
   and inversion rate decreased while ranking loss increased.
4. Rerank recovery did not decline with rho; it converged to approximately
   one.
5. Cluster-boundary and low-NN-margin queries were not disproportionately
   affected in L0 discovery under the measured descriptors.
6. The evidence does not justify full trajectory instrumentation because no
   reproducible discovery failure exists to localize.
7. The working hypothesis that **ordinary PQ frequently damages graph
   navigability under realistic geometric variation has now received multiple
   negative controlled tests**: IID Gaussian, query-only mean/radial shift,
   and balanced clustered geometry. This does not falsify it for all data, but
   it materially weakens candidate discovery as the main synthetic-data story.

## Exactly one next experiment

Move directly to **one SIFT1M fixed-topology L0 decomposition experiment** at
a calibrated high-exact-recall operating point, with PQ32x8 and the same
exact-rerank oracle and PQ-quality controls. This is one real-data test and has
higher information gain than another synthetic anisotropy sweep. Do not add
full trajectories unless SIFT1M produces a reproducible positive discovery
signal after exact-recall matching.

## Outputs

- Raw rows, quality samples, hashes, and resolved config:
  `runs/phase2b_clustered_geometry_v1/`
- Summary and exploratory-correlation tables: `results/tables/phase2b_*`
- Eight required figures: `results/figures/phase2b_*`
