# Phase 3B: fixed-candidate ranking on SIFT1M

## Pre-registration (written before Phase 3B scores or analyses)

H1: Native PQ output equals global PQ top-k over V_L0_pq for essentially
all non-tied queries.

H2: Harmful PQ errors are much more strongly associated with small top-k
boundary margins than with global average PQ distance error.

H3: Cross-boundary critical inversions are much more predictive of ranking
recall loss than random pairwise inversion rate.

H4: Increasing efSearch increases useful candidate coverage but does not solve
the fixed-candidate PQ ranking problem, consistent with Phase 3A.

These qualitative hypotheses have no post hoc significance thresholds.
Report signed Spearman coefficients and absolute differences; do not turn
association into causality or call an algebraically related diagnostic an
independent predictor. H2 and H3 can fail.

Use the existing SIFT1M files, serialized FP32 graph, and PQ64x8 model/codes.
Do not build or train any index. Primary efSearch=64, followed by efSearch=128
only after primary analysis is complete. All 10,000 queries, k=10, unchanged
query order, corrected L0 semantics and Phase 3A search parameters.
Configuration: `configs/indexes/phase3b_fixed_candidate_ranking.conf`.

## Measurement conventions fixed before results

- Distances and margins in this phase are **squared L2** (Phase 3A geometric
  descriptors used L2). Errors use double-precision subtraction of stored
  FP32 distances. ADC is FAISS's existing distance computer.
- Store every candidate's ID, exact distance and PQ score in first unique
  L0 evaluation order. These are candidate sets with a tie-breaking order,
  not full search trajectories. No queue/edge/pop logging is added.
- Exact and global-PQ top-k are independent stable sorts of these scores;
  exact ties use first evaluation order, matching the Phase 3A PQ oracle.
  Do not substitute native IDs into either computed top-k.
- Report literal set/list equality and equality modulo exactly equal PQ
  scores separately. A native output is tie-equivalent only if its ordered
  PQ score vector equals the globally sorted first k scores, every ID is in
  V, and all exchanged IDs have the exact boundary score. Any non-tied
  discrepancy blocks mechanism interpretation pending investigation.
- Stable/Benign/Harmful/Lucky are ID-membership classes, using integer GT-hit
  differences for recall-loss signs. Exact-boundary-tied cases remain in the
  primary analysis; also report an exclusion sensitivity analysis.
- Boundary window: inclusive exact ranks 5 through 15 (k-5 through k+5).
  Error scale is mean absolute error in this window. Ratios with zero
  boundary margin are null and form a separate bin, never silently clipped.
  Relative margin divides by max(dk, 1e-12). Gaps r=1,2,5 follow
  d_(k+r)-d_(k-r+1).
- Count strict cross-boundary inversions over all 10*(|V|-10) pairs; retain
  equal exact-distance pairs in the denominator but not in the numerator.
  Record every displaced/intruder pair for harmful queries, including ties
  and signed violations. Maximum violation is over all cross-boundary pairs,
  not only pairs selected after observing harm.
- Random pairwise inversion uses 512 uniformly sampled distinct candidate
  pairs per query, with replacement across draws and deterministic per-query
  seeds. This controls candidate-pool locality rather than comparing local
  inversions only to random whole-database pairs. Retain pair indices and
  counts; report ties. Per-candidate errors cover the entire V.
- Quantile bins use common-value cut points (ties are never split); report
  count, conditional mean and standard error, and zero-margin bins separately.
  Threshold curves report selected-query count, harmful-query fraction and
  mean loss. These are descriptive, with no trained model or tuned threshold.
- Compare candidate replay/native IDs, exact oracle IDs, counts and retained
  candidate samples against Phase 3A for every applicable query. Hash graph,
  codebooks/codes before and after. Raw scores must regenerate analysis without
  FAISS or the original database.

## Primary checkpoint (completed before running efSearch=128)

All 10,000 ef64 queries were measured, covering 10,343,205 unique-within-query
L0 candidates. Native/global PQ top-k differed on 74 ID sets and 697 ordered
lists; every discrepancy was an exact PQ-score tie. An independent Python
result-heap reference reproduced all 10,000 native lists exactly. H1 therefore
holds for every non-tied query, with the explicit tie conventions below.

Mean exact-candidate oracle recall=0.95200; global PQ recall=0.85246;
native PQ recall=0.85245; fixed-candidate ranking recall loss=0.09954.
Stable/Benign/Harmful/Lucky counts were 1,933/1,052/7,015/0.
Mean top-k disagreement was 0.12018.

Spearman correlations with ranking recall loss were: whole-candidate MAE
-0.06997, relative boundary margin -0.19697, boundary-local MAE 0.07263,
local MAE / boundary margin 0.28505, cross-boundary inversion count 0.69977,
cross-boundary inversion fraction 0.71660, maximum violation 0.55141, random
candidate-pair inversion rate 0.06168, and top-k agreement -0.81861.
These observations qualify H2: local margin is more informative than mean
distance error, but the adjacent gap alone has only a modest association.
H3 is descriptively supported; the inversion diagnostics are post-scoring
quantities directly related to selection and are not independent predictors.

The primary tables are preserved under the run's `primary_analysis/` before
the nearby replication. No settings are changed for efSearch=128.

## Frozen inputs and provenance

SIFT1M: 1,000,000 database vectors, 10,000 queries, 128 dimensions, no
normalization or preprocessing changes. Dataset paths, three SHA-256 checksums,
source revision, construction parameters and seeds are copied into each
export's `phase3a_config.conf`. The Phase 3A 100-query exhaustive-GT validation
is reused; every query's provided GT IDs are also checked against that run.

The graph was originally built with M=16, efConstruction=80 and seed 20260907;
PQ64x8 used seed 30260907. Phase 3B loads the serialized indexes from
`/rwproject/kdd-db/kluaq/dataset/sift1m/phase3a_cache_v1/`, with no graph building,
PQ training, encoding or graph modification. Both exports verify:

- Graph fingerprint:
  `1b3c18f6e2dd3ccec56e74eed30458f60a65f01234bf51478af25d84b5c02d16`.
- PQ codebooks:
  `e91299a14d59fb0b9349ee56cd24ded460889647b05f68e9a86a14a8348feef2`.
- PQ codes:
  `ed68be5be934d904f3debea4cd3f3d3fbbdc5fe87823efba968090dc828458ec`.

Code HEAD: `9a42b60540f90a90aaacacc71232f13569e9d8b1`, with Phase 3B changes
uncommitted during execution. FAISS: `20f14b31a6d54e243a3d1de6ae193fc4c3ec18ed`.
The export manifests include binary/source hashes and machine information:
rwcpu8.cse.ust.hk, Linux 5.14.0-687.24.1.el9_8.x86_64, GCC 11.5.0.
Search is single-threaded, bounded_queue=true, check_relative_distance=true,
k=10. NumPy 1.23.5 performs reference score analysis; Matplotlib 3.9.4 renders
figures. Random pair sampling uses PCG64(seed=60300001+query_id), 512 distinct-
item pairs per query, with repeated draws permitted.

## What native selection means

| Check, out of 10,000 queries | ef64 | ef128 |
|---|---:|---:|
| Unordered native/global PQ sets differ | 74 (0.74%) | 75 (0.75%) |
| Ordered lists differ literally | 697 (6.97%) | 677 (6.77%) |
| Ordered lists differ modulo exact PQ ties | 0 | 0 |
| Independent result-heap reference differs from native | 0 | 0 |
| Exact candidate boundary is tied | 152 | 145 |
| PQ candidate boundary is tied | 157 | 164 |

Source inspection explains these cases. In FAISS
`faiss/impl/HNSW.cpp::search_from_candidates_fixVT` (initial candidate handling
and `add_to_heap`), every L0 evaluated candidate is considered by the result
handler before the separate navigation candidate-heap insertion. With no ID
selector, an evaluated item is not omitted from the result competition because
the navigation heap is bounded. Both scalar and batch-4 paths call this handler.

`faiss/impl/ResultHandler.h::HeapResultHandler::add_result` admits only a
strictly smaller distance than the current worst score. Heap replacement and
reordering use `CMax::cmp2` in `faiss/utils/ordered_key_value.h`, which compares
IDs when scores tie. Thus arrival order affects admission, and ID affects
which already-admitted tied item is evicted. This differs from stable sorting
by (score, first evaluation order).

`scripts/audit_phase3b_native_ties.py` independently implements only this
result-heap rule with Python heapq and reproduces all 20,000 native lists,
using the saved candidate scores alone. It records every differing ID set
and the identical PQ scores of exchanged items. The native-minus-global mean
recall difference attributable to these tie choices is -0.00001 at ef64 and
-0.00002 at ef128. Native IDs were never substituted into T_pq_global.

**The Phase 1–3 ranking/selection degradation reduces to fixed-candidate PQ
score/ranking error in this configuration, modulo the documented tied-score
ID choices.** This is a statement about this FAISS path and operating regime,
not a proof about every HNSW implementation or search option.

## Disagreement versus retrieval harm

| Metric | ef64 primary | ef128 replication |
|---|---:|---:|
| Mean candidate count | 1034.3205 | 1852.0015 |
| Exact candidate oracle recall | 0.95200 | 0.98287 |
| Global PQ top-10 recall | 0.85246 | 0.86869 |
| Native PQ recall | 0.85245 | 0.86867 |
| Mean ranking recall loss | 0.09954 | 0.11418 |
| Mean top-k disagreement | 0.12018 | 0.12259 |
| Stable | 19.33% | 18.47% |
| Benign disagreement | 10.52% | 3.89% |
| Harmful disagreement | 70.15% | 77.64% |
| Lucky disagreement | 0% | 0% |

Of queries with changed membership, 13.04% at ef64 and 4.77% at ef128 have no
recall consequence. All but 9/10 of the benign cases, respectively, exchange
only non-GT candidates. Those 9/10 cases exchange equal numbers of GT IDs
at exact boundary ties. Thus counting arbitrary inversions or top-k changes
alone overstates retrieval harm. No Lucky case was discarded; none occurred.

Harmful queries lose 9,965 GT memberships and gain 11 at ef64, yielding the
net 9,954 lost hits. At ef128 the corresponding numbers are 11,432 and 14,
yielding 11,418. Raw displaced/intruder ID pairs retain GT-membership flags.

## Boundary geometry, errors, and critical inversions

Spearman correlation with per-query ranking recall loss:

| Feature | ef64 | ef128 |
|---|---:|---:|
| Mean absolute error over all candidates | -0.0700 | -0.0550 |
| Relative exact top-10 boundary margin | -0.1970 | -0.2579 |
| Boundary-window mean absolute error | 0.0726 | 0.1179 |
| Boundary-window MAE / boundary margin | 0.2851 | 0.3287 |
| Cross-boundary inversion count | 0.6998 | 0.7981 |
| Cross-boundary inversion fraction | 0.7166 | 0.7959 |
| Maximum cross-boundary violation | 0.5514 | 0.6348 |
| Random candidate-pair inversion rate | 0.0617 | 0.0653 |
| Exact/PQ top-k agreement | -0.8186 | -0.9272 |
| Candidate-set size | -0.1049 | -0.0124 |

Global candidate MAE averages 1319.55/1349.15 squared-L2 units; boundary-local
MAE averages 990.22/989.94. Their magnitude alone poorly distinguishes harmful
queries. Signed differential errors and which items compete across the
selection boundary are more informative than average absolute errors.

At ef64, the smallest relative-margin decile has mean loss 0.1139 versus
0.0539 in the largest-margin decile. Finite local-error/margin ratios have
mean loss 0.04081 in the lowest decile versus 0.12792 in the highest. The
152 zero-margin queries form a separate ratio bin, with mean loss 0.08487.
These are not evidence that zero or tiny margins are sufficient to cause
large recall loss.

Only 3,349/7,015 harmful queries at ef64 and 3,737/7,764 at ef128 satisfy the
adjacent k versus k+1 overtake condition. More than half of harmful queries
therefore require looking beyond this one adjacent pair. H2 has qualified
support in its comparison with global MAE, but a strong claim that a single
small boundary margin primarily explains harm is unsupported.

Mean strict cross-boundary inversion counts are 5.5819/5.8148. The 19,903/21,790
actual displaced/intruder pairs in harmful queries all have positive violation;
one pair at each ef has an exact-distance tie and is correctly excluded from
the strict inversion count. Median exact margins are 1160 squared-L2 units at
both efs; median signed error differences are 2557.86/2575.80 and median
violations are 1131.18/1141.80. The error differences outweigh the exact
distance advantage of the displaced items.

H3 is descriptively supported: critical inversion count/fraction has much
stronger association with loss than sampled random inversions. This comparison
is not causal or a predictive-model evaluation. In particular,
`violation = error_difference - exact_margin = d_pq(displaced)-d_pq(intruder)`
is an algebraic identity, and positive violation for a selected intruder is
not independent evidence. Its utility is identifying which exact margins
were overcome and whether GT items were lost. The random-pair estimate uses
only 512 draws per query and has sampling noise, which can attenuate its
correlation. Strong critical-inversion association is also partly structural
because it is measured after scoring the very items used to define the loss.

## Nearby replication and hypothesis assessment

H1 holds modulo exact ties, confirmed by the independent heap reference.
H2 is supported only in the limited sense of stronger association than global
MAE; adjacent margin and local error magnitude alone are incomplete summaries.
H3 holds descriptively with the limitations above. H4 holds: the larger pool
improves oracle recall by 0.03087, global PQ recall by only 0.01623, and leaves
top-k disagreement almost unchanged. Mean loss grows by 0.01464.

Boundary-local MAE and random inversion rates are nearly unchanged while
the harmful fraction rises and the benign fraction falls. This is consistent
with better candidate coverage exposing more GT items to the same selection
noise. It does not by itself prove a causal explanation for between-query
correlations. There is one dataset, one graph and one PQ model; the nearby ef
comparison is paired robustness, not an independent replicate.

## Correctness, artifacts, and reproducibility

- Every final native list, exact candidate-oracle list, candidate count and
  supplied GT list matches Phase 3A query-for-query. The first 100 candidate
  sets at each ef also match the sets previously preserved by Phase 3A.
- Existing L0 instrumentation is unchanged. Its native ID/distance replay
  check passes, as do pre/post graph and PQ codebook/code hashes.
- Recomputed scalar PQ distances match returned native scores and batch-4
  ADC scores for every complete batch across all saved candidates.
- Excluding exact-boundary-tied queries changes mean loss from 0.09954 to
  0.09977 at ef64 and 0.11418 to 0.11441 at ef128; sensitivity correlations
  are included in the tables. No tie case is silently removed from the main
  analysis, and no discovery metric is redefined.
- Five existing CTest tests and six Python reference tests pass. New tests
  cover classification, strict inversions, signed pair identities, tie
  equivalence versus literal equality, native heap tie rules, boundary-window
  indexing, random-pair reproducibility and tie-preserving quantile bins.
- A NumPy Boolean serialization error stopped the first analysis at query 0.
  That one-row partial analysis is retained under
  `analysis/failed_ef64_numpy_bool/` and excluded. Raw candidate exports were
  complete and unaffected. The build emitted the pre-existing host clock-skew
  warnings; the modified target compiled and linked successfully.
- All candidates and scores, all per-query metrics, all harmful replacement
  pairs, and sampled random-pair indices are retained. Binary candidate chunks
  have 500 queries each; schema and offsets are in each export manifest and
  `queries.jsonl`. No original dataset access is needed for score analysis.
- Tables and figures are regenerated by the analysis script. An independent
  re-derivation from raw scores reproduces per-query rows, harmful pairs and
  random-pair indices byte-for-byte; final tables and figures are also checked
  byte-for-byte. Exact commands are in `docs/experiment_log.md`.

## Answers and one next experiment

1. Native output is the PQ top-k over V_L0 up to explicitly audited exact
   score ties; no additional non-tied HNSW selection loss was found.
2. Membership changes affect 80.67%/81.53% of queries, but 10.52%/3.89% of
   all queries have benign changes. Recall consequence must be measured
   separately from membership disagreement.
3. Retrieval-critical noise displaces GT members across the exact selection
   boundary. Most harmless changes replace non-GT filler candidates;
   exact-boundary ties explain the small GT-for-GT exchange exception.
4. Local competition matters, but the adjacent boundary margin alone has
   only modest explanatory association. More than half of harmful queries
   lack a k-to-k+1 inversion.
5. Candidate-wide average error has very weak association with harm here.
6. Cross-boundary inversions provide a much cleaner descriptive account,
   subject to their post-scoring, partly algebraic relationship to selection.
7. ef128 preserves the mechanism and its association ordering despite a
   substantially larger candidate pool.
8. The single next experiment should hold these ef64 candidate sets fixed and
   compare scores from the **already-trained Phase 3A PQ32x8 and PQ64x8 models**.
   Reuse identical exact margins and GT, and measure within-query changes in
   critical inversions and recall loss. This isolates a change in score
   approximation strength from candidate-pool composition and tests whether
   the current descriptive association tracks a controlled scoring change.
   It requires no new quantizer, graph, or trajectory instrumentation and has
   not been run in Phase 3B.
