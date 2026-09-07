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

## Results

Pending primary measurement and analysis.
