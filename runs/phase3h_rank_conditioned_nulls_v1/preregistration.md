# Phase 3H pre-registration — final exact-distance-only mechanism phase

Registered before producing Phase3H null outcomes. Same five Phase3G models
R0-I1,R1-I1,R2-I1,R3-I1,R4-I1; same SIFT1M identities, FP32 exact L0 candidate
pools/scores, GT, stable exact top10, graph, efSearch64,k10,PQ64x8. No graph
traversal, new training, predictor, or algorithm. Do not modify earlier runs.

H1: exact-rank conditioning reduces loss relative to unrestricted N0.
H2: N1-N3 close a substantial part of the real-versus-N0 gap because exact
near neighbors have more favorable residual distributions.
H3: medium-to-fine conditioning has diminishing additional gap closure.
H4: d12-d9 remains strongly associated with null loss under all conditions.
H5: after sufficient conditioning, d12-d9 is weakly associated with the
remaining observed-minus-null gap. These are qualitative exploratory
hypotheses, not significance tests; do not alter them after outcomes.

Bins are inclusive one-based exact ranks:
N0:1..n
N1:1..10;11..n
N2:1..5;6..10;11..20;21..n
N3:1..5;6..10;11..15;16..20;21..40;41..n
N4:1..5;6..8;9..10;11..12;13..15;16..20;21..40;41..n
N_distance: four equal-count groups of the exact-distance stable ordering,
via array_split(order,4). First n%4 groups have one extra member. Exact
distance ties are split by historical candidate order; no adaptive edges.
Empty/invalid tail intervals are intersected with1..n and discarded; only
the last nonempty interval is truncated. No nonempty interior bins merged.
Singleton bins stay fixed. All residuals remain assigned to precisely one bin.

IMPORTANT LIMITATION: equal-count exact-distance quantiles are also ordinal
rank quartiles. This condition tests coarse bulk versus top-k-local
conditioning, not an identification of metric distance versus rank. Do not
claim that this design can independently separate those variables.

Exactly100 independent permutations/query/model/condition,30million total,
fixed before outcomes. Five model-level workers; one NumPy/BLAS thread each.
N0 uses Phase3G PCG64 SeedSequence([90700011,model_index,q,p]) over original
candidate order, so its first50 permutations must reproduce Phase3G raw
arrays exactly. Nj uses PCG64 SeedSequence([90800011,model_index,q,j,p]);
within each bin in listed order, independently permute its stable exact-rank
positions using that generator. No data-dependent permutation count/stopping.
Keep every ordered null top10 and scalar metric in per-model/per-condition
compressed arrays, plus per-query null mean/sampleSD/quantiles. No full
null inversion-pair identities. Residuals/scores use FP64 on stored FP32
values; d+(pq-d) must match actual PQ exactly. Stable score-sort ties follow
original candidate order. Strict inversions require exact_inside<exact_outside
AND score_inside>score_outside. Keep signed losses and negative null scores.
Reference full-sort/nested-pair tests precede the run.

Aggregate null mean/sampleSD/p05/p95 across100 full independently permuted
query datasets; conditional Monte Carlo uncertainty, not population CIs.
Report each model and all/oracle1 scopes. Loss means and excess use integer
hit-count sums divided by10*100 to avoid artificial floating-point ties.
Closure=(mean_N0-mean_Nj)/(mean_N0-mean_observed), only if denominator>1e-6;
no clipping of negative/>1 closure. Also report per-draw closure dispersion
with invalid-denominator counts; paired permutation indices are independent
streams, not common-random-number interventions. Analogous descriptive
inversion-gap closure is separately named. Not causal attribution, explained
variance or percentage of mechanism. Models share queries, not independent
query populations. No new numerical hypothesis thresholds.

Residual diagnostics: each exact rank1..30; tail bins31..40,41..100,101..n;
special regions1..5,6..10,1..10,11..20,21..n. For each report signed mean, MAE,
population residual SD/variance and p50/p90/p95 absolute residual, using ALL
candidate residuals in the stated group. Also use10 equal-count pooled
numeric-exact-distance quantile bins and per-query distance-quartile summaries.
Pooled candidate statistics can mix between-query effects. No causality claim.

Interpretation A: coarse conditioning closes most gap, finer conditioning
unnecessary, visible rank-dependent error scales. B: even N3/N4 retain much
larger loss than real PQ, coarse conditioning closes little. C: material
partial closure. Fine conditioning structurally restricts randomization;
approaching identity alone is not evidence for a PQ-specific mechanism.

After a saved PRIMARY checkpoint only: PQ-side observability bridge on
the same PQ-scored fixed pools, WITHOUT exact distances as feature inputs.
Features: g10_11,g9_12; each divided by scale=max(abs(PQ_10),1e-12);
counts satisfying abs(PQ_v-PQ_10)<=f*scale, f=.01,.02,.05, including both
sides and the kth item/ties. Seven features total. Stable sorting uses only
PQ scores. Offline label=observed exact-oracle minus PQ fixed-pool recall.
Existing runs do not contain a ready per-vector reconstruction-error scalar;
omit this conditional optional feature rather than add a new decoding
pipeline. No new trained detector. Note the candidate pool was selected by
exact traversal: this is feature feasibility, NOT a deployable-online test.

For each feature/model report Spearman and tie-preserving quintile summaries
of loss and harmful rate, all and oracle1 scopes. Identical feature values
must not split across bins; discrete count features can have<5 populated
bins. Also compare first/last risk-ordered occupied bins. Choose SINGLE
feature by consistent correlation sign across five models and largest
minimum absolute oracle1 Spearman, then median absolute correlation, then
feature name. This exploratory selection is not held-out validation. Record
if no stable nontrivial stratification appears. Do not claim prediction from
oracle correlations or model generalization from shared queries.

Phase3H ends exact-distance-only mechanism analysis regardless of outcome.
Recommend exactly one follow-up aimed at held-out online uncertainty /
selective-refinement feasibility, never another residual-mechanism phase.
