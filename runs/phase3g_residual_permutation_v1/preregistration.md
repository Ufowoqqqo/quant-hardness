# Phase 3G pre-registration (before null outcomes)

Question: does candidate-specific PQ residual assignment matter beyond the
exact candidate geometry and the empirical per-query residual distribution?

H1. A substantial fraction of ranking loss can be reproduced by randomly
assigning the same per-query PQ residuals to candidates, because small local
top-k margins make some queries generically noise-sensitive.
H2. Exact d12-d9 remains strongly associated with permutation-null mean loss.
H3. Observed PQ ranking loss differs measurably from the permutation null,
indicating candidate-specific structure beyond generic noise magnitude.
H4. If H3 holds, residual/rank or residual/distance dependence provides at
least one observable explanation for excess loss. Association is not proof
of mediation; a negative excess is allowed and must be interpreted as such.

Primary models are R0-I1,R1-I1,R2-I1,R3-I1,R4-I1, chosen by initialization
index, not performance. All 10,000 SIFT queries; unchanged FP32 exact L0
candidate arrays, exact scores, GT, top-10, efSearch=64, PQ64x8. No traversal,
training, or regeneration of PQ scores. Original graph/model manifests and
input checksums will be retained. No additional models are planned.

Use exactly 50 permutations for every query/model (2,500,000 total), fixed
before timing or outcomes. This meets the requested target and permits
preserving every ordered top-10 and scalar metric without excessive storage.
Each (rotation index,query_id,permutation index) has an independent PCG64
stream initialized by SeedSequence([90700011,r,q,p]); no data-driven stopping.
Candidates remain in their stored order. Residuals and null scores use FP64
on the stored FP32 exact/PQ values. Stable full argsort breaks score ties by
original candidate order, as in Phase 3C-F. Original exact ties remain fixed;
strict inversion requires d_inside < d_outside AND score_inside > score_outside.
Never clip negative null scores, residuals, or signed recall losses.

Observed reconstruction d+(pq-d) must equal the stored PQ score exactly and
reproduce all selected Phase 3F ordered results and core metrics. Every
permutation preserves the residual multiset exactly; the permutation
implementation is tested against a simple scalar full-sort / nested-pair
reference before production. No optimized PQ or search code is introduced.

Per-query null summaries: mean, sample SD (ddof=1), median,p05,p95 for
loss and inversions. Percentiles use linear interpolation. Observed empirical
rank records counts below/equal/above and the midrank (below+0.5*equal)/50;
also retain [below/50,(below+equal)/50]. These are descriptive ranks, not
p-values. Calculate loss comparisons using integer recall-hit differences
to avoid floating-point sign/tie artifacts.

Aggregate uncertainty: for permutation index p aggregate independently
shuffled queries, then report mean/sample SD/p05/p95 across the 50 complete
null datasets. This is Monte Carlo uncertainty conditional on these fixed
queries/models, NOT population sampling uncertainty or a confidence interval
for model generalization. Report each model separately and all/oracle=1
scopes; do not treat the five models as independent query replications.

Geometry: retain existing d1,dk,d(k+1),d20, boundary gaps r1/r2/r5,
relative rank-10 margin, concentration (d20-d1)/d1, candidate count and oracle
recall. Add relative d12-d9 divided by max(dk,epsilon). Compare observed,
null-mean and signed excess loss within each model and the five-model mean.
Use descriptive Spearman and quantile bins, with oracle=1 as primary control.

Residual diagnostics: exact ranks1-30 across all queries; regions ranks1-10,
11-20,21+; query-wise residual/exact-distance and residual/vector-norm
Spearman; deterministic 64 candidate samples/query for pooled binned
diagnostics (same sample IDs across models; candidate norm from identity
database). Reconstruction errors are not needed and will not be newly
computed. Boundary pair analysis compares observed replacement pairs with
the SAME pair identities under each shuffle, and separately summarizes
actual shuffled replacement pairs (selection-conditioned, not causal).
Preserve observed replacement-pair IDs and null summaries, not every null
inversion identity. Save all per-permutation top-10 IDs,hits,agreement,
displaced counts,strict inversion counts for regeneration.

Case A: observed and null loss close, excess small/weakly structured.
Case B: systematic substantial signed difference and observable assignment
structure; structured assignment may protect as well as harm retrieval.
Case C: generic fragility baseline plus material structure. No post-hoc
numeric significance threshold will be invented. H1-H4 are qualitative,
exploratory hypotheses, not significance claims. Neither per-query variety
nor correlation alone establishes a hard-query class or an online detector.

Only after primary analysis: centered-residual sanity check on q%100=0,
all five models and all50 permutations, using identical permutations.
Document any numerical ties. Do not fit parametric noise. Recommend exactly
one next experiment; do not implement any algorithm in this phase.
