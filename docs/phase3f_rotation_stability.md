# Phase 3F: orthogonal-basis stability

Status: primary experiment and post-primary diagnostics complete,2026-09-07–08.

The authoritative preregistration is
`runs/phase3f_rotation_stability_v1/preregistration.md`; all parameters are
in `configs/indexes/phase3f_rotation_stability.conf`.

Use the unchanged Phase 3C exact L0 pools and exact reference. The graph is
not rebuilt or searched. Reuse standard FAISS PQ training/scoring abstractions.
Add a score-only rotation input mode and a separate binary, preserving prior
export binaries. Validate rotations before any model training. Existing
Phase 3C single-model metric reference is reused without optimization.

H1-H5 and tolerances were fixed before observing results. No prior runs or
metric definitions were overwritten. This is a scoring-only invariance
experiment, not an ANN performance benchmark or an algorithm proposal.

## Fixed design and numerical semantics

SIFT1M:1,000,000 database vectors,10,000 queries,d128. The original FP32
HNSW has M16,efConstruction80; the recorded pools come from efSearch64,k10.
There are10,353,047 candidate records. Graph fingerprint:
`1b3c18f6e2dd3ccec56e74eed30458f60a65f01234bf51478af25d84b5c02d16`.
FAISS remains pinned at `20f14b31a6d54e243a3d1de6ae193fc4c3ec18ed`.
Execution HEAD is `676169d68f594e47d40340b20e8c77ac92c1a2f9` with dirty
research additions and source/executable hashes, not an unrecorded revision.

The single training subset is Phase 3E A:65,536 unique base IDs from FAISS
rand_perm seed80400011, in the original sampled order. Its checksum is
`c984c6530b9675be22273bd307b5bf3eecbb9a98ce4b675cccb0b33a15542c5c`.
Seeds80500011,80500029,80500047 are reused as matched initialization labels
at every basis. All15 models are newly trained; no historical codebook is
reused. Standard Train_default PQ64x8,25 iterations,1 redo, maximum256
points/centroid, no polysemous training;24 training/encoding threads,1 ADC
scoring thread,OPENBLAS_NUM_THREADS=1.
Machine:Intel Core i9-10920X,12 cores/24 hardware threads; hostname/kernel
and compiler in manifests, full CPU description in `machine_cpu.txt`.
NumPy1.23.5 and Matplotlib3.9.4 are recorded for analysis reproduction.

R0=I. For R1-R4 draw dense128x128 standard normals with independent PCG64
seeds90600011,90600037,90600059,90600083. Float64 QR, column sign correction
using diag(R_QR), yields an orthogonal matrix. Column-vector convention is
x_R=R x; row arrays use X_R=X @ R.T. Transform in float64 then round once
to FP32. No normalization, centering, learning, or graph operation. Save
the actual matrices and complete transformed FP32 database/query arrays.
Training vectors are selected from this same transformed database by the
unchanged IDs. Matrix, transformed-input, sample, codebook and code hashes
are retained for every model.

## Pre-training invariance results

All five conditions passed before the first PQ train call. Orthogonality
max absolute error<=1.67e-15 (threshold1e-12).1000 deterministic sampled
norm/base-pair/query-base distances pass both float64 and FP32 checks:
float64 atol1e-8+rtol1e-12; stored FP32 with float64 squared-distance
accumulation atol.005+rtol5e-7; FP32 accumulation rtol2e-6, same atol.
Maximum sampled L2 norm error for the random bases is below6.5e-6.

Every candidate score is independently recomputed from transformed vectors
with direct float64 squared differences. Max absolute errors by R1-R4 are
.007331,.006892,.006859,.007025; max relative errors are3.079e-7,2.525e-7,
2.879e-7,3.720e-7. None exceeds the predeclared combined tolerance. There
are ZERO reversals between distinct original exact-distance values across
all saved candidate pools. The original reference ordering and oracle
Recall@10=.95315 are retained, unmodified, for every PQ condition.

Raw floating-point tie order is NOT claimed identical. Among all10,000
queries, R1-R4 raw top10 membership changes occur for7,5,6,6 queries, and
raw top10 ordered-list changes for32,35,32,31. These are original exact
ties; every query's original and rotated raw top10 is preserved in
`rotation_r/candidate_validation.jsonl.gz`. Whole-pool ordering changes for
9713,9682,9696,9703 queries because much larger candidate pools contain many
integer-distance ties. R3's raw mean oracle is.95316 instead of.95315;
the other raw means remain.95315. This is an explicitly recorded tie-policy
effect, not evidence of changed geometry, and is not used as a new reference.

Exhaustive GT verification covers100 deterministic queries over all1m
database vectors, at every basis (not a claim of exhaustive revalidation
of all10,000 queries). Recomputed unrotated GT matches provided GT modulo
exact ties. Rotated GT has no non-tied membership change; R4 has one
provided-GT membership tie change. Direct-difference spot checks validate
the float64 norm/dot exhaustive implementation. All original exact GT
identities remain the scoring reference. Gate artifacts retain sample IDs,
unrotated exhaustive distances and per-basis validation details.

## Analysis definitions and limitations fixed before outcomes

Single-model metrics reuse the Phase 3C reference: strict exact_i<exact_j
and ADC_i>ADC_j for inversion; stable original-candidate-order score ties.
Global MAE is a per-query candidate mean, then averaged over queries;
candidate-weighted MAE is also reported. The same512 random candidate pairs
per query as3C serve the generic inversion control. Maximum violation
retains the existing all-AxB definition; tied exact replacements remain
recorded but are not counted as strict inversions. Losses stay signed.

A rotation is majority-harmful for q if at least2 of its3 models lose
recall. Classes by the number of majority-harmful rotations: robust0,
occasional1-2,usual3-4,persistent5. This robust category can include a query
harmful in one initialization of a rotation. Report class counts and signed
and positive loss sums, including the oracle=1 stratum.

W_pop=mean_r Var_i(L_ri), B_pop=Var_r(mean_i L_ri), T_pop=W+B; all population
variances. Observed B contains finite-initialization variability. Also
report the signed sample-variance contrast B_sample-W_sample/3, without
clipping or random-effects claims. The matched seed design additionally
permits a descriptive two-way rotation/init/interaction sum of squares;
none is an identified causal geometry/alignment fraction.

All105 model pairs:15 within and90 between rotations. Report marginal
harmful prevalences and independent-marginal Jaccard baselines; high overlap
can occur mechanically when global PQ loss becomes more widespread. Separate
identity-to-random/random-to-random, matched/unmatched seed labels, oracle=1,
and exact-boundary-tie exclusion. Pairwise entries share models and queries;
their spread is descriptive, not independent replication-based uncertainty.

After a saved primary checkpoint only: uniform float64 score averages over
I1 at five bases, plus each basis's three initializations. Compare to each
group's own member baseline, not just identity. Five versus three models
and unequal member quality limit any diversity-source interpretation. No
optimization, timing benchmark, or algorithm recommendation is intended.

## Observations: aggregate quality

Every random basis modestly worsens aggregate PQ64 quality relative to the
identity. None causes collapse. The difference in rotation-mean Recall
(.00564–.00662 below identity) exceeds the observed within-basis initialization
SDs (.00029–.00114). R1-R4 are arbitrary basis labels, NOT ordered severities.

| Basis | Mean Recall ± init SD | Recall min–max | Mean loss | Critical inversions | Candidate MAE | Random inversion rate |
|---|---:|---:|---:|---:|---:|---:|
| R0 | .852903 ± .000667 | .852340–.853640 | .100247 | 5.56853 | 1312.572 | .0345030 |
| R1 | .846283 ± .000662 | .845520–.846700 | .106867 | 6.19450 | 1378.831 | .0360297 |
| R2 | .847147 ± .001141 | .845830–.847830 | .106003 | 6.16027 | 1377.725 | .0359752 |
| R3 | .846940 ± .000288 | .846700–.847260 | .106210 | 6.11220 | 1373.415 | .0359029 |
| R4 | .847267 ± .000738 | .846640–.848080 | .105883 | 6.06193 | 1375.362 | .0359101 |

All15 individual-model means are in `phase3f_models.csv/json`; mean, sample
SD, min and max for EVERY metric/basis are in
`phase3f_rotation_quality.csv/json`, including candidate-weighted MAE.
No condition was dropped or normalized to match identity quality.

## Observations: query stability

Median pairwise similarities:

| Model-pair group | Pairs | Harmful-set Jaccard | Loss Spearman | Critical-count Spearman | Inversion-pair Jaccard |
|---|---:|---:|---:|---:|---:|
| Within rotation | 15 | .68536 | .41177 | .44106 | .14487 |
| Between rotations | 90 | .66903 | .36351 | .37991 | .11702 |
| Identity to random | 36 | .65839 | .35682 | .36856 | .11658 |
| Random to random | 54 | .67366 | .36858 | .38622 | .11707 |

Changing basis reduces severity stability, but harmful-query membership does
not collapse. Within-minus-between median differences are .01633 for Jaccard,
.04825 for loss Spearman, and .06115 for critical-count Spearman. Between-basis
matched-init/unmatched-init loss correlations are .36430/.36318, respectively.
Thus the between-basis result is not rescued simply by matching seed labels.

Observed overlap is not sufficient on its own: model harmful prevalences are
high (identity mean70.37%; random-basis means about72–73%). Median Jaccard under
a homogeneous independent-marginal baseline is .57017 within/.56853 between,
below the observed values. This is a descriptive reference, not a test of
independent queries or proof of an intrinsic class.

The oracle=1 stratum (7136 queries) has within/between harmful Jaccard
.73964/.72699 and loss Spearman .35692/.30435. Excluding the171 exact-boundary
tie queries gives .68577/.66946 and .40931/.36156. The principal direction
does not depend on the incomplete-oracle population or exact boundary ties.
All pairwise rows, distributions, min/max/SD and sensitivity results are saved.

## Observations: rotation-persistent population and variance

Counts by number of majority-harmful rotations0..5:
674,529,705,1142,1970,4980. Majority means at least2/3 initializations; it does
NOT mean all15 models harm each rotation-persistent query.

| Query category | Queries | Fraction | Share of total ranking loss |
|---|---:|---:|---:|
| Rotation-robust (0) | 674 | 6.74% | .72% |
| Occasionally vulnerable (1–2) | 1234 | 12.34% | 4.98% |
| Usually vulnerable (3–4) | 3112 | 31.12% | 25.39% |
| Rotation-persistent (5) | 4980 | 49.80% | 68.91% |

Among oracle=1 queries,4417/7136 (61.90%) are rotation-persistent and carry
75.76% of that stratum's loss. An independent-rotation-majority marginal
reference predicts2582.96 persistent queries versus4980 observed, again a
descriptive null reference only. No negative net query/model loss occurred;
signed values were retained rather than clipped.

Mean W_pop=.00286698, observed B_pop=.00143299, total=.00429996, with exact
per-query W+B=T validation. Observed shares are66.67% within versus33.33%
between rotation means; this is NOT a causal allocation to initialization
versus alignment. Mean W_sample=.00430047 and B_sample=.00179123; their
signed contrast B_sample-W_sample/3=.000357744, about8.32% of W_sample.
4160 negative contrasts and186 zero-model-variance queries remain recorded.

The descriptive matched5x3 two-way decomposition has mean rotation main
variance .00143299, init main variance .000567564, interaction .00229941.
Rotation means average3 models while initialization means average5, and
interactions are substantial: these values cannot establish that alignment
accounts for a particular fraction of realized errors. Additional basis-linked
variation is present; initialization/codebook realization noise remains large.

## Observations: invariant geometry

Spearman correlations in the oracle=1 stratum:

| Exact descriptor | Overall15-model mean loss | Harmful-rotation frequency |
|---|---:|---:|
| Relative rank10/11 gap | −.49363 | −.44140 |
| d12−d9 | −.73912 | −.56766 |
| d15−d6 | −.65568 | −.46694 |
| Candidate-pool size | .02784 | .04200 |

For d12−d9, mean single-model realized-loss correlation is−.44074
(range−.46094 to−.42150), versus−.73912 for the15-model mean. Median d12−d9
for oracle=1 robust/occasional/usual/persistent groups is4056.5/2747/1693/915.
Median relative rank10/11 gaps are .05359/.02571/.01116/.00552.
The wider local span remains more informative than a single adjacent gap.

Other oracle=1 mean-loss correlations are d1=.08128,dk=−.05553,d(k+1)=−.07139,
d20=−.10501,local distance concentration=−.33745; the constant oracle=1
descriptor correctly yields an undefined correlation. All descriptors and
all-query/complete-oracle analyses, including weak associations, are retained.
No predictor was fitted. Averaging15 models rather than9 in Phase3E can
itself strengthen correlations by reducing noise; the increase over Phase3E
is not attributed solely to rotation diversity.

## Observations: inversion-pair identity

The union contains362,197 query/inside/outside triples inverted at least once.
44.55% occur in exactly one model;49.55% are confined to one rotation;
50.45% recur across>=2 rotations; only3.58% recur across all5 rotations.
Only ONE pair is inverted in all15 models (.000276% of the union).
Pair rotation frequency uses >=1 initialization, a less stringent criterion
than the query majority rule, yet all-basis pair recurrence is still rare.

Pair Jaccard drops from .14487 within to .11702 between bases and is much
lower than harmful-query Jaccard. Specific pairs are realization-sensitive
even within a basis, with additional basis sensitivity. A query can remain
harmful while different pairs fail; the data do not support equating a stable
query susceptibility with a stable list of failed candidate pairs.

## Post-primary diagnostics (not methods)

The primary checkpoint was saved before calculating any of these averages.

| Fixed averaging group | Models | Recall | Loss | Critical inversions | Member-average loss recovered |
|---|---:|---:|---:|---:|---:|
| I1 across R0–R4 | 5 | .90574 | .04741 | 1.5996 | 54.84% |
| R0 initializations | 3 | .88588 | .06727 | 2.8399 | 32.90% |
| R1 initializations | 3 | .88500 | .06815 | 2.8878 | 36.23% |
| R2 initializations | 3 | .88518 | .06797 | 2.9063 | 35.88% |
| R3 initializations | 3 | .88442 | .06873 | 2.8760 | 35.29% |
| R4 initializations | 3 | .88492 | .06823 | 2.8302 | 35.56% |

The five-basis group's member mean Recall is .848168, loss .104982. All
groups' individual baselines are in `phase3f_ensemble.csv/json`. Errors can
partly cancel without changing the candidate pool; the remaining gap to
the .95315 oracle is not eliminated. Five versus three models, unequal
member quality and only one fixed composition prevent attributing the gain
specifically to rotation diversity. No weights, composition, performance
optimization or algorithm claim is made.

## Hypotheses and decision

H1 passes under the preregistered numeric/tie semantics, not literal equality
of floating-point tie order. H2 has directional support: an additional but
modest stability reduction beyond initialization. H3 is supported by the
49.80% rotation-persistent population. H4 is supported descriptively,
including oracle=1. H5 is supported by far lower pair than query stability.
None of H2-H5 is a statistical-significance claim.

**Best-supported interpretation: CASE A for rotation-averaged susceptibility.**
This is a qualitative comparison of the proposed explanations, not a causal
variance-dominance estimate. Exact local distance geometry predicts mean
susceptibility after changing basis; query membership survives; pair identity
does not. Case B's collapse/weak-geometry pattern is absent. Case C correctly
describes remaining realized-error noise, but the evidence does not show that
coordinate alignment determines the main part of average query vulnerability.

1. **How much survives?**49.80% remain majority-harmful at all5 bases and
   account for68.91% of loss;61.90% persistence within oracle=1. Between-basis
   harmful Jaccard stays .66903, versus .68536 within.
2. **Is alignment more important than initialization?**For aggregate quality,
   changing from identity to these random bases has a larger shift than the
   observed within-basis initialization SD. For per-query realized failures,
   the additional loss-correlation drop is only .04825 and initialization/
   interaction variability remains substantial. Alignment dominance is not
   established by either the variance shares or the shared-seed design.
3. **Does invariant geometry explain mean susceptibility?**It remains strongly
   associated: oracle=1 d12−d9 correlation−.73912, versus−.44074 for a typical
   single-model loss. This is association, not a demonstrated causal gap effect.
4. **Are pairs basis-specific while queries remain vulnerable?**Yes, but pair
   instability already exists within bases: pair Jaccard .14487/.11702,
   only3.58% across all bases, versus49.80% majority-persistent queries.
5. **Which case?**A for averaged susceptibility, with explicit realization
   noise and modest basis-linked effects; no universal intrinsic hard-query
   class, novelty, or navigability claim follows from this single dataset,
   fixed pool, PQ architecture, one training subset and four random draws.

### Exactly one next experiment (not run)

Run a **within-query PQ-residual permutation negative control** on these
same saved pools and scores. For each recorded model/query, randomly permute
`error(v)=d_PQ(v)-d_exact(v)` over candidate IDs using a small fixed set of
deterministic permutation seeds, then score `d_exact(v)+permuted_error(v)`.
This preserves each query/model's full marginal error distribution, signed
bias and MAE but destroys candidate-specific error alignment. Compare the
same mean-loss, local-gap association, harmful-query and critical-inversion
metrics to actual PQ without retraining or graph search. If the null
reproduces susceptibility, much of the story may be generic noisy top-k
selection; if not, structured candidate-dependent PQ errors need explanation.
This is a mechanism-falsification control, not a quantizer or retrieval method.

## Artifacts and reproducibility

Raw data include every model's scores/index/training IDs; complete transformed
inputs and QR matrices; numerical validation rows;15-model query metrics;
all union inversion masks/violations and replacement pairs; and optional
diagnostic query outputs. Original candidate/exact/GT files remain immutable
and referenced by checksum rather than silently replaced. Files are under
`runs/phase3f_rotation_stability_v1/`; tables and8 SVG figures use the
`results/tables/phase3f_*` and `results/figures/phase3f_*` prefixes.
Commands and operational anomalies are recorded in `docs/experiment_log.md`.
All49 Python tests and5 C++ tests pass. Independent derivation from raw
scores reproduces45 analysis/table/figure files byte-for-byte, including
all8 SVGs. Geometry, aggregate-recall and diagnostic figures were rendered
and visually checked. See
`runs/phase3f_rotation_stability_v1/final_verification.json` for the final
frozen-input/model/transform audit and artifact hashes.
