# Phase 3E pre-registration

Registered before training/scoring any new model or inspecting Phase 3E results.
Execution base Git: 913d7eddbe5f6f6e7b83efa7f2244de1bc2bb246.

## Frozen inputs and balanced design

SIFT1M (1M database vectors, 10k queries, d128), exact-FP32 HNSW M16,
efConstruction80, graph seed20260907, efSearch64/k10 pool provenance.
Use ONLY the saved Phase 3C exact-FP32 unique L0 evaluation pools (10,353,047
records), exact squared-L2 scores, GT, query order and first-evaluation-order
stable tie policy. No traversal, preprocessing, graph or candidate changes.
FAISS 20f14b31a6d54e243a3d1de6ae193fc4c3ec18ed, standard PQ64x8,
Train_default/RANDOM initialization, niter25/nredo1, max points/centroid256,
min39, no polysemous/spherical/integer/frozen centroids/faster subsampling.

Draw 65,536 database IDs without replacement in FAISS rand_perm order with
sample seeds A=80400011, B=80400037, C=80400059. Subsets are independently
drawn, NOT artificially disjoint: overlaps are expected and will be reported.
Within each subset, use identical IDs/order/vectors. Initialize 9 NEW models:
A1/A2/A3 seeds80500011/80500029/80500047;
B1/B2/B3 seeds80500063/80500081/80500103;
C1/C2/C3 seeds80500121/80500139/80500157.
All nine seeds differ. This is a balanced 3 subsets x 3 independent
initializations nested within subset, not reuse of a common initialization
seed crossed with all subsets. No parameter selection from outcomes.
Training/encoding threads24, scoring1, OPENBLAS_NUM_THREADS=1.

Retain all full models, training IDs, codebooks/codes checksums and all ADC
scores, plus source/binary/config/Git/FAISS/machine metadata. Verify sample
identity within subset, distinct subset hashes, frozen graph/source hashes,
score alignment and scalar/batch4 equivalence. Independent sample draws may
overlap about 4295 IDs out of65536 in expectation; do not force disjointness.

Existing Phase 3D O1..O5, original subset seed30260907, are eligible ONLY
for separately labeled secondary model-quality and query-stability analyses
after verifying metadata, candidate hashes and raw-score/model integrity.
They do not enter any balanced primary variance/frequency/ensemble statistic.

## Hypotheses (unchanged)

H1: Aggregate PQ64 quality remains stable across independent training subsets.
H2: Within-subset harmful-query similarity is higher than between-subset.
H3: A meaningful subset remains harmful across all three independently drawn
training subsets, supporting a geometry-related susceptibility component.
H4: Exact inversion-pair identity is substantially less stable across training
subsets than query-level harmfulness.
H5: Fragile exact local top-k geometry associates with greater mean harmful
frequency across training-sample and initialization randomness.

No post-hoc numerical pass thresholds will be invented for these qualitative
hypotheses. For H2 show both median difference and all pair distributions;
a tiny directional difference is not 'substantially' lower stability. H3
reports population/loss shares, oracle1 stratum, and marginal-prevalence
independence context rather than claiming significance or an intrinsic class.

## Metrics and transparent variance accounting

Use the validated Phase 3D single-model metrics unchanged, including strict
inversions (exact_i<exact_j AND pq_i>pq_j), signed losses, same512 random
pairs/query, PCG64(60400001+qid), epsilon1e-12. Retain all query/model top-10
ordered lists/sets, replacements, and every pair inverted by >=1/9 models
with nine-bit mask and per-subset counts. Pair model frequency is count/9;
subset frequency is number of subsets with >=1 inversion (also save /3).
Also distinguish repetition within subset (>=2 or all3 init inversions).

For L shaped [3,3] for one query, mu_s=mean_i L_si, mu=mean_s mu_s:

W_pop = (1/9) sum_si (L_si-mu_s)^2  [initialization_variability]
B_pop = (1/3) sum_s (mu_s-mu)^2     [training_subset_variability, observed]
T_pop = (1/9) sum_si (L_si-mu)^2    [total_model_variability]

Require T_pop = W_pop+B_pop to floating precision, for every query.
within_subset_variance/variability aliases refer to W_pop;
between_subset_variance/variability aliases refer to B_pop, not pure effects.
Report variances in recall-squared units; SDs are explicitly named sqrt.
Aggregate by mean across queries and distributions; shares are ratio of sums,
not mean of undefined per-query ratios. Zero-variance query counts retained.

B_pop includes variation from averaging only3 initializations. Also report
W_sample=mean_s var_i(L_si,ddof=1), B_sample=var_s(mu_s,ddof=1), and the
SIGNED diagnostic C=B_sample-W_sample/3. Under an additional independent
homoscedastic nested sampling model, C would be a method-of-moments sample
variance contrast. Here it is only a descriptive noise-floor comparison;
no random-effects inference, confidence intervals, nonnegative clipping or
causal 'percent due to samples' is claimed. Negative contrasts are retained.

Between-query var_q(mu_q) and mean_q T_pop also partition variance over all
query/model entries; the between-query term is NOT labeled causal geometry.
One fixed dataset, PQ layout, oracle and finite samples leave other common
sources of systematic susceptibility unseparated.

## Stability and geometry

Primary pairwise stats: 9 within-subset pairs, 27 between-subset pairs,
Jaccard harmful sets, Spearman signed losses and inversion counts, all
distributions/medians, with marginal harmful-prevalence baselines. Pairwise
comparisons share models/queries; they are not independent replicates.

Per-query harmful_count_all=0..9; harmful_frequency_all=count/9;
harmful_subset_count=number of subsets with >=2/3 harmful models, and
harmful_subset_frequency=count/3. Robust=<=2/9, model-specific=3..5/9,
usually=6..8/9, universal=9/9. Subset-persistent iff all3 subset majorities.
Report category population, signed and positive-loss contributions, and
subset-persistent contributions separately (overlapping the categories).
Do not reuse Phase 3D's different 'robust' definition silently.

Exact descriptors: d1,d10,d11,d20, boundary/relative margin, local gaps r1/2/5,
(d20-d1)/max(d1,epsilon), pool size, oracle recall. Spearman and tied-value-
preserving quantile bins against harmful_frequency_all, harmful_subset_frequency,
overall_mean_loss, B_pop and W_pop, both all queries and oracle=1. Compare
geometry's correlation with mean loss versus per-model realized loss; averaging
itself reduces noise and must not be called causal predictive validation.
No fitted classifier. Exact-tie exclusion is a separately labeled sensitivity.

## Sequencing and optional diagnostics

Unit-test variance identities, special pure-init/pure-subset/constant cases,
signed contrasts, counts/category boundaries, exhaustive inversion masks,
pair subset membership and frozen-score/tie/reference controls.
Freeze a primary checkpoint after all primary/secondary analysis. Then score
two predetermined uniform float64 averages on the SAME candidates:
init diagnostic=A1/A2/A3; cross-subset diagnostic=A1/B1/C1. Equal3 scores,
no weight/composition optimization or benchmark. Report member-average baseline
as well as resulting recall/loss/inversions to avoid confounding base quality.

Preserve old runs. Regenerate analyses/figures from raw scores independently.
Assess Cases A/B/C without forcing geometry dominance; exactly one next
experiment, no new algorithm or traversal instrumentation.
