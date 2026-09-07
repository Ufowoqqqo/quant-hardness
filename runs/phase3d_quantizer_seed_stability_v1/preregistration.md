# Phase 3D pre-registration

Registered before any new model training or Phase 3D result inspection.
Execution starts from Git 71cb5be96e7cd062127282b48c715841d55d9ff5.

## Frozen experiment

SIFT1M: 1,000,000 database vectors, 10,000 queries, dimension 128;
no preprocessing changes. Reuse only the existing Phase 3C exact-FP32
efSearch=64, k=10, corrected unique L0 candidate pools, exact scores, GT and
stable first-evaluation-order tie policy. No graph search, graph rebuild,
traversal changes, or new quantizer. Source hashes will be checked before
and after scoring; original files are read-only inputs.

PQ64x8, standard FAISS Train_default, squared L2 ADC, niter=25, nredo=1,
max_points_per_centroid=256, min_points_per_centroid=39, random initialization,
no polysemous training, no spherical/integer/frozen centroids, no faster
subsampling. FAISS commit 20f14b31a6d54e243a3d1de6ae193fc4c3ec18ed.
Training seeds: 30260907, 70300001, 70300019, 70300043, 70300067.

Pinned FAISS ClusteringHelpers.cpp uses rand_perm over database IDs with
cp.seed and takes the first 256*256=65,536 IDs. All 64 subspaces use the
same subset/order. Materialize that original subset with sample seed
30260907 and use it unchanged for all five training initializations. This
avoids confounding initialization with training-sample variation. First
retrain seed 0 and require byte-identical Phase 3C codebooks, codes, and
all saved PQ64 candidate scores. Stop and investigate if identity fails.
Training/encoding threads=24, scoring threads=1, OPENBLAS_NUM_THREADS=1.
Save training IDs, full models/codes, source/binary/config hashes, machine,
FAISS/Git revisions and dirty state. Fresh models encode all database IDs
in their original order; check re-encoding alignment.

## Hypotheses (unchanged user definitions)

H1: Aggregate PQ64 quality is relatively stable across independently trained
standard PQ models. Report mean, sample SD (ddof=1), min/max, and coefficient
of variation without imposing a post-hoc binary stability threshold.

H2: Harmfulness is substantially more stable than idiosyncratic codebook
noise: median of the 10 off-diagonal pairwise harmful-set Jaccards >=0.60
AND median pairwise Spearman of signed per-query ranking losses >=0.50.
Exploratory criteria, not significance claims. Also report independence
baselines based on marginal harmful prevalence, expected all-five harmful
count, and tie-sensitive top-severity overlaps. Do not replace thresholds.

H3: A nontrivial subset is harmful under all five models. Report its size
and independence expectation; do not invent a post-hoc 'nontrivial' cutoff.

H4: Persistent queries have more fragile exact local geometry than robust
queries, including the pre-specified oracle_recall=1 stratum. Report all
descriptors, group sizes, distributions and Spearman correlations, not just
favorable ones. No predictive model or intrinsic hard-class assertion.

H5: Individual inverted pairs are less stable than query harmfulness.
Compare pair/query frequency distributions and pooled pair Jaccards
(support sizes differ; do not equate their numerical baselines).

## Metrics and conventions

Reuse Phase 3C strict inversions: exact(i)<exact(j) AND PQ(i)>PQ(j).
Exact and PQ score ties are not strict inversions. Preserve ordered top-k
lists, sets, signed losses, all selected displaced/intruder Cartesian
pairs, and every cross-boundary pair inverted in any seed with its
five-bit mask, scores/margins/violations recoverable from raw candidates.
Pair-frequency classes: one-off=1/5, minority=2/5, majority=3/5 or 4/5,
persistent=5/5. Query categories: robust=0, occasional=1/5 or 2/5,
usual=3/5 or 4/5, persistent=5/5. No clipping of recall losses.

Global metrics are query-weighted means; also report candidate-weighted
MAE. Random pairs use the identical 512 pairs per query as Phase 3C,
PCG64(60400001+query_id). Query SD across seeds uses ddof=1. Top 5/10/20%
severity ties use ascending query ID, with tie-inclusive Jaccard and
expected overlap under independent random tie resolution as sensitivity.

Loss attribution to query categories is a disjoint sum of signed losses.
Pair-level attribution is descriptive, NOT causal: for each missing GT
item in exact top-k, allocate its +1/k debit equally among actual selected
intruders that strictly overtake it. Assign to the pair's frequency bucket.
Unexplained score ties/no strict overtake get an explicit unassigned debit;
GT intruders get a separate -1/k credit. This accounts exactly for signed
loss without double-counting Cartesian pairs. Also report query/model loss
stratified by presence of high-frequency (>=3/5) inversions; these strata
do not constitute interventions or unique causal attribution.

## Sequencing, controls and output

Unit-test metrics against tiny exhaustive references, uniqueness, score
ordering, union masks, signed loss attribution, ties, and top-severity
overlaps. Require seed0 score/metric identity with Phase 3C. Hash all prior
candidate chunks/metadata and graph state; do not re-generate exact scores.
Save a primary summary checkpoint BEFORE ensemble scoring analysis.
Then optionally average five float32 ADC scores in float64, uniform weights,
on the identical pools; report recall/loss/inversions only as a diagnostic,
not an algorithm, with no timing benchmark or optimized weights.

All raw per-query/score/pair/model data under this run; tables/figures under
phase3d prefixes. Analysis must regenerate without FAISS/database access.
Report Cases A/B/C skeptically; exactly one next experiment, no algorithm.
