# Phase 3D: quantizer-seed stability on frozen SIFT1M candidates

## Pre-registered design

This is a score-only stability experiment, not an algorithm or graph-search
experiment. Hypotheses, including the unchanged H2 thresholds (median
harmful-set Jaccard >=.60 AND median ranking-loss Spearman >=.50), were
registered in `runs/phase3d_quantizer_seed_stability_v1/preregistration.md`
before training any new model. The config is
`configs/indexes/phase3d_quantizer_seed_stability.conf`.

Reuse all 10,000 SIFT1M queries, 1M database vectors, dimension 128, graph
M16/efConstruction80/seed20260907, efSearch64/k10 candidate provenance,
corrected L0-only semantics. No preprocessing, graph, query, GT, exact
score or candidate-order changes. The 10,353,047 FP32-traversal candidate
records were generated once in Phase 3C and are read, never regenerated.
The graph fingerprint is
`1b3c18f6e2dd3ccec56e74eed30458f60a65f01234bf51478af25d84b5c02d16`.

### Isolating training initialization from training samples

In pinned FAISS `20f14b31a6d54e243a3d1de6ae193fc4c3ec18ed`, the relevant
source is `faiss/impl/ProductQuantizer.cpp` (`ProductQuantizer::train`),
`faiss/Clustering.cpp` (`Clustering::train_encoded`), and
`faiss/impl/ClusteringHelpers.cpp` (`subsample_training_set`). Standard
training independently slices each of 64 subspaces, then uses `rand_perm`
with `cp.seed` to subsample to 256*256=65,536 database IDs, in that order.
Every subspace uses the same seed and therefore the same IDs. Initialization
subsequently uses a separate call with `cp.seed+1` (one redo).

Simply changing `cp.seed` while supplying all 1M vectors would change both
sample and initialization. Instead, materialize the original sample with
seed30260907 once per model and verify its hash. Supply these identical
65,536 vectors in identical order to standard FAISS PQ training, avoiding
any second subsampling; vary only `cp.seed`. This is equivalent for seed0,
confirmed by byte-identical codebooks, all database codes and all saved
candidate ADC scores. Models use seeds
`30260907, 70300001, 70300019, 70300043, 70300067`.

PQ64x8, dsub2, Train_default, RANDOM initialization, niter25, nredo1,
max_points_per_centroid256, min_points_per_centroid39, no polysemous,
spherical, integer or frozen centroid training; no faster subsampling.
Threads: 24 training/encoding, 1 scoring, OPENBLAS_NUM_THREADS=1. Full
models, training IDs and checksums are retained. Original cache is read-only.

### Metric conventions

Exact and PQ top-10 use stable ascending scores with first recorded L0
evaluation order breaking ties. Membership and ordered lists are retained.
The exact oracle remains .95315, rather than the native exact .95319 that
differs at tied boundaries; no old metric is overwritten. Strict inversions
require both strict exact and reversed strict PQ order. Signed losses and
score ties are retained. Global MAE/random inversion metrics are averages
of per-query values; candidate-weighted MAE is additional, explicitly named.
All 512 random candidate pairs/query are byte-identical to Phase 3C.
Standard deviations use ddof=1. Spearman uses average tied ranks.
Exact boundary margin is d11-d10, relative margin divides by max(d10,1e-12).
Local gaps are d(k+r)-d(k-r+1), r=1/2/5; local concentration is
(d20-d1)/max(d1,1e-12), preserving the Phase 3C descriptor. Inversion
fractions divide by all 10*(|V|-10) opportunities. Maximum violation,
as in Phase 3C, is max(PQ(i)-PQ(j)) over those opportunities, including
exact-tied pairs; strict inversion counts still exclude exact ties.

Query categories are disjoint: robust=0 harmful models, occasional=1/2,
usual=3/4, persistent=5. Pair categories are one-off=1 inversion model,
minority=2, majority=3/4, persistent=5. The latter counts pairs from the
union over five models, not all A x B opportunities. Top-severity overlaps
use query-ID tie breaking and include independent random-tie expected
overlap plus tie-inclusive Jaccard. Geometry bins retain tied values.

Pair-frequency loss allocation is descriptive accounting, NOT causal:
each missing exact-top-k GT neighbor contributes +.1, split equally among
actual selected intruders strictly overtaking it. Sum by pair-frequency
class; retain unassigned tie/no-strict-overtake debits, and separate -.1
credits for selected GT intruders. This avoids double-counting replacement
Cartesian pairs and reconciles exactly with signed recall loss. No claim
is made that removing one allocated pair would independently recover recall.

## Observations: aggregate quality

The primary analysis completed and was copied into `primary_checkpoint/`
before running the optional diagnostic. All values below use the same
exact oracle (.95315), candidate pools and training subset.

| Model / initialization seed | Recall@10 | Ranking loss | Critical inversions/query | Global MAE | Random inversion rate |
|---|---:|---:|---:|---:|---:|
| 0 / 30260907 | .85313 | .10002 | 5.5846 | 1315.332 | .0343750 |
| 1 / 70300001 | .85331 | .09984 | 5.5208 | 1310.416 | .0343578 |
| 2 / 70300019 | .85266 | .10049 | 5.5110 | 1315.119 | .0344609 |
| 3 / 70300043 | .85304 | .10011 | 5.5048 | 1315.449 | .0344514 |
| 4 / 70300067 | .85312 | .10003 | 5.5914 | 1316.034 | .0344301 |

| Metric | Across-seed mean | Sample SD | Minimum | Maximum |
|---|---:|---:|---:|---:|
| Recall@10 | .853052 | .00024035 | .852660 | .853310 |
| Ranking loss | .100098 | .00024035 | .099840 | .100490 |
| Critical inversions/query | 5.54252 | .041976 | 5.50480 | 5.59140 |
| Global MAE (squared-L2 units) | 1314.4698 | 2.2916 | 1310.4157 | 1316.0343 |
| Random pair inversion rate | .03441504 | .00004618 | .03435781 | .03446094 |

Recall SD is .024 percentage points. Ranking-loss coefficient of variation
is .240%; critical-inversion CV is .757%. Aggregate quality is very similar
across these five initializations; none was dropped or normalized.

## Observations: query and severity stability

| Harmful-frequency category | Queries | Fraction | Fraction of total signed loss | Mean oracle recall |
|---|---:|---:|---:|---:|
| Robust (0/5) | 647 | 6.47% | 0% | .80974 |
| Occasionally harmful (1/5 or 2/5) | 1812 | 18.12% | 6.32% | .90806 |
| Usually harmful (3/5 or 4/5) | 3841 | 38.41% | 35.15% | .96600 |
| Persistently harmful (5/5) | 3700 | 37.00% | 58.53% | .98697 |

Counts at harmful frequencies 0/.2/.4/.6/.8/1 are
647/736/1076/1518/2323/3700. Queries harmful in at least 3/5 models account
for 93.68% of total loss, but comprise 75.41% of queries: this is not evidence
of a small rare hard-query tail. No negative **net** query-model losses were
observed; signed values and individual GT-intruder credits remain preserved.

Across all 10 model pairs:

- Harmful-set Jaccard median **.67116**, range .66128–.67815.
- Ranking-loss Spearman median **.41848**, mean .41781, range .40166–.42739.
- Critical-inversion-count Spearman mean **.44899**, range .43882–.45952.
- Mean per-query across-seed loss SD is .05690, despite aggregate loss SD
  of only .00024035. Stable averages do not imply stable per-query severity.

**H2 fails its pre-registered conjunction:** Jaccard passes .60, but loss
Spearman fails .50. Thresholds are unchanged. All 10 loss correlations are
below .50, not just the median.

Marginal harmful prevalence is 70.11–71.11%. A homogeneous independent
failure baseline with these marginals gives Jaccards .53994–.54864 and
1737.5 expected all-five harmful queries versus 3700 observed. Expected
all-five robust count is 22.45 versus 647 observed. Thus stability is above
this simple prevalence baseline, but that baseline assumes exchangeable
queries and does not adjust for oracle recall or local geometry; it is not
a significance test or a causal null model.

Mean overlap of the most severe queries across pairs:

| Top fraction | ID-tiebroken overlap | Independent random-tie expected overlap | Tie-inclusive Jaccard |
|---|---:|---:|---:|
| 5% | 25.08% | 23.17% | .23074 |
| 10% | 38.85% | 25.79% | .30780 |
| 20% | 45.10% | 39.89% | .30780 |

Losses are discrete in .1 steps. The top-10% result especially is sensitive
to reusing query-ID tie breaking. Tie-inclusive sets have unequal sizes and
are not directly comparable to fixed-size overlap. These observations
argue against treating a single seed's severe-query ranking as definitive.

## Observations: exact inversion pairs

The union contains **174,278** distinct (query, exact-top-k item, outsider)
pairs. There are 277,126 seed-specific inversion occurrences in total.

| Models inverting pair | Pairs | Fraction of union |
|---|---:|---:|
| 1 (one-off) | 107990 | 61.96% |
| 2 (minority) | 40341 | 23.15% |
| 3 (majority) | 17310 | 9.93% |
| 4 (majority) | 6661 | 3.82% |
| 5 (persistent) | 1976 | 1.13% |

Pooled inversion-pair Jaccard median is .15879 (range .15662–.16220), much
lower than harmful-query Jaccard. These supports have different marginal
sizes, so the values alone are not a normalized stability comparison.
More directly, only **963/3700 (26.03%)** persistently harmful queries have
even one all-five inverted pair. The other 2737 remain harmful without any
pair persisting in all models. Conversely, 47/647 robust queries have an
all-five inversion pair: not every critical-boundary membership inversion
changes GT recall in an incomplete/tied candidate pool.

Under the pre-registered descriptive loss allocation, low-frequency pairs
(1–2 models) receive 61.94% of total net loss as positive debits; high-frequency
pairs (3–5) receive 38.22%, including 4.93% for all-five pairs. There is an
additional .004% unassigned debit and -.170% GT-intruder credit. Debits can
sum above 100% before subtracting credits. These are accounting shares,
**not fractions of causally removable error**. In a separate non-overlapping
query-model presence stratification, 56.43% of query-model instances have
some >=3-frequency inverted pair and account for 75.41% of total loss;
co-occurring low-frequency pairs prevent a unique causal attribution.

## Observations: quantizer-independent geometry

All requested exact descriptors are tabulated for every category, with
mean/median/p90/p95/p99/min/max. The oracle-recall confound is substantial:
robust queries have mean oracle .810 versus .987 for persistent queries.
Zero ranking loss does not imply successful retrieval. The pre-registered
oracle=1 comparison retains 7136 queries:

| Category | Queries with oracle=1 | Median relative d11-d10 | Median d12-d9 | Median d15-d6 |
|---|---:|---:|---:|---:|
| Robust | 161 | .043743 | 3891 | 8319 |
| Occasionally harmful | 835 | .018439 | 2196 | 5904 |
| Usually harmful | 2857 | .008494 | 1381 | 4497 |
| Persistently harmful | 3283 | .005288 | 887 | 3435 |

Spearman associations with harmful frequency (all / oracle=1):

- Relative boundary margin: -.21863 / **-.39360**.
- Absolute d11-d10: -.28242 / -.39795.
- d12-d9: -.39683 / **-.48781** (largest absolute association among the
  measured exact descriptors in the complete-GT stratum).
- d15-d6: -.34619 / -.39698.
- Local (d20-d1)/d1 concentration: -.03968 / -.21739.
- Candidate count: -.14642 / +.02381.
- d1: -.13147 / +.07253; direction reverses after restricting oracle recall.

Persistent queries have smaller local gaps even with complete candidate GT,
supporting H4 descriptively. This is a graded association, not a disjoint
intrinsic class or demonstrated prediction rule. Robust complete-GT queries
number only 161. Correlations are exploratory, dependent descriptors are
not independent evidence, and conditioning on oracle recall changes the
population. In the oracle=1 stratum, median Jaccard rises to .72517 but
loss Spearman falls to .36231: H2 still does not pass.

## Post-primary ensemble diagnostic (not a proposed method)

Uniform float64 averaging of the five stored float32 ADC scores, on exactly
the same candidates, gives:

| Score | Recall@10 | Ranking loss | Critical inversions/query |
|---|---:|---:|---:|
| Exact oracle | .953150 | 0 | 0 |
| Average single-model performance | .853052 | .100098 | 5.54252 |
| Five-score average diagnostic | .895770 | .057380 | 2.18340 |

Compared with average single-model performance, loss decreases **42.68%**
and inversions decrease **60.61%**. A substantial residual loss remains.
Independent initialization errors partially cancel, but this number is not
a formal variance decomposition or proof that the remainder is intrinsic
geometry. Models share the same training sample, PQ axes, code rate and
training objective. No weights were fitted and no performance was benchmarked.

## Hypothesis assessment and required decision

- **H1:** descriptively supported for these five initializations: very small
  aggregate variability, without any post-hoc pass threshold.
- **H2:** **fails** the unchanged operational criterion (.67116 Jaccard,
  .41848 loss Spearman). Above-baseline overlap does not override failure.
- **H3:** 37% harmful under all five is a sizable persistent subset, beyond
  the simple prevalence baseline; no post-hoc statistical claim.
- **H4:** supported as a descriptive local-gap association, also in oracle=1.
  It does not establish an intrinsic or predictably separable query class.
- **H5:** supported descriptively: most union pairs occur in one/two models,
  only 1.13% in all five; most all-five harmful queries do not share even one
  all-five inverted pair. Query/pair marginal prevalence differs.

Answers to the seven requested questions:

1. **Aggregate quality stable?** Yes, under initialization changes with a
   fixed sample: recall range .85266–.85331, loss CV .240%.
2. **Query harmfulness stable?** Partly, above a homogeneous independent
   baseline, but not enough to pass H2; severity rank stability is moderate.
3. **Exact inversion pairs stable?** Mostly no: 85.11% of union pairs occur
   in only one/two models; all-five pair persistence is 1.13%.
4. **Intrinsic vulnerable class?** Evidence supports graded local-geometry
   susceptibility, not an intrinsic class: small multi-item boundary gaps
   associate with repeated harm, but seed severity and pair identity vary.
5. **How much is codebook-specific?** 56.53% of queries switch harmful status;
   only 6.32% of total loss is in the 1/2-harmful category, while repeated
   susceptibility carries most loss. Specific errors are much less stable
   than these query categories (61.96% one-off pairs). The score-average
   diagnostic removes 42.68% of loss, but no unique causal percentage of
   'codebook-specific loss' is identifiable from these summaries.
6. **Does score averaging substantially help?** Yes, to recall .89577, still
   below oracle .95315; it is a diagnostic only.
7. **Exactly one next experiment:** repeat this same five-initialization,
   score-only protocol on **one new independently sampled, internally fixed
   65,536-vector training subset**. Keep candidate pools, graph, exact scores,
   GT, PQ64x8 and initialization seeds unchanged; compare cross-subset
   harmful-frequency/severity stability and the oracle=1 local-gap
   associations. This isolates whether the observed persistent susceptibility
   generalizes beyond the one shared training sample. It has not been run.

Among the requested null interpretations, this is **not strong Case A**:
the pre-registered severity criterion fails. It has the stable-aggregate /
substantial-codebook-noise aspect of **Case B**, but not exchangeable random
failures: persistent queries and local-gap associations remain. **Case C**
is inconsistent with the very stable aggregate results here. The supported
description is mixed geometry-linked susceptibility and seed-specific
selection noise, not a rescued navigability hypothesis or a new algorithm.

## Correctness, anomalies, reproducibility and limitations

- Five codebooks differ; all training-ID and training-vector hashes match.
  Training-ID SHA256:
  `e99d821c04138d72a50bd98b5819284fa9590b5c26212cb1cd79bfc13f6062e8`.
  All full codebook/codes/model hashes are in `seed_*/manifest.json`.
- Seed0 reproduces original codebooks, all 1M codes, every candidate ADC
  score, every shared Phase 3C per-query metric and exact descriptor.
- Graph topology and the serialized graph file remain unchanged. No search
  is run. All source chunks/metadata, exact scores/GT/ID order are hashed
  before and after; all five scorers use the same unique ordered ID arrays.
- Scalar versus batch4 ADC agrees for every complete batch; deterministic
  re-encoding checks cover 1001 IDs/model, including both database endpoints.
- Strict-inversion masks and loss accounting match tiny exhaustive tests;
  all 23 Python tests and all 5 existing CTests pass.
- Independent re-derivation from saved candidate scores reproduces all 45
  derived/table/figure files byte-for-byte, including the seven SVG figures
  and post-primary diagnostic. No FAISS search or training is needed for
  that regeneration; model files and source hashes are independently checked.
- 171 queries have exact boundary ties; 145–176/model have PQ boundary ties.
  Excluding exact-boundary-tied queries leaves median Jaccard .67177 and
  loss Spearman .41728, not a change in H2's conclusion.
- The .2 unassigned pair-allocation debit consists of query1352 under models
  0 and 4: exact boundary ties, not a candidate/control anomaly. Exact-tied
  replacements are retained in pair logs but excluded from strict inversions.
  GT-intruder credits total -8.5 across all 50,000 query-model instances;
  these do not produce any negative net query-model loss.
- Shared training sample, PQ subspace layout and SIFT preprocessing limit
  generalization. Only five initialization draws were tested, all on one
  graph/query/database realization. No significance or novelty claim.
- Execution base Git: `71cb5be96e7cd062127282b48c715841d55d9ff5` with dirty
  sources, separately hashed. Host rwcpu8.cse.ust.hk, Linux
  5.14.0-687.24.1.el9_8.x86_64, GCC11.5.0; NumPy1.23.5/Matplotlib3.9.4.
  Existing filesystem clock-skew warnings appeared during build; compilation
  and linking completed. New executable preserves the old Phase 3C binary.

Raw: `runs/phase3d_quantizer_seed_stability_v1/` (including full models and
scores, per-query data, all replacement/union inversion pairs). Tables:
`results/tables/phase3d_*`; seven figures: `results/figures/phase3d_*`.
Commands/config are in `docs/experiment_log.md`. Independent re-derivation
and verification status is recorded in the run's `final_verification.json`.
