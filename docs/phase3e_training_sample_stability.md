# Phase 3E: training-sample versus initialization stability

## Pre-registered score-only protocol

The balanced design and unchanged H1–H5 were registered before new training
in `runs/phase3e_training_sample_stability_v1/preregistration.md`; configuration
is `configs/indexes/phase3e_training_sample_stability.conf`. No thresholds
were selected from outcomes. Historical Phase 3D O1..O5 are secondary only.

All 10,000 SIFT1M queries, 1M database vectors, dimension128, exact-FP32 HNSW
M16/efConstruction80/graph seed20260907; candidate provenance efSearch64/k10.
Use the identical 10,353,047 saved unique L0 FP32-traversal candidate records,
query/ID order, exact squared-L2 scores and provided GT from Phase 3C. There
is no new graph search, graph build or preprocessing. Graph fingerprint:
`1b3c18f6e2dd3ccec56e74eed30458f60a65f01234bf51478af25d84b5c02d16`.

Three independent ordered 65,536-ID FAISS rand_perm samples from the same
SIFT1M database source, without replacement within a draw, using seeds
A80400011/B80400037/C80400059. Natural overlaps are allowed and recorded.
Within each subset, all3 models train on byte-identical vectors/order.
Initialization seeds:

| Subset | Model1 | Model2 | Model3 |
|---|---:|---:|---:|
| A | 80500011 | 80500029 | 80500047 |
| B | 80500063 | 80500081 | 80500103 |
| C | 80500121 | 80500139 | 80500157 |

These are nine independent initializations nested within three subsets,
not three common seeds crossed with all subsets. All nine models are new.
PQ64x8/Train_default/RANDOM, niter25/nredo1, min39/max256 points per centroid,
no polysemous/spherical/integer/frozen centroids/faster subsampling. The
explicit sample size avoids a second internal FAISS subsampling. Encode all
1M vectors in original ID order; score saved candidates only. Training24
threads, scoring1, OPENBLAS_NUM_THREADS=1, pinned FAISS
`20f14b31a6d54e243a3d1de6ae193fc4c3ec18ed`.

## Descriptive accounting, not formal variance components

For one query L[s,i], define subset means mu_s and overall mean mu. Use:

```
W_pop = mean_si (L_si - mu_s)^2
B_pop = mean_s  (mu_s - mu)^2
T_pop = mean_si (L_si - mu)^2 = W_pop + B_pop
```

W_pop is observed within-subset initialization variability; B_pop is observed
variation of subset means. Both use population denominators. B_pop is NOT
pure training-sample variance: each subset mean averages only3 initializations.
Record W_sample=mean of three ddof1 within variances and B_sample=ddof1
variance of subset means, plus signed C=B_sample-W_sample/3, as a
noise-floor diagnostic, not random-effects inference. Negative C is retained.
Center before variance evaluation to make exactly constant inputs exactly
zero; no genuine variability is clipped. Zero/negative diagnostic count
tolerance is1e-14 recall-squared; all raw values remain preserved.

Report distributions/means and ratio-of-sums shares, not mean per-query
ratios. Between-query variance of mean loss is distinguished from within-query
model variability, but not named a causal 'geometry variance'. One dataset,
fixed PQ layout, shared source and finite draws leave common mechanisms
unidentified. Pairwise comparisons share models/queries and are not
independent samples for significance testing.

## Metric and grouping conventions

Reuse Phase 3D metrics unchanged: stable score ordering with first-recorded
ID order breaking ties; exact oracle .95315 (not native tie-different .95319).
Strict cross-boundary inversion requires exact_i<exact_j AND PQ_i>PQ_j;
maximum violation considers all A x B opportunities, including exact ties.
Signed losses, replacements and all model/subset masks are retained. Random
pair rates reuse the same512 PCG64 pairs/query from Phase 3C; global MAE is
query-weighted. SDs for model quality/pair distributions use ddof1.

Query categories: robust<=2/9 harmful, model-specific3..5/9, usual6..8/9,
universal9/9. This 'robust' definition is deliberately different from 3D.
Subset-persistent means >=2/3 harmful models in all three subsets and is an
overlapping diagnostic, not a fifth disjoint category. Query subset frequency
uses >=2/3; pair subset frequency instead uses >=1 inverted model/subset.

Geometry uses only exact d1/d10/d11/d20, gaps d(k+r)-d(k-r+1), r1/2/5,
relative d11-d10 divided by max(d10,1e-12), (d20-d1)/max(d1,1e-12), pool size
and oracle recall. All-query and oracle=1 comparisons are prespecified;
quantile bins preserve tied descriptor values. Correlation with averaged
susceptibility is compared with individual realized loss, recognizing that
averaging itself suppresses noise. No fitted predictor or algorithm.

## Observations: sample/model identity and aggregate quality

All nine models trained/encoded/scored successfully. The sample-ID SHA256s
are identical within each subset and distinct between subsets:

- A: `c984c6530b9675be22273bd307b5bf3eecbb9a98ce4b675cccb0b33a15542c5c`
- B: `66288f864fa893d7b910efafd427ff04e362f2e09ecd12522078224d4cfe1d2c`
- C: `d21682b14af02018ee5d9d889dd07044aac7a9787965be7d5c36b7c9e56133d3`

A/B, A/C and B/C share4206/4288/4356 IDs respectively, near the independent
sampling expectation4294.97. These are independent draws, not disjoint data.
All9 codebook hashes differ; full codebook/codes/index hashes and training
IDs are preserved per model. No model was removed or normalized.

| Model | Recall@10 | Ranking loss | Critical inversions/query | Global MAE | Random inversion rate |
|---|---:|---:|---:|---:|---:|
| A1 | .85364 | .09951 | 5.5562 | 1309.709 | .0344393 |
| A2 | .85273 | .10042 | 5.5446 | 1318.945 | .0345602 |
| A3 | .85234 | .10081 | 5.6048 | 1309.062 | .0345096 |
| B1 | .85255 | .10060 | 5.5427 | 1309.649 | .0343338 |
| B2 | .85319 | .09996 | 5.4616 | 1310.004 | .0344229 |
| B3 | .85299 | .10016 | 5.5113 | 1313.714 | .0344213 |
| C1 | .85303 | .10012 | 5.5884 | 1314.960 | .0345648 |
| C2 | .85245 | .10070 | 5.5426 | 1314.611 | .0342824 |
| C3 | .85154 | .10161 | 5.5852 | 1313.630 | .0345080 |

| Metric over all9 models | Mean | Sample SD | Min | Max |
|---|---:|---:|---:|---:|
| Recall | .8527178 | .00059947 | .85154 | .85364 |
| Ranking loss | .1004322 | .00059947 | .09951 | .10161 |
| Critical inversions | 5.54860 | .04357 | 5.4616 | 5.6048 |
| Global MAE | 1312.6982 | 3.3244 | 1309.0621 | 1318.9454 |
| Random inversion rate | .03444913 | .00009714 | .03428242 | .03456484 |

Within each subset, recall mean/SD/range:

| Subset | Mean recall | Init SD | Min | Max |
|---|---:|---:|---:|---:|
| A | .8529033 | .0006671 | .85234 | .85364 |
| B | .8529100 | .0003274 | .85255 | .85319 |
| C | .8523400 | .0007511 | .85154 | .85303 |

Between the three **subset means**, recall SD is .0003272, range
.8523400–.8529100. This is not SD of individual model quality. Mean loss
is .1002467/.1002400/.1008100 for A/B/C; same within SDs as recall.
Critical-inversion means are5.56853/5.50520/5.57207 with within-init
SDs .03194/.04089/.02557; their between-subset-mean SD is .03763.
MAE means1312.572/1311.122/1314.400, within SDs5.529/2.251/.690,
between-mean SD1.643. Random inversion-rate means .0345030/.0343926/.0344518,
within SDs .00006072/.00005098/.00014938, between-mean SD .00005522.
Full min/max for every within/between metric are in
`results/tables/phase3e_aggregate_variation.*`. Similarity of aggregate
quality does not mean all metrics have identical sample/init variation.

## Observations: within versus between subset stability

| Metric | Within median (9 pairs) | Between median (27 pairs) | Within minus between |
|---|---:|---:|---:|
| Harmful-query Jaccard | .66836 | .66060 | .00776 |
| Ranking-loss Spearman | .41282 | .39313 | .01969 |
| Critical-count Spearman | .44764 | .41674 | .03090 |
| Inversion-pair Jaccard | .15871 | .14264 | .01607 |

Harmful Jaccard ranges are .66139–.67745 within and .65289–.66950 between;
their sample SDs are .00519/.00414. Loss Spearman ranges
.40366–.42907 within and .38048–.41769 between, SDs .00854/.00853.
Thus H2's direction holds, but changing sample does **not** collapse
query-level stability. The median Jaccard decrease is .776 percentage
points, not a claim that .776% of failures are caused by shared samples.
Pair distributions overlap and are dependent; no significance inference.

The oracle=1 sensitivity gives within/between Jaccard .72910/.71664 and
loss Spearman .36619/.33171. Excluding171 exact-boundary-tied queries gives
.66899/.66103 Jaccard and .41237/.39148 loss Spearman. Neither control
turns the moderate sample effect into training-sample-dominated failure.

### Secondary O group (not in the balanced design)

Original O1..O5 pass metadata, candidate/source/model integrity checks;
100 deterministic queries/model are independently re-measured from saved
O scores, matching all single-model metrics. Their original mean recall
.853052 remains close to ABC's .852718. O-within Jaccard/loss-Spearman
medians .67116/.41848 become .66149/.39418 for45 O-to-ABC pairs. This
separate check has the same direction as ABC; it is not added to primary
variance estimates, frequency denominators or ensemble composition.

## Observations: per-query variability

All10,000 queries satisfy the pre-registered W_pop+B_pop=T_pop identity.
Means across queries (recall-squared units):

| Descriptive quantity | Mean |
|---|---:|
| Within-subset W_pop | .0027245926 |
| Observed subset-mean B_pop | .0010177778 |
| Total model T_pop | .0037423704 |
| Bessel-corrected W_sample | .0040868889 |
| Bessel-corrected B_sample | .0015266667 |
| Signed contrast B_sample−W_sample/3 | .0001643704 |

The observed partition is **72.80% within / 27.20% between subset means**.
It is wrong to call27.20% the causal training-sample share: the three-init
means retain sampling noise. W_sample/3=.0013622963 is the simple
initialization-mean noise-floor comparison; the signed excess .0001643704
is about4.02% of W_sample. This contrast suggests a smaller sample-linked
component than initialization variation under the stated illustrative
independence assumptions, not a formal variance estimate or confidence claim.
4522 queries have negative contrasts and430 have zero total model variance
(count tolerance1e-14); signed per-query values are preserved.

Variance of per-query mean loss is .0033511095; mean within-query model
variance is .0037423704. Together they equal .0070934799 over all query/model
entries (47.24% between-query means /52.76% within-query models). The former
contains shared PQ-layout/data effects and finite-model averaging noise;
it is **not an identified geometry percentage**. These figures prevent
equating a stable mean susceptibility with a fixed realized error.

## Observations: vulnerability across both random sources

| Category | Queries | Fraction | Fraction of total signed loss |
|---|---:|---:|---:|
| Robust (<=2/9 harmful) | 1103 | 11.03% | 1.33% |
| Model-specific (3–5/9) | 1953 | 19.53% | 10.17% |
| Usually harmful (6–8/9) | 4503 | 45.03% | 47.62% |
| Universally harmful (9/9) | 2441 | 24.41% | 40.89% |

The overlapping **subset-persistent** population is **5676 queries (56.76%)**,
contributing **77.59%** of total loss. In the complete-oracle stratum it is
4851/7136 (67.98%), contributing82.62% of that stratum's loss. Universal
harmfulness remains2269/7136 (31.80%) there. There is no negative net
query/model ranking loss in this run; no value was clipped.

Harmful-model counts0..9 have350/351/402/515/578/860/1183/1442/1878/2441
queries. Subsets with a harmful majority0..3 have1141/1151/2032/5676 queries.
Simple independent marginal baselines predict433.1 universal queries and
4075.8 subset-persistent queries, versus2441 and5676 observed. The latter
uses observed subset-majority prevalences, not independent-initialization
assumptions within subset. These are descriptive homogeneous-query baselines,
not p-values; common oracle/geometry and PQ architecture are not removed.

Do not compare11.03% 'robust' directly with Phase3D's6.47%, because its
definition changed. Likewise9/9 is stricter than5/5; the smaller universal
fraction alone does not show that susceptibility disappeared. This remains
a broad susceptibility pattern, not a rare heavy-tail finding.

## Observations: inversion pairs

The union has243,988 distinct (query, exact-top-k item, outsider) strict
inversion pairs:

- Unique to one model:123,538 (**50.63%**).
- Confined to one training subset:141,593 (**58.03%**).
- Recurring in at least2 subsets:102,395 (**41.97%**).
- Recurring in all3 subsets (>=1 model each): **14.21%**.
- Inverted by all9 models:371 (**.1521%** of union).
- Repeated in >=2 initializations of some same subset: **30.47%**.

All joint model-count/subset-count cells and within-subset majority/all3
repetition counts are saved, rather than conflating these criteria. A pair
can occur once in each subset without repeating within any subset. Pair
Jaccard decreases .15871 to .14264 across subsets, but remains much lower
than query harmful-set Jaccard .66060. Pair and query supports have different
prevalence, so this numerical contrast alone is not a normalized effect
size. The rarity of all9 persistent pairs versus24.41% universal queries
supports H4 descriptively: persistent query loss need not use the same pair.

## Observations: exact geometry and mean versus realized harm

Raw categories differ in mean oracle recall: .83146 robust, .92616
model-specific, .97353 usual, .99213 universal. Oracle recall correlates
.47159 with all-model harmful frequency, so the complete-GT control matters.

For the7136 oracle=1 queries:

| Category | Queries | Median relative d11−d10 | Median d12−d9 | Median d15−d6 |
|---|---:|---:|---:|---:|
| Robust | 306 | .039845 | 3397.5 | 7421.5 |
| Model-specific | 1014 | .016772 | 2135 | 5665 |
| Usually harmful | 3547 | .007718 | 1262 | 4265 |
| Universally harmful | 2269 | .004598 | 776 | 3172 |
| Subset-persistent (overlapping) | 4851 | .005803 | 978 | 3643 |
| Not subset-persistent | 2285 | .014442 | 1968 | 5464 |

Spearman coefficients in the same complete-GT stratum:

| Exact descriptor | Harmful frequency | Subset-majority frequency | Mean loss | B_pop | W_pop |
|---|---:|---:|---:|---:|---:|
| Relative boundary margin | −.44620 | −.40690 | −.46377 | −.10200 | −.15845 |
| d12−d9 | **−.55815** | −.49013 | **−.66865** | −.18185 | −.32366 |
| d15−d6 | −.45522 | −.39470 | −.57736 | −.18858 | −.31545 |
| Candidate count | .00435 | .03273 | .02544 | .05378 | .07151 |
| d1 | .05880 | .05640 | .06952 | .04728 | .05433 |

For d12−d9, correlation with mean loss is−.66865 versus mean single-model
realized-loss correlation−.43060 (range−.44816 to−.42150). For relative
boundary margin these are−.46377 versus−.30411; for d15−d6,−.57736 versus
−.37297. The all-query d12−d9 comparison also strengthens from mean realized
−.35711 to mean-loss−.53006. Thus local geometry is more associated with
mean susceptibility than the exact realized loss. Averaging suppresses
noise; these correlations do not establish a predictive causal explanation
or an intrinsic hard-query class. The complete-GT robust group has306 queries.
All descriptors/targets are reported, including weak/reversed d1 effects
(all-query d1-to-frequency−.14903, oracle=1+.05880).

## Post-primary fixed ensemble diagnostics

Primary and secondary tables were frozen before either diagnostic. No
composition/weight selection, performance optimization or benchmark:

| Three-score diagnostic | Member-average recall | Diagnostic recall | Member-average loss | Diagnostic loss | Critical inversions | Fraction member loss recovered |
|---|---:|---:|---:|---:|---:|---:|
| A1/A2/A3 (initialization) | .8529033 | .88588 | .1002467 | .06727 | 2.8399 | 32.90% |
| A1/B1/C1 (cross-subset) | .8530733 | .89097 | .1000767 | .06218 | 2.5363 | 37.87% |

The cross-subset diagnostic has .00509 higher recall, whereas its
member-average baseline advantage is only .00017. In this fixed comparison,
sample diversity gives additional error cancellation, consistent with a
nonzero shared-sample component. Only these two triplets were tested; they
share A1 and are not an estimate over all compositions. The result does
not turn ensemble PQ into a proposed method, or contradict the much larger
observed initialization variability.

## Interpretation and H1–H5

- **H1 supported descriptively:** aggregate quality is broadly comparable
  across these independently sampled subsets; nine-model recall range
  .85154–.85364. No post-hoc stability threshold or normalization.
- **H2 direction supported:** within similarity exceeds between similarity,
  but the query-level reduction is modest, not a breakdown of stability.
- **H3 supported as susceptibility evidence:**56.76% subset-persistent and
  24.41% universal; complete-GT controls retain a large population. Not a
  statement that exact geometry is the only shared source of stability.
- **H4 supported descriptively:** exact pair identities are substantially
  less persistent, with50.63% one-model pairs and only371 all-nine pairs.
- **H5 supported descriptively:** narrower multi-item exact local gaps are
  associated with higher mean harmfulness, including oracle=1; not causal
  prediction or a fitted classifier.

**Best supported interpretation: Case C — mixed mechanism.** There is a
stable geometry-related susceptibility component and substantial realized
codebook noise, with a smaller but observable training-sample effect. Several
Case A descriptive features hold (persistent queries, local-gap associations,
unstable pair identities), but they do not identify geometry as dominant:
shared PQ axes/layout and finite-model noise in query means remain untested,
and per-query model variability is substantial. Case B is not supported:
changing training sample does not collapse harmful-query overlap, the
subset-persistent population is large, and the sample contrast is much
smaller than within-init variance. This is not evidence rescuing navigability.

### Answers to the seven final questions

1. **How much stability comes from shared samples?** A modest observed
   increment: within-to-between median Jaccard drops .00776 and loss
   Spearman .01969. A nonzero sample component is consistent with the
   signed variance contrast and cross-subset diagnostic advantage, but no
   exact causal percentage is identified.
2. **Does vulnerability survive independent subsets?** Yes descriptively:
   56.76% are subset-persistent, carrying77.59% of loss; oracle=1 retains
   67.98% subset-persistence. This is susceptibility, not a discovered class.
3. **Sample variation larger or smaller than initialization?** Smaller for
   per-query ranking loss under the declared accounting: W_pop .002725
   versus observed B_pop .001018, and signed sample contrast .000164 versus
   W_sample .004087. B_pop contains finite-init noise; do not treat27.20%
   as the pure sample share. Other aggregate quality metrics can have
   different relative variation, as the full table shows.
4. **Are critical pairs stable across subsets?** Mostly not:58.03% confined
   to one subset;41.97% recur across>=2,14.21% across all3; all-nine pairs
   are exceptionally rare (.1521%).
5. **Does geometry explain mean susceptibility better than realization?**
   It is more strongly associated: d12−d9 mean-loss Spearman−.669 versus
   average realized-loss Spearman−.431 in oracle=1. This is exploratory
   association; averaging and selection of a complete-GT stratum matter.
6. **Which case?** Case C, with a strong repeatable susceptibility signal,
   predominantly initialization-scale realization variation, and a smaller
   sample-linked effect; neither pure geometry nor training-sample dominance
   is justified.
7. **Exactly one next experiment (not run):** an **orthogonal-coordinate
   control** on these same frozen candidates. Apply one deterministic,
   non-learned random orthogonal rotation to database and queries; use
   standard PQ64x8 with the same ABC sample IDs/initialization seeds, score
   only the saved pools, and compare harmful-frequency persistence and
   local-gap associations with the unrotated run. Keep original exact
   distances/GT as the invariant reference and validate rotated exact L2
   numerical equivalence first; report PQ approximation quality without
   tuning it. Exact Euclidean geometry is preserved while its alignment
   with PQ subspaces changes. This distinguishes intrinsic distance-margin
   susceptibility from shared coordinate/subspace alignment, without a
   learned/optimized rotation, new quantizer or algorithm proposal.

## Correctness, reproducibility, limitations

- All9 source/model/sample manifests validated; source candidate chunks,
  graph file, queries/GT/exact scores and old O artifacts remain unchanged.
  Every graph fingerprint matches the frozen fingerprint before/after.
- All35 Python tests (12 Phase3E references,23 prior) and all5 CTests pass.
  Tests cover variance decomposition/pure-factor/constant/negative-contrast
  cases, nine-model pair masks, differing model/subset frequency semantics,
  stable exact ties, no frozen-array mutation and9/27 primary pair counts.
- Independent re-derivation of all61 derived/table/figure files is
  byte-identical, including the8 SVG figures and both diagnostics. Model,
  training-sample, executable and frozen-source checks pass; the final audit
  records174 artifact hashes without replacing any earlier result.
- Every query reuses the same512 random pairs and Phase3C exact geometry;
  historical O source metrics are directly audited on100 queries/model.
- Every full scalar/batch4 ADC batch agrees. Re-encoding checks cover1001
  deterministic IDs/model, including database endpoints; complete databases
  are added in their original ID order.
- No negative net query-model loss observed, but signed fields retained.
  There are171 exact boundary-tied queries and149–187 PQ boundary ties/model.
  Tied replacements remain in raw data; strict inversions remain strict.
- Shared dataset, overlapping independently drawn samples, one graph/query
  realization, fixed PQ architecture/layout and just three subset draws
  limit generalization. No formal random-effects, significance or novelty
  claims. Independent subset draws do not mean disjoint vectors.
  A training subset includes its recorded vector order; this experiment
  does not separately identify membership versus ordering effects.
- Execution Git `913d7eddbe5f6f6e7b83efa7f2244de1bc2bb246` with dirty
  source/binary/config hashes; host rwcpu8.cse.ust.hk, Linux
  5.14.0-687.24.1.el9_8.x86_64, GCC11.5.0; NumPy1.23.5/Matplotlib3.9.4.
  Existing filesystem clock-skew build warnings recur; compilation/linking
  completed. Matplotlib emitted a deprecated `boxplot(labels=...)` warning;
  figures completed under the pinned local environment, without changing
  numeric analysis. Separate build target preserves old export binaries.

Raw data/full models: `runs/phase3e_training_sample_stability_v1/`.
Summary tables: `results/tables/phase3e_*`; eight useful figures:
`results/figures/phase3e_*`. Exact commands/configs are in
`docs/experiment_log.md`. Independent raw-data regeneration and final hash
audit are recorded in `final_verification.json` under the run.
