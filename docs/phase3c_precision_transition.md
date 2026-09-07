# Phase 3C: quantization precision on one exact-traversal candidate pool

## Pre-registration — before exporting Phase 3C scores

H1. PQ64 has lower fixed-candidate ranking recall loss than PQ32.

H2. Recall improvement from PQ32 to PQ64 is much more strongly associated
with reduction in cross-boundary critical inversions than reduction in global
MAE or random pairwise inversion rate.

H3. A nontrivial subset of queries harmful under PQ32 becomes correct under
PQ64 (fully rescued queries).

H4. Persistently harmful queries have systematically more fragile exact local
top-k geometry than stable queries.

H5. Most corrected PQ32 critical inversions involve relatively small exact
cross-boundary margins.

No significance thresholds will be invented after results. Report numerical
associations, distributions, regressions and evidence against these claims.

## Protocol and interpretation conventions fixed before results

- SIFT1M, all 10,000 queries, k=10, efSearch=64. Load the Phase 3A serialized
  FP32 graph and both existing PQ32x8/PQ64x8 indexes; do not build or train.
  Generate the primary candidate sets once using existing corrected L0
  recording of **FP32 traversal**. Native replay is only a correctness check.
  All three score columns share the same candidate-ID column and per-query
  ID hash. Score functions cannot add or remove candidates.
- Squared L2 throughout. Exact and PQ rankings use independent stable sorts
  with first unique L0 evaluation order breaking exact score ties. Do not
  substitute native IDs. For the exact native control, require equality of
  the sorted FP32 distance multiset and explicitly report any tie-only ID/GT
  difference. This is different from Phase 3A's native-compatible exact tie
  realization and must not overwrite that metric.
- Compute signs using integer GT-hit differences. The user's categories
  Stable (both losses zero), Rescued (loss32>loss64), Persistent (both losses
  positive), Regressed (loss64>loss32) overlap: Rescued/Persistent and
  Regressed/Persistent can coincide. Report those definitions as overlapping
  flags and intersections, not a stacked partition. Separately report fully
  rescued (loss32>0, loss64=0), partially improved persistent, unchanged
  harmful, regressed, and any other signed/tie cases.
- Harmful means positive recall loss. Harmful-set overlap is distinct from
  membership disagreement. Top 10/20/50% query overlap uses descending loss,
  then ascending query ID at ties, with exactly ceil(p*n) queries. Report
  overlap/count-normalized overlap and Jaccard, cutoff tie sizes, and a
  tie-inclusive sensitivity analysis. These recall scores are discrete.
- Boundary window ranks 5..15; gaps r=1,2,5. Geometry includes d1,dk,dk+1,d20,
  relative gap, candidate count, and (d20-d1)/max(d1,epsilon). Candidate-wide
  relative MAE is mean(|error|/d_exact) over d_exact>epsilon, with omitted
  denominator counts recorded. Report each change as metric32-metric64.
- Sample the same 512 distinct-item random candidate pairs for both PQs,
  PCG64(seed=60400001+query_id), with replacement across draws. Record sample
  indices. Cross-boundary pairs are exhaustive, with strict exact and PQ
  inequalities; exact ties are retained in the denominator, not inversions.
- Preserve every pair inverted by either PQ, labeling corrected/persistent/
  new. Preserve all selected displaced/intruder pairs for both scorers,
  regardless of recall harm. Record exact ranks, signed errors and violations.
  "Intruder" means an actually PQ-selected item outside T_exact. Separately
  report all outside items that overtake an exact top-k item; do not conflate
  these counts with selected intruders.
- H5 operational context: report relative exact pair margin / dk at cutoffs
  .01, .05 and .10; use <=.05 as the pre-declared "relatively small" description
  and >.10 as a large-margin comparison. Also report raw margins and whether
  inside rank>=5 and outside rank<=20. Pairwise small margins and proximity in
  rank are different claims. Compare corrected pairs to persistent/new pairs
  and to the full A×B opportunity distribution. Give pair-weighted summaries
  and query-weighted summaries so high-inversion queries cannot dominate
  unnoticed.
- Associations use Spearman, binned conditional means and descriptive SEs.
  Critical inversions and displaced counts are post-scoring and structurally
  linked to ranking loss, not independent predictors. PQ32 and PQ64 are
  separately trained, non-nested codebooks with different subvector sizes;
  this is not a monotonic per-query error guarantee or bitwise refinement.
- After completing the primary analysis, perform one optional control on the
  previously recorded Phase 3B PQ64 ef64 candidate pools for query IDs
  0,10,...,9990 (1,000 queries). Reuse those recorded IDs without new PQ
  traversal. Compare the primary's same 1,000 queries as well as the full
  primary result, and keep control tables separate.

Config: `configs/indexes/phase3c_precision_transition.conf`. Preserve all prior
runs; derivations must use new output directories. Raw candidate scores alone,
plus the preserved GT/query metadata, must regenerate all scientific outputs.

## Primary checkpoint, completed before optional control

All 10,000 exact-traversal pools were exported once (10,353,047 candidates).
Oracle recall=0.95315, PQ32 recall=0.70398, PQ64 recall=0.85313. Mean loss falls
from 0.24917 to 0.10002; mean critical inversions from 33.3636 to 5.5846.

Overlapping Stable/Improved/Persistent/Regressed flags have counts
255/7871/6906/408. Fully rescued queries number 2731 (27.31% of all queries).
P(harmful64|harmful32)=0.71661 and harmful-query Jaccard=0.70867.

Spearman correlation of loss recovery with reduction in critical inversion
count=0.60204, versus global MAE=0.05028 and random inversion rate=0.08292.

Of 333,636 PQ32 inversions, 310,999 are corrected and 22,637 persist; 33,209
new inversions appear only under PQ64. Corrected-pair relative margin median
is 0.09449; 19.87% have relative margin<=0.05 and 46.54% exceed 0.10. Persistent
pair relative margin median is 0.02610, with 81.02% at most 0.05. H5 is not
supported under its pre-declared small-margin interpretation. This is an
explicit negative result, not a reason to change the cutoff.

The primary tables and this checkpoint are preserved in `primary_checkpoint/`
before preparing the optional 1,000-query recorded-PQ64-pool control.

## Frozen inputs and scoring implementation

SIFT1M files, their SHA-256 checksums, source revision, and dataset/graph/PQ
seeds are inherited unchanged from `configs/indexes/faiss_hnsw_phase3a_sift1m.conf`
and copied to each raw condition directory. There are 1,000,000 base vectors,
10,000 queries, d=128, and k=10. The existing 100-query exhaustive FP32 GT
validation from Phase 3A is reused; every query's GT IDs match the prior run.

The serialized M16/efConstruction80 graph (seed 20260907) and PQ32x8/PQ64x8
models (seed 30260907) are loaded from
`/rwproject/kdd-db/kluaq/dataset/sift1m/phase3a_cache_v1/`. The primary exporter
calls existing `search_with_level0_recording` once with FP32 storage and
efSearch=64, then applies both FAISS PQ distance computers to those saved IDs.
It uses bounded_queue=true, check_relative_distance=true, one search thread.
No quantized traversal runs in the primary experiment. Native FP32 replay is
the existing instrumentation-identity control, not a second candidate pool.

The graph fingerprint remains
`1b3c18f6e2dd3ccec56e74eed30458f60a65f01234bf51478af25d84b5c02d16`.
PQ codebook/code hashes match the committed Phase 3A values in the new config
before and after scoring. All three scoring conditions have identical per-
query candidate-order hashes, and the binary representation has one shared
ID column: int32 ID, float32 exact squared L2, float32 PQ32 ADC, float32 PQ64
ADC. Scalars and batch-4 ADC agree for both models. Candidate counts and the
previously retained first 100 exact candidate sets match Phase 3A.

Execution HEAD was `170cf03a9f20ba1629dce2ed905e3ab06df15600`, with Phase 3C
changes uncommitted; FAISS remained
`20f14b31a6d54e243a3d1de6ae193fc4c3ec18ed`. Export manifests contain executable
and source hashes, machine/compiler details, and frozen-state checks.
Machine: rwcpu8.cse.ust.hk, Linux 5.14.0-687.24.1.el9_8.x86_64, GCC 11.5.0.
Analysis uses NumPy 1.23.5 and Matplotlib 3.9.4. Sampled candidate pairs use
PCG64(seed=60400001+query_id), identical under both scorers.

## Primary precision response

| Score | Mean Recall@10 | Mean ranking loss | Mean critical inversions | Mean displaced items |
|---|---:|---:|---:|---:|
| Exact | 0.95315 | 0 | 0 | 0 |
| PQ64x8 | 0.85313 | 0.10002 | 5.5846 | 1.2018 |
| PQ32x8 | 0.70398 | 0.24917 | 33.3636 | 2.7749 |

Switching the scorer on this exact same pool improves recall by 0.14915,
reduces mean loss by 59.86%, and reduces mean critical-inversion count by
83.26%. This supports H1 for the two frozen models. It is not a statement
that adding bits to a nested representation monotonically improves every
query: these models use different subvector partitions and independently
trained codebooks.

| Global/local score control | PQ32 mean | PQ64 mean |
|---|---:|---:|
| Candidate-wide MAE, squared-L2 units | 3312.205 | 1315.332 |
| Mean absolute relative error | 0.04233 | 0.01699 |
| Random candidate-pair inversion rate | 0.08453 | 0.03438 |
| Boundary-window MAE | 2737.971 | 990.499 |
| Maximum cross-boundary violation | 6026.266 | 1195.247 |

Mean controls are means of per-query measurements, not candidate-count-
weighted pooled errors. Full distributions and signed changes are in the
summary JSON and per-query records.

## Query transitions: improvement is not the same as full rescue

| User-defined flag | Count | Fraction |
|---|---:|---:|
| Stable under both: both losses zero | 255 | 2.55% |
| Rescued/improved: loss32 > loss64 | 7871 | 78.71% |
| Persistently harmful: both losses positive | 6906 | 69.06% |
| Regressed: loss64 > loss32 | 408 | 4.08% |

These flags overlap. There are 5,140 improved-and-persistent queries and 300
regressed-and-persistent queries. Fully rescued queries number 2,731: 27.31%
of all queries, or 28.34% of the 9,637 queries harmful under PQ32. Another
1,466 remain harmful with equal loss. No negative-loss or otherwise uncovered
case occurred; the implementation retains signed outcomes and supports them.
H3 is supported, but most PQ32-harmful queries are not made fully correct.

9,382 queries have fewer inversions, 204 have equal counts, and 414 have
more. All 408 recall regressions occur despite lower candidate-wide MAE;
169 also have fewer total critical inversions. Thus a reduction in the total
number of inversions is not sufficient to guarantee recall improvement for
one query: which GT items cross the selection boundary still matters.

## Which metric changes track recall recovery?

Spearman correlation of metric32-metric64 with loss32-loss64:

| Change in metric | Primary exact pool | Optional PQ64 pool |
|---|---:|---:|
| Global candidate MAE | 0.0503 | 0.0640 |
| Absolute relative error | 0.1114 | 0.0924 |
| Random pairwise inversion rate | 0.0829 | 0.0965 |
| Boundary-window MAE | 0.2136 | 0.2577 |
| Critical-inversion count | 0.6020 | 0.6080 |
| Critical-inversion fraction | 0.5958 | 0.5964 |
| Maximum violation | 0.3713 | 0.3826 |
| Displaced top-k count | 0.9257 | 0.9268 |

H2 is descriptively supported. Generic error quality improves substantially
in aggregate but barely tracks *which* queries recover recall. Critical
inversion reduction tracks recovery much better, while still leaving many
exceptions. Displaced-count reduction is closest to the outcome by definition;
its high association is not evidence of an independent predictor. All these
are post-scoring measurements. Random inversion rates have finite-sample
noise (512 pairs/query), which can attenuate their correlations.

## Persistence and exact geometry

P(harmful64 | harmful32)=71.66%; P(harmful32 | harmful64)=98.46%; harmful-set
Jaccard=0.70867. These sound large, but the marginal harmful fractions are
96.37% and 70.14%. As exploratory prevalence context, independent sets with
these sizes would have an expected intersection of 6,759.4, compared with
6,906 observed. Most membership persistence can therefore occur without a
small, distinctive intrinsic hard-query class.

With descending loss and query-ID tie breaking, top 10/20/50% loss-set overlap
is 32.0%/35.8%/61.42% of each selected set (Jaccard .1905/.2180/.4432).
Cutoff ties are large: the top-10% cutoffs tie 1,397 PQ32 queries and 1,916
PQ64 queries. Tie-inclusive sets have Jaccard .2373/.2974/.6577 for the three
cutoffs, but different sizes. These are not identical query populations and
do not prove intrinsic vulnerability.

Raw group geometry:

| Median | Stable both (n=255) | Fully rescued (n=2731) | Persistent harmful (n=6906) |
|---|---:|---:|---:|
| Relative d11-d10 margin | .00867 | .00876 | .00589 |
| d15-d6 gap, squared L2 | 6092 | 4724 | 3789 |
| d1 | 42097 | 41503 | 36803.5 |
| d10 | 58616 | 55217 | 49669.5 |
| d20 | 63903 | 59239 | 53017 |
| (d20-d1)/d1 | .4058 | .3553 | .3514 |
| Candidate count | 1093 | 1112 | 1063 |

An important confounder emerged: mean oracle recall differs markedly between
these groups (.81804/.91366/.97388 respectively). "Stable" here means zero
ranking recall loss; it does not mean a complete GT pool or identical top-k
membership. Raw geometric differences must not be assigned an intrinsic
query explanation without considering this.

**Post-hoc sensitivity, added after observing that confounder:** restrict only
this descriptive geometry comparison to queries with oracle recall exactly 1.
The corresponding stable/fully-rescued/persistent counts are 82/1423/5559.
Median relative margins are .02657/.01377/.00656 and d15-d6 gaps are
8771/5196/3903. The direction of local-gap differences remains and is larger,
but some absolute-distance comparisons reverse (stable median d1 becomes
20579.5 versus persistent 34318). This supports a narrower H4 observation:
persistent loss is associated with tighter local selection geometry, including
among complete-GT pools. It does not establish a stable intrinsic class or
causality; the stable complete-GT group is small and this restriction was
exploratory, not pre-registered.

## Inversion transitions: H5 is not supported as registered

| Pair state | Count | Median relative margin | Margin <=5% of dk | Margin >10% of dk | Inside rank>=5 and outside rank<=20 |
|---|---:|---:|---:|---:|---:|
| Corrected PQ32 inversions | 310999 | .09449 | 19.87% | 46.54% | 37.90% |
| Persistent inversions | 22637 | .02610 | 81.02% | 2.64% | 90.02% |
| New PQ64-only inversions | 33209 | .03394 | 71.43% | 4.38% | 84.96% |

93.22% of PQ32 inversions disappear, but 59.47% of the remaining PQ64
inversions are new pairs, not a surviving subset of PQ32 inversions. The
codebook change has substantial turnover even though the net count declines.

Corrected pairs have median exact margin 4584, median PQ32 violation +1629.30,
and median PQ64 violation -4484.86 squared-L2 units. Persistent pairs have
median margin 1220 and violations +2374.59/+775.79. New pairs have median
margin 1570 and violations -3140.49/+655.72. Median exact outside rank is
22 for corrected pairs, 13 for persistent pairs, and 14 for new pairs;
corrected outside rank p95 is 80. Precision repairs substantial wider-rank
and larger-margin failures, not only near-adjacent top-k errors.

This conclusion is not solely driven by queries with many inversions.
Query-averaged small-margin fractions are 21.32%/83.12%/74.04% for corrected/
persistent/new states; query-averaged near-rank-boundary fractions are
49.43%/94.11%/90.82%. Queries with no pair in a state are omitted only from
that state's conditional mean, with counts reported.

For context, only 0.218% of all 102,530,470 A×B opportunities have margin at
most 5% of dk. Corrected inversions are still strongly enriched for small
margins relative to that opportunity universe. Nevertheless, "most corrected
inversions have relatively small margins" is false under the declared <=5%
criterion; the raw and relative distributions, <=1/5/10% counts and rank
statistics are all retained rather than changing the definition.

The supported observation is that higher-precision scoring removes many
broad errors and leaves residual/new failures more concentrated near the
exact selection boundary. The algebraic identity
`violation_Q = d_Q(inside) - d_Q(outside)`
is implemented as `error_Q(inside)-error_Q(outside)-exact_margin`; positive
values encode the pairwise reversal, not an independent causal test.

## Optional control, separated from the primary experiment

After saving the completed primary checkpoint, query IDs 0,10,...,9990 were
selected from the previously recorded Phase 3B PQ64 ef64 pools. No PQ traversal
was rerun. Their candidate IDs, exact scores and PQ64 scores match the saved
Phase 3B records exactly; only the missing PQ32 score column was added.

| Metric | Exact pools, same 1000 queries | Recorded PQ64 pools, same 1000 queries |
|---|---:|---:|
| Oracle recall | .95420 | .95200 |
| PQ32 recall | .70680 | .70660 |
| PQ64 recall | .84960 | .84780 |
| Mean PQ32 inversions | 32.432 | 32.711 |
| Mean PQ64 inversions | 5.857 | 5.892 |
| Fully rescued fraction | 27.0% | 27.3% |
| Regressed fraction | 4.2% | 4.4% |

The optional pool gives the same qualitative metric-change correlation
ordering. Only 20.30% of corrected pairs have small margins and 45.73% have
large margins, again contradicting H5's majority claim. This is a small
paired candidate-source control, not another dataset/index replicate.

## Validation, raw schema, and limitations

- No changes to graph construction, quantizers, or traversal instrumentation.
  Exact-traversal candidate pools are independent of the scorers and exported
  once; the optional pool comes from committed prior records.
- Native FP32 search IDs match Phase 3A for all queries. Stable exact reranking
  has the same sorted score vector for every query. 78 native/stable set
  differences are exact boundary ties; 18 queries gain a GT hit and 22 lose
  one from the tie choice, yielding oracle .95315 versus native .95319. This
  is explicitly recorded and does not overwrite previous exact-control metrics.
- 171 primary queries have tied exact boundaries. Excluding them yields mean
  recall recovery .14920 and 402/9829 regressions (4.09%); this is not a
  tie-driven conclusion. Main results keep all ties and signed values.
- Seven new Python tests cover scoring control, overlapping transitions,
  regression, inversion transitions against a nested-loop reference, signed
  violation identities, retained tied replacements, shared random samples,
  selected-intruder versus all-overtaker counts, and discrete top-loss ties.
  The six Phase 3B tests and all five CTests also pass.
- The score-export files contain every candidate under every scorer. Query
  JSONL stores offsets, counts, shared hashes, GT and native exact controls.
  Analysis JSONL stores ordered top-k lists, sorted membership sets, signed
  losses, flags, geometry, score-quality controls and precision deltas.
  Gzipped JSONL preserves every selected replacement pair and every union-
  inversion pair with corrected/persistent/new labels. Random-pair positions
  are stored as little-endian int32 pairs for reproducible paired comparisons.
- Per-query derivations, replacement pairs, inversion transitions, random
  sample files, tables and SVGs can be regenerated from raw score files
  without FAISS or database access; independent regeneration is checked
  byte-for-byte. Hashes and environment versions are recorded by the final
  verification script. The host again emitted its existing build clock-skew
  warning; compilation and linking completed.
- Two non-nested codebooks, one graph and one real dataset limit external
  validity. The scorer swap is controlled; the correlations and group
  descriptors remain descriptive. Precision, partition size and learned
  centroids change together, so results do not isolate an abstract bit count.

## Decision and exactly one next experiment

1. **Higher precision systematically reduces critical inversions on this
   same pool in aggregate:** yes, by 83.26%, and in 93.82% of queries. It is
   not pointwise monotonic; new pairs and regressions remain.
2. **Does critical-inversion reduction track recall recovery better than
   generic error improvement?** Yes (.602 versus .050/.083 Spearman), but
   counts alone are incomplete, as the 169 regressions with fewer inversions
   demonstrate.
3. **Are queries mostly rescued?** Most improve, but only 28.34% of PQ32-
   harmful queries become fully correct. Raw harmful-set persistence is high
   partly because almost all queries are harmful under PQ32; intrinsic
   vulnerability is not established.
4. **What geometry distinguishes persistent harm?** Smaller relative top-k
   boundary gaps and tighter broader local gaps. The complete-GT sensitivity
   retains that direction, with a small stable group and no causal claim.
5. **Are most corrected errors near the boundary?** No under the registered
   small-margin and rank-neighborhood descriptions. Residual and new PQ64
   errors, rather than corrected ones, are more boundary-concentrated.
6. **Is retrieval-critical inversion a justified central mechanism?** Yes
   as a controlled scoring/selection description, provided actual GT
   displacement is distinguished from redundant cross-boundary inversions.
   It is not yet a validated pre-scoring predictor or an intrinsic query class.
7. **One next experiment:** keep these exact candidate pools fixed and train
   three standard PQ64x8 models with independent codebook seeds under otherwise
   identical training settings. Score the pools only, and measure repeatability
   of harmful queries and residual small-margin inversions, reporting both all
   queries and the complete-GT stratum. This separates codebook-specific errors
   from persistent local-geometric sensitivity without a new quantizer,
   selective reranking or traversal instrumentation. It has not been run here.
