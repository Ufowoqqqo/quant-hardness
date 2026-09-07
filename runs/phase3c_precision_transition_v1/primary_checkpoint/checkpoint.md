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
