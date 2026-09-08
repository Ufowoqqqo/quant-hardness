# Phase 3H — Hierarchical rank-conditioned residual-permutation controls

This is the final exact-distance-only mechanism phase. The later section is
a PQ-only observability feasibility bridge, not a trained detector or an
algorithm. No new graph traversal, candidate-pool construction, PQ training,
or quantizer optimization is performed. All prior experiments are preserved.

## Pre-registered design

The immutable pre-registration is
`runs/phase3h_rank_conditioned_nulls_v1/preregistration.md`, with H1-H5 and
interpretation cases written before outcomes. Configuration:
`configs/indexes/phase3h_rank_conditioned_nulls.conf`.

SIFT1M,1,000,000 database vectors,10,000 queries,d128. Five saved PQ64x8
models:R0-I1 through R4-I1. Same exact-traversal L0 pools as Phase3C-G,
10,353,047 candidate records,efSearch64,k10,FP32-built HNSW M16,
efConstruction80. Exact distances,GT,canonical exact top-k and candidate
order are frozen. Source model manifests and dataset/search settings,
FAISS revision,graph fingerprint,input hashes,machine and execution Git
revision are retained in `provenance.json`.

Exact-rank bin boundaries are not chosen from recall:

| Null | Inclusive exact-rank bins |
|---|---|
| N0 | 1..n |
| N1 | 1..10;11..n |
| N2 | 1..5;6..10;11..20;21..n |
| N3 | 1..5;6..10;11..15;16..20;21..40;41..n |
| N4 | 1..5;6..8;9..10;11..12;13..15;16..20;21..40;41..n |
| N_distance | Four equal-count groups of stable exact-distance ordering |

Each range is intersected with1..n; empty tails are discarded, the last
nonempty range is truncated, and no valid interior bin is merged. Singleton
bins stay fixed. Distance quartiles use `array_split(order,4)`; the first
n%4 groups contain one additional candidate. Exact ties are resolved by
the historical candidate order, including when a bin boundary splits a tie.

**Identification limit:** equal-count exact-distance quantiles are also
ordinal rank quartiles. Their comparison with N2/N3 tests bulk conditioning
versus near-top-k-local conditioning, not metric distance versus ordinal
rank as independently manipulated variables. No additional adaptive bins
are introduced to work around this limitation.

100 permutations/query/model/null,30million total, chosen before outcomes.
Five model-level worker processes,each with one BLAS thread; no ANN
performance benchmarking. N0 uses Phase3G's PCG64
SeedSequence([90700011,model_index,query_id,p]) and original candidate order,
so its first50 draws reproduce the previous experiment exactly. Other nulls
use SeedSequence([90800011,model_index,query_id,condition_index,p]); each
generator permutes the listed bins in order. Cross-condition permutation
indices are not common-random-number interventions.

Scores are FP64 `d+(PQ-d)[permutation]` from frozen FP32 scores. Observed
reconstruction must equal the saved PQ scores; full stable score-sort uses
original candidate order for ties. A critical inversion requires BOTH strict
exact-distance and strict reversed-score inequalities. Bin assignments are
checked on every permutation. No clipping of scores, losses or closure.

Per-query null means, sample SD,median,p05,p95 are saved; linear quantiles
and sample SD(ddof=1). Loss means/excess derive from integer hit-count sums,
avoiding spurious rounding ties. Aggregate uncertainty is across100 whole
permuted query datasets, conditional on the fixed queries/models. It is not
a population confidence interval. Combined figures distinguish this Monte
Carlo dispersion from variation between the five model means.

Gap closure=(mean_N0−mean_Nj)/(mean_N0−observed), reported only when the
denominator exceeds1e−6. Negative and>1 values remain. Per-draw closure
dispersion and invalid-denominator counts are separate. Inversion-gap closure
is analogous but separately labeled. Neither ratio is causal contribution,
explained variance or percentage of mechanism explained.

## PQ-only bridge conventions (post-primary)

The bridge runs only after a saved primary checkpoint. Inputs to the feature
function are PQ scores alone, not exact distance,exact rank,GT or labels.
Features are g10_11,g9_12,their normalized versions and three candidate
counts in symmetric windows around PQ_10. Scale=max(abs(PQ_10),1e−12);
the1%,2%,5% windows count candidates satisfying abs(PQ_v−PQ_10)<=f*scale,
including both sides and the kth item/ties. Scores are squared-L2 ADC values.
Normalized gaps use the same scale. This is seven features,not a learned
predictor. No ready per-vector reconstruction-error norm is stored in the
input artifacts; that conditional optional feature is omitted rather than
adding a decoding pipeline.

Exact ranking loss is only an offline label; oracle=1 is a separate label-
based control. Quantile bins preserve equal feature values and can yield
fewer than five occupied bins for discrete counts. Each model reports
Spearman,loss and harmful rate by observable quintile. The single feature
is selected by consistent sign across five models and maximum minimum
absolute oracle=1 Spearman,then median absolute correlation,then name.
This pre-registered selection is exploratory,not held-out validation.
Although feature values are PQ-only,the fixed pool was discovered using
exact traversal: online usefulness still requires a realistic PQ-pool test.

## Outputs and reproducibility

Each model under `runs/phase3h_rank_conditioned_nulls_v1/` stores:

- `N0.npz`...`N4.npz`,`N_distance.npz`:every ordered top10 and scalar metric,
  shaped[query,permutation] (top10 has a final length10 axis). `hits/10` is
  recall,`loss_hits/10` signed loss,`agreement_hits/10` agreement; displacement
  and strict inversion counts are integers. No per-pair null trajectories.
- `queries.jsonl.gz`:per-query observations,all null summaries,canonical
  exact IDs,truth,candidate fingerprint and distance-quartile residual stats.
- `residual_diagnostics.json`:ALL-candidate residual statistics by each
  exact rank1..30,coarse tails and special regions,plus pooled exact-distance
  quantile bins. SD/variance are population residual moments; absolute
  quantiles use the complete samples,not random-pair approximations.
- `validation.json`:N0 regression,bin restrictions,scalar tests and counts.

The raw bridge contains every query/model's seven features and offline
labels. Tables/figures can be regenerated to separate directories; scripts
refuse to overwrite a primary raw run. Exact commands are appended to
`docs/experiment_log.md`. All earlier data remain intact.

## Results and decision

### Primary observations

All30million permutations completed in972.8 seconds after setup. All five
models retained all10,000 queries; the smallest pool has247 candidates,so
no actual query required tail truncation. The N0 first50 draws reproduce
every saved Phase3G metric and ordered top10 exactly. N0 below averages100
draws,so its aggregate differs slightly from Phase3G's50-draw estimate.
Observed PQ recall/loss/inversions remain exactly the same.

Five-model means,all queries; ± values are conditional Monte Carlo sample SD
of the five-model mean over100 draws,not model/population confidence limits:

| Condition | Recall@10 | Ranking loss ± MC SD | Critical inversions | Harmful fraction | Top-k agreement |
|---|---:|---:|---:|---:|---:|
| Observed | .848168 | .104982 (fixed) | 6.01094 | .72252 | .874046 |
| N0 | .811580 | .141570 ± .000332 | 10.13126 | .83195 | .835296 |
| N1 | .804929 | .148221 ± .000304 | 10.25832 | .83917 | .825255 |
| N2 | .838735 | .114415 ± .000227 | 7.26554 | .75143 | .863038 |
| N3 | .846910 | .106240 ± .000202 | 6.19020 | .72920 | .872561 |
| N4 | .847115 | .106035 ± .000159 | 6.17975 | .72619 | .872822 |
| N_distance | .834974 | .118176 ± .000320 | 7.29762 | .77134 | .860205 |

Per-model means,p05/p95 and all100 aggregate draws are retained,as are
between-model ranges/SD. A tiny numerical SD around2.8e−17 for repeated
observed loss is floating-point roundoff,not uncertainty in a fixed score.

### Descriptive gap closure — N1 is a counterexample

The following ratios use the five-model mean gaps,not the average of
per-model ratios:

| Condition | Loss-gap closure | Inversion-gap closure | Oracle=1 loss-gap closure |
|---|---:|---:|---:|
| N0 | 0 | 0 | 0 |
| N1 | −.1818 | −.0308 | −.2475 |
| N2 | .7422 | .6955 | .7368 |
| N3 | .9656 | .9565 | .9598 |
| N4 | .9712 | .9590 | .9691 |
| N_distance | .6394 | .6877 | .6368 |

These describe closure of a specified real-versus-null gap,NOT causal
contributions or percentages of mechanism. All denominator checks pass;
no draws are omitted because of near-zero/negative denominators.

Per-model loss-gap closure(all queries):

| Model | N1 | N2 | N3 | N4 | N_distance |
|---|---:|---:|---:|---:|---:|
| R0-I1 | .1149 | .7765 | .9504 | .9670 | .6371 |
| R1-I1 | −.2738 | .7389 | .9745 | .9650 | .6542 |
| R2-I1 | −.2520 | .7241 | .9628 | .9774 | .6244 |
| R3-I1 | −.2575 | .7282 | .9634 | .9658 | .6367 |
| R4-I1 | −.2500 | .7429 | .9779 | .9807 | .6456 |

N1 is worse than unrestricted N0 in four rotated models and in the combined
mean. Therefore merely separating exact top10 from outsiders does NOT
reproduce protection,and the hierarchy is not monotonically beneficial.
N2 closes most of the gap,N3 nearly reaches observed aggregate quality.
N4 improves the five-model mean loss by only.000205 relative to N3; R1-I1
actually gets slightly worse. Do not hide these failures of strict monotonicity.
The remaining N3/N4 minus observed losses are.001258/.001053 overall.

For7,136 oracle=1 queries,the observed loss is.119877; N0,N1,N2,N3,N4,
N_distance losses are.158850,.168498,.130133,.121442,.121082,.134032.
Thus near-closure is not an artifact of incomplete oracle recall. Neither
near-closure nor small mean excess implies identical per-query failures.

### Rank/distance-dependent residual distributions

All-candidate absolute residual means (squared-L2 score units):

| Model | Exact ranks1–5 | 6–10 | 11–20 | 21+ |
|---|---:|---:|---:|---:|
| R0-I1 | 928.2 | 986.2 | 1010.8 | 1332.1 |
| R1-I1 | 1049.0 | 1077.2 | 1109.1 | 1397.5 |
| R2-I1 | 1039.0 | 1077.0 | 1102.6 | 1397.0 |
| R3-I1 | 1030.1 | 1079.3 | 1094.7 | 1392.8 |
| R4-I1 | 1034.4 | 1075.0 | 1098.6 | 1396.3 |

MAE increases across these four pre-fixed rank regions in every model.
Top10 residual SD is1226–1266 versus1740–1813 for ranks21+. Top10 absolute
p95 is2538–2657 versus3491–3612 for ranks21+. Full signed means,SD,variance,
absolute p50/p90/p95 for every rank1–30 and all requested coarse regions
are in `phase3h_residual_diagnostics.*`; statistics are not limited to means.

There is ALSO signed rank-dependent bias: top10 mean residual is+266 for
R0 and+433 to+456 for rotated models. Ranks11–20 have+227 and+364 to+382,
respectively; candidate-weighted ranks21+ have+115 and−44 to−29. Both close
regions are overestimated on average,not oppositely signed. The top10
region has more favorable magnitude/variance but less favorable signed bias.
Consequently "closer always gets better error" is too broad.

This offers an observational explanation for the N1 counterexample. N1
preserves the positive top10 residual distribution but allows far-outside
residuals to replace near-outside residuals. For the rotated models,this
can increase the top10-versus-near-outsider signed disadvantage. N2/N3
preserve the near-outside distribution as well. However the intervention
simultaneously preserves bias,scale,tails and assignment restrictions;
we have NOT causally isolated signed bias or variance as the sole mediator.

Pooled exact-distance deciles also show larger residual scale at larger
distances: R0 MAE is824 in the lowest decile and1848 in the highest;
rotated models have938–945 versus1997–2009. Absolute p95 increases from
2142 to4673 for R0,and2305–2322 to5083–5123 for rotated models. Pooled
bins mix queries and cannot by themselves establish a within-query causal
distance law. Per-query distance-quartile statistics are separately preserved.

N_distance closes.6394 of the loss gap,less than N2/N3. Its first bin
contains at least62 candidates here and is typically much broader than the
top-k-local bins. The result supports near-boundary localization of the
useful conditioning information; it does NOT separate rank from distance
as independent variables,for the reason pre-registered above.

### Geometry susceptibility survives; remaining excess is weakly related

Spearman of exact d12-d9 with five-model-mean outcomes under oracle=1:

| Condition | Ranking loss | Observed-minus-null excess |
|---|---:|---:|
| Observed | −.6584 | — |
| N0 | −.7851 | .1470 |
| N1 | −.7635 | .2364 |
| N2 | −.7322 | .1371 |
| N3 | −.7133 | .0144 |
| N4 | −.6913 | .0278 |
| N_distance | −.7981 | .0495 |

N3 per-model excess correlations range−.0295 to+.0324; N4 ranges−.0015 to
+.0531. Geometry continues to associate with loss while becoming largely
unrelated to the remaining observed-minus-conditioned-null difference.
The null means average100 draws whereas observed model loss is one
realization; comparing correlation magnitudes directly can reflect smoothing.
No correlation here establishes causality or a universal hard-query class.

### Hypotheses and mechanism decision

H1 is NOT supported as a claim about every rank-conditioned null: N1 is
a reproducible counterexample. H2 is supported for N2/N3,not N1. H3 is
supported at the aggregate level,without strict per-model monotonicity.
H4 and H5 are supported descriptively,with the averaging caveat above.

Closest interpretation is **Case A: coarse/medium rank-conditioned residual
distributions largely reproduce the aggregate protection**. Specifically N2
is useful and N3 suffices to approach observed loss/inversions; N1 alone does
not. Fine candidate identity is not necessary to reproduce these aggregate
quantities. A small residual gap and realization-specific pair/query effects
remain,so do not claim the entire PQ score process is reproduced.

The protected quantity is the conditional residual DISTRIBUTION,not just
variance. Heteroscedastic error is visible,but the experiment does not
separate its effect from signed bias,tails or other preserved structure.
Exact local geometry is best interpreted as generic noisy-top-k susceptibility,
not a PQ-specific navigability or excess-damage mechanism. This ends the
exact-distance-only mechanism sequence; no further residual-decomposition
experiment is recommended.

## PQ-only bridge observations

The primary tables/figures were checkpointed before feature analysis. All
seven feature families show the expected consistent risk direction across
the five models. Selection uses the pre-registered oracle=1 stability rule:

| Observable | Minimum absolute Spearman across models | Median absolute Spearman |
|---|---:|---:|
| g10_11_pq | .3113 | .3198 |
| **g9_12_pq** | **.4294** | **.4362** |
| relative g10_11_pq | .2945 | .3057 |
| relative g9_12_pq | .4185 | .4255 |
| count within1% | .3539 | .3639 |
| count within2% | .4203 | .4287 |
| count within5% | .3723 | .4039 |

Gap correlations are negative; window-count correlations are positive.
The selected SINGLE observable is **g9_12_pq = PQ_(12)−PQ_(9)**,without
normalization. Its oracle=1 Spearman is−.4294 to−.4423; all-query Spearman
is−.3683 to−.3778. The count-within2% and normalized wider gap are close
competitors; the selected feature's small advantage is not a statistical
superiority claim or proof it transfers across score scales/datasets.

All-query loss by ascending g9_12_pq quintile:

| Model | Q1:smallest gap | Q2 | Q3 | Q4 | Q5:largest gap |
|---|---:|---:|---:|---:|---:|
| R0-I1 | .14705 | .11495 | .09880 | .08060 | .05615 |
| R1-I1 | .15520 | .12560 | .10730 | .09040 | .05965 |
| R2-I1 | .15320 | .12500 | .10270 | .08470 | .06125 |
| R3-I1 | .15240 | .12550 | .10685 | .08390 | .06080 |
| R4-I1 | .15160 | .12485 | .10720 | .09130 | .05760 |

Every model has monotonically lower quintile mean loss with larger PQ span.
Mean loss across model-specific smallest-gap quintiles is.15189 versus
.05909 in the largest-gap quintiles (about2.57x); corresponding harmful rates
are.8705 versus.5168. This is meaningful descriptive stratification but
substantial harmfulness remains even at large gaps. Under oracle=1,the
smallest-gap loss is.1673–.1762 versus.0643–.0681 for the largest-gap quintile.
Full harmful-rate and loss quintiles for EVERY feature/model/scope are in
`phase3h_bridge_quintiles.*`; no feature is omitted based on poor performance.

## Correctness, anomalies and scope limits

- 64 Phase3 metric tests pass,including8 new pre-run reference tests. All
  50,000 observed query/model rankings match Phase3G; all within-bin
  permutations preserve bin membership exactly. N0 first50 reproduces
  the prior2.5million samples exactly.
- All30million saved top10s are independently audited for candidate
  membership,unique IDs,GT hits,agreement,displacement and signed losses;
  aggregate summaries are checked against integer counts.300,000 sampled
  permutations are regenerated with all scores/metrics/ordered IDs identical.
  This is sampled full score replay,not a second complete30million run.
- All50,000 PQ-only feature rows pass an independent scalar reference using
  only saved PQ scores. The primary checkpoint remains unchanged by bridge
  analysis. Independent table/figure/bridge-output reproduction is recorded
  in `final_audit.json`.
- There are12,270 negative null losses across all conditions; all are
  explained by GT substitutions at pre-existing exact boundary ties. Counts
  N0,N1,N2,N3,N4,N_distance are1462,1548,2287,2406,2489,2078. They are not
  removed.3,181 negative candidate null scores also remain unclipped; these
  shuffled scores need not be realizable squared distances from a quantizer.
- The minimum pool has247 candidates,so no actual tail repair is required.
  No model/query/condition/permutation count was tuned or discarded.
- Matplotlib used temporary cache directories because the default cache was
  unwritable. No metric changed and plots generated successfully.
- Conditioning on exact rank is deliberately an OFFLINE null; it supplies
  oracle information and constrains how randomization can act. It is not a
  realizable online quantization method. Finer bins approaching identity
  are not by themselves evidence of a special mechanism.
- The seven PQ-side features use no exact inputs,but their current candidate
  pools came from exact search. Score-only feasibility does not establish
  native-PQ-search deployment behavior or end-to-end cost. Feature selection
  used these same SIFT queries and is not held-out predictive validation.
- No ready per-vector reconstruction-error scalar is available in existing
  inputs; its conditional optional diagnostic is omitted. No learned
  predictor,quantizer,refinement algorithm or performance optimization was
  introduced. All earlier runs remain preserved.

## Final decision and exactly one next experiment

Mechanism answers:

1. N2/N3/N4 have descriptive loss-gap closure.7422/.9656/.9712; the bulk
   distance-quartile null gives.6394. The oracle=1 control reaches similar
   conclusions. These are specified gap ratios,not causal shares.
2. N3's medium near-boundary conditioning suffices for near-equality of
   aggregate recall/inversions; fine candidate identity is not needed for
   those aggregates. N1 alone fails,and exact per-query errors are not claimed
   to be identical.
3. Yes: local geometry supports generic noisy-top-k susceptibility; the
   remaining N3/N4 excess is weakly related to d12-d9. Do not revive the
   navigability or PQ-specific excess-damage story from this result.

Bridge answers:

4. Yes: PQ-only features meaningfully stratify offline ranking risk under
   the fixed-pool control. This warrants testing online-visible uncertainty,
   not declaring a working detector or selective-refinement method.
5. The SINGLE selected observable is **g9_12_pq**,by the pre-registered
   cross-model stability criterion. Smaller span corresponds to higher risk.
6. Proceed to exactly one **held-out fixed-budget refinement feasibility
   gate using frozen g9_12_pq**. Use genuinely unused independent queries
   and actual PQ-traversal candidate pools; do not relabel these already
   analyzed SIFT queries as unseen test data. Freeze the feature,orientation
   and one query-level refinement budget (e.g.20%) before evaluation. Compare
   smallest-span-prioritized exact reranking with random query selection at
   the SAME budget,plus unrefined recall and an offline oracle bound. Record
   loss recovery and exact-distance work as well as risk stratification;
   use no exact geometry as a selection input and do not fit a classifier.
   This is one prospective observability/refinement-benefit experiment,not
   another residual mechanism analysis. It is NOT run or implemented here.

Phase3H closes pure exact-distance mechanism analysis. The next question is
whether the PQ-visible signal transfers to independent queries and real PQ
candidate pools strongly enough to justify spending a limited exact-distance
budget,not whether another oracle-only decomposition can be constructed.
