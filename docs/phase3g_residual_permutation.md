# Phase 3G — Within-query PQ residual-permutation negative control

## Design and interpretation (registered before outcomes)

This is a score-only negative control, not an algorithm experiment. See
`runs/phase3g_residual_permutation_v1/preregistration.md` for the immutable
H1-H4 hypotheses, interpretation cases and analysis conventions. Five
existing models are fixed by name: R0-I1,R1-I1,R2-I1,R3-I1,R4-I1. No model
is selected using recall. No graph traversal, training or new PQ scoring is
performed; stored Phase3F ADC scores are reused.

SIFT1M:1,000,000 database vectors,10,000 queries,d=128. The original FP32
HNSW graph uses M=16,efConstruction=80. The fixed candidate pools are the
Phase3C FP32 L0 evaluated pools at efSearch=64,k=10:10,353,047 total
query/candidate records, in original first-evaluation order. Their exact
distances, IDs, GT and exact top-k remain unchanged. PQ64x8 uses the same
65,536-vector training subset and I1 initialization80500011 in each basis.
All source model manifests, codebook/code checksums, FAISS revision, graph
fingerprint, original dataset paths/checksums, execution Git revision,
source hashes and machine information are recorded in `provenance.json`.

For every candidate, e=float64(stored_PQ)-float64(stored_exact). Observed
d+e must reproduce the stored PQ score exactly and all five models' Phase3F
ordered top-10 IDs and core metrics. Null scores are d+e[permutation], with
no clipping of negative scores or signed losses. Residual permutation
preserves the full empirical residual multiset, not just its first moments.
Every permutation is checked to be a bijection.

Exactly50 permutations/query/model,2,500,000 total. Independent PCG64
streams use SeedSequence([90700011,rotation_index,query_id,permutation_index]).
The count was fixed before runtime or outcomes. Stable full sorts resolve
score ties by original candidate order, identical to Phase3C-F. Strict
cross-boundary inversion excludes exact-distance ties and PQ-score ties.

Per-query loss uses integer hit differences divided by10. Null mean,
sample SD(ddof=1),median,p05,p95 are stored; quantiles use linear interpolation.
Observed empirical rank stores counts below/equal/above, midrank
(below+0.5*equal)/50 and interval[below/50,(below+equal)/50]. These are
descriptive ranks, not per-query hypothesis tests.

Each permutation index defines one full independently shuffled query
dataset. Aggregate null uncertainty is the sample SD and p05/p95 across
the50 aggregate values. It measures Monte Carlo variation conditional on
these fixed queries/models, NOT query-population confidence intervals.
All models are reported separately; their shared queries are not treated
as independent replications. The oracle_recall=1 subset controls for
incomplete exact candidate recall.

## Raw schema and regeneration

Run root:`runs/phase3g_residual_permutation_v1/`.
Each model directory contains:

- `queries.jsonl.gz`:one row/query, observed IDs/metrics, null summaries,
  excess values, empirical ranks, candidate fingerprint, truth and residual
  diagnostics. Null per-query distributions are not replaced by averages.
- `permutations.npz`:arrays shaped[query,permutation], with `topk` additionally
  having a final length10 axis. `hits/10` is fixed-candidate recall;
  `loss_hits/10` is signed ranking loss; `agreement_hits/10` is top-k
  agreement; `displaced_count` and strict `inversions` are integer counts.
  Ordered top-10 IDs are stored for every permutation.
- `replacement_pairs.jsonl.gz`:observed displaced/intruder IDs, exact
  margins and residual differences, plus null summaries for those SAME pair
  identities. These observed-selected pairs are selection-conditioned.
- `diagnostics.npz`:residuals at exact ranks1-30;64 deterministic candidate
  samples/query (ID,exact squared distance,residual,identity-vector L2 norm);
  and per-permutation count/summed residual difference/summed exact margin
  for actual null-selected replacement pairs. Candidate samples are identical
  across models. Full candidate residuals can be reconstructed from hashed
  historical exact and ADC arrays; shuffled residual arrays need not be stored.
- `validation.json`:regression/invariance checks and progress/runtime counts.

The analyzer regenerates tables and figures from these files. Residual
statistics by rank are across queries. Region summaries identify whether
they weight queries or candidates; farther-candidate pools differ in size.
Pooled candidate bins are descriptive and can mix between-query effects;
within-query Spearman values are also reported. No reconstruction-error
predictor, parametric noise fit, or learned detector is introduced.

Exact geometry follows Phase3F: d12-d9=`boundary_gap_r2`,d15-d6=`boundary_gap_r5`,
relative rank-10 margin=(d11-d10)/max(d10,epsilon), concentration=(d20-d1)/d1.
Relative d12-d9 divides by max(d10,epsilon). All distances are squared L2,
except the explicitly labeled candidate-vector L2 norm.

## Results

### Observations: real PQ is LESS harmful than global residual reassignment

The exact candidate oracle remains0.95315. All50,000 observed query/model
rankings reproduce Phase3F exactly. Main derivation took570.6 seconds after
input setup. No model was added, removed or retrained after seeing results.

| Model | Observed recall | Null recall | Observed loss | Null mean loss ± permutation SD | Excess loss |
|---|---:|---:|---:|---:|---:|
| R0-I1 | .853640 | .816120 | .099510 | .137030 ± .000715 | −.037520 |
| R1-I1 | .845520 | .810258 | .107630 | .142892 ± .000682 | −.035262 |
| R2-I1 | .847780 | .810192 | .105370 | .142958 ± .000766 | −.037588 |
| R3-I1 | .847260 | .810966 | .105890 | .142184 ± .000730 | −.036294 |
| R4-I1 | .846640 | .810538 | .106510 | .142612 ± .000799 | −.036102 |

The five-model mean observed loss is.104982; null mean loss is.14153544;
signed difference is−.03655344. The null generates132.76–137.70% of each
model's observed loss (pooled model-mean ratio134.82%). This is NOT a causal
"fraction explained": shuffling changes the errors assigned to individual
candidates and produces MORE loss, not an additive decomposition of actual
errors. Real assignment reduces the null baseline loss by25.83% on the
five-model mean comparison.

| Model | Observed inversions | Null inversions ± permutation SD | Difference | Observed harmful fraction | Null harmful fraction |
|---|---:|---:|---:|---:|---:|
| R0-I1 | 5.5562 | 9.6549 ± .0825 | −4.0987 | .70010 | .81844 |
| R1-I1 | 6.2528 | 10.2623 ± .0851 | −4.0095 | .73370 | .83597 |
| R2-I1 | 6.1024 | 10.2883 ± .0782 | −4.1859 | .72460 | .83579 |
| R3-I1 | 6.0736 | 10.1962 ± .0718 | −4.1226 | .72680 | .83386 |
| R4-I1 | 6.0697 | 10.2478 ± .0754 | −4.1781 | .72740 | .83442 |

Every observed aggregate loss lies below the corresponding50 aggregate null
losses, but this is a descriptive conditional comparison, not a population
significance claim. Per-query behavior is not uniform: mean observed loss
midranks are.3716–.3798, rather than uniformly zero; median excess loss is
−.036 to−.038. Individual PQ queries can be worse than their null mean.
All query comparisons and empirical tie intervals remain available.

### Oracle=1 control and exact geometry

There are7,136 oracle=1 queries. Their observed/null losses are:

| Model | Observed loss | Null loss | Excess |
|---|---:|---:|---:|
| R0-I1 | .114084 | .155292 | −.041209 |
| R1-I1 | .122702 | .159963 | −.037261 |
| R2-I1 | .119367 | .159911 | −.040545 |
| R3-I1 | .121188 | .159211 | −.038023 |
| R4-I1 | .122043 | .159694 | −.037651 |

Thus incomplete oracle recall does not explain the aggregate protection.
Spearman associations of exact d12-d9 in this controlled subset are:

| Model/averaging | Observed loss | Null mean loss | Excess loss |
|---|---:|---:|---:|
| R0-I1 | −.4243 | −.7618 | .0841 |
| R1-I1 | −.4318 | −.7678 | .0799 |
| R2-I1 | −.4424 | −.7650 | .0662 |
| R3-I1 | −.4282 | −.7647 | .0837 |
| R4-I1 | −.4581 | −.7658 | .0536 |
| Five-model mean | −.6584 | −.7840 | .1472 |

Generic reassignment retains a strong geometry–loss relationship. The same
span associates only weakly with signed observed-minus-null loss. This is
evidence AGAINST claiming the Phase3F geometry correlation requires
PQ-specific assignment of residuals to candidates. Note the averaging
asymmetry: each null mean averages50 shuffles; an individual observed model
does not. The five observed models here also differ in averaging count from
the15-model Phase3F mean, so−.6584 versus−.739 is not a direct robustness
failure. Full descriptor correlations and query-quantile bins are in the
`phase3g_geometry_*` tables, separately for all/oracle=1 queries.

### Observable residual structure (not a causal allocation)

Residual magnitudes and variances are smaller near the exact top-k than
among farther candidates. Candidate-weighted region values:

| Model | MAE ranks1–10 | MAE ranks11–20 | MAE ranks21+ | Signed mean top10 | Signed mean11–20 |
|---|---:|---:|---:|---:|---:|
| R0-I1 | 957.2 | 1010.8 | 1332.1 | +266.1 | +226.9 |
| R1-I1 | 1063.1 | 1109.1 | 1397.5 | +456.2 | +373.0 |
| R2-I1 | 1058.0 | 1102.6 | 1397.0 | +449.7 | +381.9 |
| R3-I1 | 1054.7 | 1094.7 | 1392.8 | +442.4 | +368.2 |
| R4-I1 | 1054.7 | 1098.6 | 1396.3 | +433.0 | +364.5 |

Top10 residual variance is1.50–1.60 million versus3.03–3.29 million for
ranks21+. Full ranks1–30 statistics and residual-versus-distance/norm bins
are saved. The farther region comprises10,153,047 of10,353,047 candidate
records, so an unconstrained shuffle often assigns farther-region residuals
to top-ranked candidates. This destroys conditional error magnitude as well
as particular candidate identity. It is a plausible explanation for the
protective observed-minus-null difference, NOT a demonstrated sole mediator.

Signed bias alone points in the opposite direction: exact top10 distances
are overestimated on average, by39–83 more than ranks11–20. Near outsiders
are ALSO overestimated, not systematically underestimated in absolute
terms. Their relative bias can harm membership; it does not explain why
real assignment beats the global shuffle. Within-query residual/distance
Spearman averages−.0345 for R0 and−.1072 to−.1088 for rotated models;
residual/norm associations are weak (−.0506 for R0,−.0143 to−.0156 otherwise).
Under oracle=1, top10-minus-near signed bias associates with excess loss at
.4206–.4545; global within-query residual/distance correlation associates
with excess at only−.0106 to+.0228. These are exploratory associations and
share the same realized residuals used to construct loss, not independent
predictive or causal evidence.

Observed displaced/intruder pairs have mean error differences2802–2911
against exact margins1410–1466. Shuffling those SAME identities yields mean
differences−3.96 to+5.04, approximately the zero expectation of exchangeable
assignments, with strict inversion fractions.2844–.2887. Actual
shuffle-selected replacement pairs instead have mean error differences
3911–4024 and margins1966–2025. These are different, outcome-selected pairs;
the contrast must not be used as an unbiased causal estimate. It shows why
removing the observed failing pair identities does not remove ranking loss:
the null creates other failures.

## Correctness, anomalies and limitations

- All2.5 million null top-10 lists are preserved. A full saved-ID audit
  recalculates membership,hits,agreement,displacement and query summaries.
- All50,000 observed query/model outputs reproduce historical ordered IDs
  and metrics; score reconstruction is exactly equal. Candidate/GT/graph,
  model and score input checksums are unchanged before and after the run.
- Independent replay covers100 deterministic queries/model ×50
  permutations ×5 models=25,000 samples, with every saved metric and ordered
  list identical. This is a sampled replay, not a claim to have rerun every
  primary permutation independently.
- The post-primary centered-residual control uses those25,000 samples.
  Zero ordered-list or membership changes occur; maximum difference from
  exact query-wide score translation is2.91e−11 in FP64.
- There are171 pre-existing exact-boundary-tied queries. The null has744
  negative losses, all−.1 and all in these queries; GT-identical-distance
  substitutions explain them. None is discarded. Observed PQ has no negative
  losses. A secondary tie audit checks the gained GT candidates' distances
  equal the canonical exact boundary distance.
- There are1060 negative null candidate scores across the five models,
  deliberately not clipped. Permuted residuals need not correspond to a
  realizable squared-distance function or quantizer; this is a residual
  assignment control, not a surrogate ANN system.
- The initial independent verifier was stopped because repeated NPZ reads
  re-decompressed whole arrays. It was retried after materializing each
  model's arrays once. This changed only verification I/O; primary data and
  tables were not rerun or altered. Both execution logs are retained.
- Matplotlib used a temporary cache because the configured home cache was
  unwritable; figure generation succeeded. This does not affect metrics.
-56 existing/new Phase3 metric tests pass. The new7 tests include scalar
  nested-pair/full-sort and historical metric comparisons before production.
  Independent table/figure regeneration and final checksums are recorded in
  `final_audit.json`.
- Residual permutation destroys rank-dependent bias,heteroscedasticity and
  candidate-specific dependence simultaneously. The present experiment
  cannot separate those contributions. Shared queries/fixed graph and only
  five selected existing models limit external validity. No novelty or
  heavy-tail claim is made.

## Hypotheses and decision

H1 is supported in the limited sense that generic reassignment readily
produces ranking loss, and even exceeds observed loss; it does NOT reproduce
the exact observed aggregate rate. H2 is supported. H3 is supported as a
signed difference, but there is NO positive average excess PQ loss. H4 has
observable rank-dependent magnitude/variance and bias associations, but the
specific protective mediator is not established. Do not convert that into
a confirmed harmful-PQ-bias explanation.

Closest pre-registered category: **Case B, with a protective rather than
damaging assignment effect**. The generic noisy-top-k geometry component is
strong, but observed and shuffled losses are not close, so strict Case A
does not fit. The stipulated Case C wording "structured assignment adds
additional loss" does not fit the negative mean excess either. In ordinary
terms the mechanism is mixed: generic local-margin susceptibility plus
material, net-protective PQ residual assignment. This does NOT establish
that PQ-specific structure is necessary for harmful ranking errors.

Answers to the final questions:

1. Generic reassignment produces132.76–137.70% of observed mean loss, not
   merely a small portion. These ratios are descriptive, not additive causal
   explained fractions.
2. Yes: oracle=1 d12-d9/null-loss correlations are−.762 to−.768 per model
   and−.784 across five-model means; geometry susceptibility survives.
3. No positive mean excess. Real PQ loss is lower by.0353–.0376 overall
   and.0373–.0412 under oracle=1, consistently across all five bases.
4. Smaller top-k residual variance/MAE is compatible with protection;
   relative signed bias can contribute to individual harm. This experiment
   does not isolate their respective effects or rule out further dependence.
5. Generic fragile geometry plus structured protection; closest formal
   case is signed Case B, not a PQ-specific excess-damage story.
6. Not yet sufficient to justify online uncertainty detection/selective
   refinement. Exact gaps and residuals used here require unavailable exact
   information online; neither an online observable nor out-of-sample
   detection/refinement benefit has been validated. This is not an argument
   for an open-ended mechanism program, but one remaining discriminating
   control is warranted before assigning the protective effect a mechanism.

**Exactly one next experiment (not run): rank-stratified within-query
residual permutation.** On these same five models/pools, shuffle only within
pre-fixed exact-rank bands1–10,11–20,21–50,51–100,101+. Keep50 shuffles and
the current metrics. This preserves coarse rank-dependent residual
distributions while disrupting within-band candidate assignment. Compare
observed loss with this conditional null and the existing global null. If
the conditional null closes the gap, coarse rank-conditioned error scales/
bias suffice; if a material gap survives, finer candidate dependence remains.
Use the existing oracle=1 control. This is an offline diagnostic, not an
online feature, new quantizer, detector or refinement algorithm.

## Commands and artifacts

Exact commands/configuration and the verifier retry are in
`docs/experiment_log.md`. Main entry points:

```bash
OPENBLAS_NUM_THREADS=1 /tmp/phase3b-plot-env/bin/python scripts/run_phase3g_residual_permutation.py --config configs/indexes/phase3g_residual_permutation.conf --run runs/phase3g_residual_permutation_v1
OPENBLAS_NUM_THREADS=1 /tmp/phase3b-plot-env/bin/python scripts/analyze_phase3g_residual_permutation.py --config configs/indexes/phase3g_residual_permutation.conf --run runs/phase3g_residual_permutation_v1
OPENBLAS_NUM_THREADS=1 /tmp/phase3b-plot-env/bin/python scripts/verify_phase3g_artifacts.py --config configs/indexes/phase3g_residual_permutation.conf --run runs/phase3g_residual_permutation_v1
```

The primary runner refuses to overwrite an existing run; for a full rerun,
use a fresh directory containing the same pre-registration. The analyzer's
`--tables` and `--figures` arguments support independent reproduction without
overwriting published outputs. All prior phases and this run's raw outputs
remain preserved.
