# Phase 4A — held-out online uncertainty-guided selective refinement

Status: complete. The preregistered useful held-out stratification criterion
passes at all three depths; all nine aggregate budgets and all 36 shard
budgets exceed their random allocation 95% interval upper endpoint.
This supports a practical feasibility test, not a demonstrated latency gain
or superiority to inexpensive uniform refinement.

## Frozen design and data integrity

See `runs/phase4a_gap_guided_refinement_v1/preregistration.md` and
`configs/indexes/phase4a_gap_guided_refinement.conf`. Primary GAP is the
unmodified raw PQ rank-12 minus rank-9 gap. L=16/32/64, fractions=10/25/50%,
100 random allocations. No learned model, quantizer selection, graph changes,
trajectory instrumentation or wall-clock benchmark.

The standard learning vectors were downloaded from the same
[fixed dataset revision as the existing base](https://huggingface.co/datasets/qbo-odp/sift1m/tree/bd8ccad6c2a0a0a3a7519f6d37c0e5a2d59fe55b).
File: 100,000 × 128 fvecs; SHA256
`331bc82b6a0e89465776a3ba0c2113e0bd0cceaa014ec3ed639bc8b981af72ea`.
Initial download timed out; resumed bytes passed the complete checksum.
The original FTP endpoint had a TLS hostname mismatch; TLS verification was
not disabled. No unverified substitute dataset was used.

Crucially, **10,017 learning records match old standard queries by exact
vector content**. Another 195 records are duplicate learning contents.
The preregistered filtering removes these, leaving 89,788 eligible records.
None match a base vector. PCG64 seed 94000011 permutes eligible original
learning IDs: first 65,536 train, next 10,000 test. Train/test/old-query/base
content disjointness is explicitly checked. `training_learn_ids.i64` and
`queries_learn_ids.i64` contain full little-endian int64 source ID lists;
`provenance.json` records their checksums, all source hashes and machine details.
All vectors stay unnormalized FP32; no preprocessing beyond the documented
content filtering and role split.

One fresh identity-basis FAISS PQ64×8 uses seed 94000037, standard 25 k-means
iterations and 24 training threads. Same 65,536-vector count as recent phases;
using the learning source fixes the preferred identical-source disjoint roles.
It is not selected among alternatives. Existing FP32 HNSW M16/efConstruction80
is loaded without rebuilding; historical graph fingerprint
`1b3c18f6e2dd3ccec56e74eed30458f60a65f01234bf51478af25d84b5c02d16`.
efSearch64, k10, one search thread. Pinned FAISS
`20f14b31a6d54e243a3d1de6ae193fc4c3ec18ed`; execution Git `a0f5daa` plus
recorded working-tree implementation, not a fictitious clean commit.

## Cost, selection and tie conventions

All implementable selections use only actual PQ-guided L0 candidate IDs and
PQ ADC scores. Exact search is separate control data, never a refinement
candidate source. Full candidate exact reranking and exhaustive GT are offline
labels/controls; their measurement cost is not charged to online refinement.
Refinement computes L distinct exact distances per selected query. Baseline
search is unchanged. This is a workload-level selector, not a streaming rule.

All selectors choose floor(f*n) queries with query-ID tie breaking. GAP uses
only raw scores. Stable first-L0-evaluation order breaks candidate score ties.
ORACLE-QUERY ranks by each L's **actual recoverable recall gain**, the genuine
depth-specific selector upper bound. A separately named FULL-LOSS-SELECTOR
uses full-pool ranking loss to expose truncation-related differences. Neither
offline selector enters GAP. Negative gains and exact-distance ties are retained.

Random interval: empirical 2.5th/97.5th percentiles over 100 equally sized
random allocations of this fixed workload. Not a confidence interval for new
query populations. Four query-ID-modulo-4 shards independently repeat identical
policies. No shard-specific tuning. Epsilon=1e-12 for ratio denominators.
Candidate pools below 64 trigger a stop before budget interpretation; below
12 have preregistered gap=+infinity. All such cases must be reported.

## Pre-registered decision

Useful held-out stratification requires at least two L to exceed RANDOM's
mean and 97.5th percentile at **both** 25% and 50%. Strong success additionally
requires broadly stable advantages at most budgets and meaningful upper-bound
headroom capture; these latter descriptions are qualitative, not post-hoc
thresholds. Failure explanations to distinguish: signal transfer, depth
truncation, or insufficient query-selector headroom. No policy changes allowed.

## Validation before interpretation

Six new policy reference tests and five C++ backend/metric/loader tests pass.
A compile-time FAISS container-comparison issue was fixed before execution;
no experiment results were overwritten. Build logs report clock skew in this
environment; the new target compiled and linked successfully. Final Python
regression suite: 71 tests (including seven Phase 4A tests), all pass.

The entire base is byte-compared with graph storage. Every PQ code is
independently re-encoded in base-ID order and compared. The existing native
search/recorder identity control passes for all 10,000 exact and PQ queries,
including result scores. Historical graph fingerprint is unchanged before
and after both searches. Minimum PQ pool size is 236, mean 1036.1485:
no short-pool rule was invoked. All **10,361,485** candidate exact distances
equal independently computed direct FP64 squared distances exactly.

Full FP32 ground truth is computed for all 10,000 queries (top11 retained to
document boundary ties). On 100 fixed queries, independent direct FP64
exhaustive search verifies all mandatory strict neighbors and admissible
boundary neighbors; distances match exactly. A separately saved direct
reference also covers these same 100 queries, not another independent sample.

### Explicit tie investigation; primary metrics were not rewritten

686 queries have native/global-PQ ordered-list differences, all with exactly
the same ordered PQ score vector. Native IDs all belong to the PQ pool;
no strictly better PQ score is excluded. This is the previously understood
score-tie behavior, not a change in selection semantics.

717 exact-native/control ordered lists differ at exact distance ties.
**44 queries have ID-based delta_exact_control=-0.1** (mean -0.00044).
Every changed membership is at the exhaustive GT rank-10 distance. All
native/control exact score lists agree. Of the 44 changed ID pairs, 40 are
identical base-vector contents; four are distinct vectors at exactly equal
query distance. Thus these are established label/tie anomalies, not upper-only
L0 contamination or missed strictly closer candidates. Detailed IDs/distances
remain in `validation.json`. Six negative ID-based ranking-loss queries are
also retained and checked as GT-boundary ties.

A **post-primary, separately named tie-aware sensitivity** uses strict-distance
hits plus min(available GT boundary slots, retrieved boundary hits), /10.
It does not change query selections, candidate selections, primary results,
or the pre-registered criterion. The capacity cap prevents awarding excessive
credit for boundary ties while missing strictly closer neighbors.
Tie-aware exact-control discrepancies are zero for every query. Useful
stratification still passes for L16/32/64. Tie-aware native/oracle recalls are
0.850650/0.951980; exact native is 0.952840. Original ID-based values below
remain the primary measurements.

## Observations: risk transfer and reference recalls

| Reference | Recall@10 | Exact refinement distances/workload query |
|---|---:|---:|
| PQ native | 0.850560 | 0 |
| Top16, every query | 0.941020 | 16 |
| Top32, every query | 0.951290 | 32 |
| Top64, every query | 0.951390 | 64 |
| Full PQ candidate oracle | 0.951390 | 1036.1485 |

Exact-native recall=0.952760; exact-L0-oracle recall=0.952320.
Mean signed discovery delta=0.000930; candidate recoverable gap=0.100830.
Full PQ-pool reranking recovers 98.66% of the exact-native/PQ-native gap;
tie-aware recovery is 99.16%, discovery delta=0.000860. No large unexpected
discovery anomaly justifies reopening the navigability direction.

Spearman(raw g9_12_pq, ID-based ranking loss) = **-0.357433**.
Shard correlations: -0.373564, -0.342020, -0.361028, -0.352993.
The secondary normalized score is weaker here (-0.275785), but was never
eligible to replace the primary observable. Phase 3H's roughly -0.43 summary
was oracle-recall=1 restricted; its all-query correlations were roughly
-0.37. Do not interpret the unmatched -0.43/-0.357 comparison as an isolated
change in signal strength. This test also changes query population and PQ
training source, and uses real PQ rather than FP32 candidate pools.

| Raw-gap quintile, low to high | Mean ranking loss / available exact-oracle gain | Harmful fraction |
|---|---:|---:|
| 1 | 0.146400 | 0.8470 |
| 2 | 0.115550 | 0.7795 |
| 3 | 0.101650 | 0.7420 |
| 4 | 0.084350 | 0.6665 |
| 5 | 0.056200 | 0.4925 |

Each quintile has 2,000 queries; its definition uses scores/query IDs only.
This is descriptive held-out risk stratification, not a claim of causal
explanation or a distinct intrinsic hard-query class.

## Observations: all matched-cost policy points

The ORACLE column is the depth-specific, nonimplementable gain selector.
Random intervals below are empirical allocation intervals, not population CIs.
All values are raw Recall@10, not percentages.

| L | Refined | Cost/query | GAP | RANDOM mean [95% interval] | ORACLE | GAP−random | Selector efficiency |
|---|---:|---:|---:|---|---:|---:|---:|
| 16 | 10% | 1.6 | .863760 | .859593 [.859185,.860066] | .873740 | .004167 | 29.46% |
| 16 | 25% | 4 | .881250 | .873171 [.872649,.873721] | .899200 | .008079 | 31.04% |
| 16 | 50% | 8 | .906200 | .895774 [.894960,.896435] | .924200 | .010426 | 36.68% |
| 32 | 10% | 3.2 | .866050 | .860626 [.860139,.861051] | .875790 | .005424 | 35.77% |
| 32 | 25% | 8 | .885950 | .875761 [.875139,.876356] | .905790 | .010189 | 33.93% |
| 32 | 50% | 16 | .913540 | .900944 [.900168,.901617] | .930820 | .012597 | 42.16% |
| 64 | 10% | 6.4 | .866100 | .860637 [.860139,.861066] | .875840 | .005463 | 35.93% |
| 64 | 25% | 16 | .886020 | .875787 [.875165,.876376] | .905840 | .010233 | 34.05% |
| 64 | 50% | 32 | .913620 | .900995 [.900207,.901677] | .930900 | .012625 | 42.22% |

Precision shown is rounded; machine-readable tables retain full precision.
GAP recovery fractions for 10/25/50% respectively:
L16 = 13.09/30.44/55.18%; L32 = 15.36/35.10/62.46%;
L64 = 15.41/35.17/62.54%. Tables additionally retain recall gain per exact
evaluation, recovery differences, all 4,500 random allocation outcomes,
and FULL-LOSS-SELECTOR results. The latter is weaker than the true L-specific
upper bound at L16 (e.g. .922060 vs .924200 at 50%); not all full-pool loss
is recoverable at shallow depth.

All nine aggregate GAP advantages exceed the random interval upper endpoint.
Each L passes both required 25% and 50% comparisons, so the pre-registered
**useful held-out risk stratification criterion passes**. Advantages increase
over the three tested fractions for each L. Capturing 29.46–42.22% of the
descriptive selector headroom, together with 9/9 wins and stable budget
behavior, is consistent with the qualitative **strong gatekeeper success**
description. This is not a statistical-significance claim.

## Repeatability across four equal shards

Every shard has 2,500 queries and its own equally budgeted 100 random
allocations. All 36 shard GAP points exceed their random 97.5th percentile.

| L | 10% advantage range over shards | 25% range | 50% range |
|---|---:|---:|---:|
| 16 | .00403–.00449 | .00802–.00845 | .00989–.01088 |
| 32 | .00513–.00571 | .00960–.01080 | .01225–.01355 |
| 64 | .00513–.00574 | .00972–.01081 | .01229–.01358 |

These are query shards of one frozen graph/model, **not four independently
trained indexes**, and their ranges are not sampling confidence intervals.

## Depth limitations and secondary candidate diagnostics

Top16-all recovers 89.72% of the candidate recoverable gap; top32-all
99.90%; top64-all 100%. Thus depth beyond 32 has practically no aggregate
headroom here. Among GAP-selected queries at 25% budget, the fraction with
at least one candidate-oracle top10 member outside PQ top-L is 21.08% for
L16, 0.36% for L32, and 0 for L64. Mean correct oracle members newly added
per selected query are 1.4304/1.6728/1.6768. Newly recovered oracle-member
exact ranks have median 9 and p95 10. The separately recorded distribution
of *all* newly returned members includes non-oracle candidates (L16 p95 11);
these must not all be called corrected ground-truth neighbors.

The depth tradeoff must be compared at **equal cost**, not equal selected
fraction. At cost 8, GAP L16/50% gives .906200 versus L32/25% .885950.
For restricted budgets, L16 is the most sensible tested depth: broader
query coverage outweighs its remaining per-query truncation loss.

Important evidence against overclaiming a selective method: at cost 16,
**uniform top16-all gives .941020**, exceeding GAP L32/50% (.913540) and
GAP L64/25% (.886020). At cost 32, uniform top32-all (.951290) exceeds GAP
L64/50% (.913620). GAP beats random *within each fixed L/fraction*, but it
does not universally beat inexpensive uniform refinement. At budgets where
shallow refinement for every query is affordable, that baseline is stronger.
No wall-clock conclusions follow from distance counts alone.

## Decision and exactly one next experiment

1. **Risk signal generalizes:** negative association and monotonic quintile
   means persist on independent content-disjoint learning queries and real
   PQ L0 pools, without changing the score.
2. **Allocation beats random at equal cost:** 9/9 aggregate and 36/36 shard
   points, with tie-aware sensitivity preserving the useful criterion.
3. **Headroom capture is partial:** 29.46–42.22% above random relative to
   the depth-specific offline selector. It is not near-perfect detection.
4. **L16 is the cost-conscious depth** in the tested restricted-budget
   regime; L32 is the near-complete depth, and L64 is redundant here.
5. **Main remaining restricted-budget bottleneck is query selection / query
   coverage**, with measurable L16 truncation. Candidate discovery is not
   the bottleneck. Uniform shallow refinement becomes preferable when its
   full-workload cost is affordable.
6. **Proceed to a practical feasibility test, not a claimed algorithm win.**
   This experiment validates workload prioritization only. Online threshold
   stability, selector overhead, data fetches, tail latency and other
   codebooks/datasets remain untested. No learned risk model was trained.

**One next experiment:** on the remaining 14,252 content-eligible learning
vectors unused in this phase, test a streaming implementation with frozen
L16 and the Phase4A raw-gap 25th-percentile cutoff (no label fitting or
threshold retuning). Compare against RANDOM at the same realized count of
refined queries; retain uniform top16-all as an explicitly higher-cost
reference. Measure recall, realized exact-distance budget and end-to-end
latency/selector overhead. This single test asks whether the held-out
allocation advantage survives an implementable fixed threshold and real
execution costs. Do not retrain the PQ model or rebuild the graph.

## Artifacts and reproduction

Raw input IDs, split vectors, PQ model, candidate offsets/IDs/PQ and exact
scores, native and GT results: `runs/phase4a_gap_guided_refinement_v1/`.
`analysis/queries.jsonl.gz` preserves every per-query outcome; `selections.npz`
stores GAP/oracle selected IDs and every complete random order. Secondary
rescued-rank arrays are separately named. Primary completion checkpoint
precedes secondary diagnostics. Tables/figures: `results/{tables,figures}/phase4a_*`.
Exact commands, failures, checks and execution configuration are appended to
`docs/experiment_log.md`. `audit_phase4a.py` independently checks scalar
rankings, budgets and saved selections and compares regenerated tables and
figures against a separate output directory; no graph/PQ execution is needed
to regenerate the policy analysis.

Final audit **PASS**: all 10,000 scalar query rankings, 135 nonrandom policy
points and 4,500 random budget outcomes independently checked. All 22
table/figure files are byte-identical after regeneration; per-query rows and
selection arrays also regenerate exactly. `final_audit.json` records artifact,
source and executable hashes. Prior experimental runs were not modified.

The full budget sweep is reproducible offline replay of saved candidate
scores, not a timing benchmark. To additionally check the execution-level
cost assumption, `verify_phase4a_refinement_execution.py` **does not read
the candidate exact-distance cache**: it fetches only selected PQ top-L base
vectors and computes their FP32 squared distances afresh. All nine aggregate
budgets for GAP, offline ORACLE and RANDOM replicate0 (27 controls) match
saved result IDs, leave unselected native IDs unchanged, and count exactly
**2,856,000** FP32 distances in total. All 100 random allocations remain
independently scalar-audited; no wall-clock measurement is presented.
