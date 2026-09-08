# Phase 4B — frozen streaming GAP-L16 feasibility

Status: complete. The frozen risk signal generalizes, but the preregistered
systems-value gate **fails**. No threshold, policy or implementation performance
tuning followed inspection of Phase4B outcomes.

## Frozen policy and untouched queries

Raw `g9_12_pq = score(rank12)-score(rank9)`, ascending ADC scores;
refine16 if and only if `gap <= 710.40625`. Numeric provenance is stored in
`threshold_provenance.json`. This is both Phase4A's linear 25th percentile
and its rank2500 score. A tie also includes rank2501 (25.01%) under inclusive
comparison. Phase4A's quota selection used query-ID tie-breaking; this phase
does not replicate that quota and never computes a new percentile.

The 14,252 queries are the remaining suffix of the exact Phase4A eligible-ID
permutation. Both earlier prefixes are checked against the 65,536 training
IDs and 10,000 Phase4A query IDs. Learning content already overlapping old
standard queries or duplicate contents was excluded in Phase4A. No base
contents overlap the eligible learning set. Warmup uses the first512 old
Phase4A queries, completely separate from the 14,252 measured vectors.
All source IDs, vectors and checksums are retained. No model training occurs.

Same identity-basis PQ64x8 `runs/phase4a_gap_guided_refinement_v1/model.index`;
same FP32 HNSW M16/efConstruction80, graph construction seed20260907,
24 construction threads, historical fingerprint
`1b3c18f6e2dd3ccec56e74eed30458f60a65f01234bf51478af25d84b5c02d16`.
efSearch64, k10, L16; no graph construction or altered search topology.
FAISS `20f14b31a6d54e243a3d1de6ae193fc4c3ec18ed`;
execution Git `31e7412` plus recorded implementation hashes.

## Candidate accessibility and online timing boundary

The native k10 result heap does not expose all distance-evaluated candidates
or retain the needed top16. Requesting k16 would change a search parameter
and is not used. The old research recorder also does replay validation and
whole-graph hashes; timing that wrapper would measure research machinery.

`streaming_refinement.cpp` therefore calls FAISS's existing single-query
HNSW APIs. Native calls `HNSW::search`. Retention paths call the existing
upper greedy helper, then `HNSW::search_level_0`, without implementing ANN
search anew. A delegated DistanceComputer records each unique L0 ID/ADC
score using reusable uint32 epoch tags and ID/score arrays. Upper-only
evaluations are excluded; the cached L0-entry score is admitted once, giving
the same membership/order as the old recorder's explicit seed reevaluation.
There is no second graph search or candidate ADC rescoring online.

Top16 is a standard partial sort with first-L0-evaluation-order tie breaking;
exact reranking preserves that tie convention. The full-sort reference and
all evaluated score/ID ordering are checked before timing. No short pool is
silently padded or assigned a changed L.

All four policies use the same low-level FAISS single-query integration, with
public `IndexHNSW::search` as an output-identity correctness control. Thus
timings are not claimed to be the stock high-level Python/C++ wrapper's
performance. Query-local distance-computer setup/destruction, visited-table
handling and result heap setup are timed in every policy.

GAP retains/extracts top16 for every query, computes its own gap and branches
inside **every timed call**. Its untimed correctness run supplies only the
count needed to generate RANDOM masks, never a lookup decision in timed GAP.
RANDOM mask0 is fixed, content-independent and matches the actual count;
unselected RANDOM queries skip retention/extraction entirely. ALL always
retains/extracts top16 but has no gap-based decision. No artificial equal
overhead is imposed on an easier baseline.

Components (clocks ON diagnostic):

- `t_search`: query setup, PQ ADC table, upper/L0 search, required retention;
- `t_gap`: top16 extraction, score-gap calculation where needed, policy branch;
- `t_refine`: the selected16 exact FP32 distances;
- `t_final_selection`: native heap ordering, selected exact top10 construction,
  output copying and query-local cleanup.

Their sum equals `total_online` apart from floating-point reporting precision.
Primary latency uses component clocks OFF and only an outer steady-clock
pair around the same complete API call. Uninstrumented here means no internal
component clocks, **not removal of required application candidate retention**.
Input query pointer/random mask bit are already available to that API;
network queueing and input transport are outside scope. Actual stream QPS
also brackets the entire query-call/output-collection loop; correctness
comparisons and file writing happen only after the whole timed stream.

## Benchmark discipline and preregistered gate

Single thread pinned to CPU2; seven repetitions of each policy with component
clocks OFF and ON. All eight conditions are independently shuffled per
repetition (seed94100011), with512 old warmup queries before each. Identical
measured query order, no run trimming, no parameter changes after timing.
30 content-independent RANDOM masks use PCG64 SeedSequence([94100037,r]);
mask0 is fixed for latency. The plotted RANDOM recall is mask0's own recall,
not substituted with the mean of30 masks; the mean/interval is also reported.

Primary service-latency statistics pool the equal-length seven repeats;
variation of repetition means is separate. A service QPS=1e6/mean_us and
measured stream QPS are both retained. The random recall 95% interval describes
mask allocation randomness, not query-population confidence. Component-clock
ON−OFF differences also include ordering/cache/frequency noise, not a pure
microbenchmark of timestamp instructions.

No agent-owned build, GT, validation or analysis worker runs during timings.
Host process check found no heavy competing job. Existing performance governor
and enabled turbo are left unchanged; no privileged tuning or unrelated
process termination. Shared-host/frequency snapshots and repetition spread
are retained rather than claiming exclusive isolation.

The original exploratory success criterion is unchanged: beat matched RANDOM,
save >=20% mean latency versus ALL16, and retain >=60% of ALL16's recall gain
over native. Strong additionally improves the practical frontier with a
nonpathological tail. No post-hoc alternative numerical success thresholds.

## Correctness before timing

512 old queries: identical public/native IDs and scores, identical historical
L0 first-evaluation IDs and ADC scores. FAISS indexed batched distances equal
8,192 direct scalar distances. All14,252 new queries pass public/native
identity and full-sort top16/policy references. All14,744,891 candidate exact
distances equal independent FP64 direct squared distances. 100 deterministic
full-base FP64 comparisons validate exhaustive GT with explicit boundary ties.

Offline GT uses the existing FAISS direct FP32 exhaustive kernel, 24 threads,
instead of the reference BLAS route used in Phase4A. This explicitly documented
backend choice changes neither squared-L2 nor the exhaustive database, and
does not enter timed execution. Primary ID-based recall remains separate from
the inherited tie-aware sensitivity. No raw prior-phase metric is overwritten.

## Secondary batch diagnostic (after primary only)

Batch size32, same threshold/L16/PQ and one thread. Compare scalar-deferred
refinement with FAISS `pairwise_indexed_L2sqr` across the selected query/vector
pairs. Both generate decisions from fresh searches and defer return through
the batch; both include temporary index-list storage and final selection.
Seven repetitions, randomized two-mode order seed94100083. No batch weights,
model changes or new policy. Whole-batch amortized service time is **not**
individual streaming response latency; delayed batching is not credited with
an artificial per-request latency improvement.

## Observations: streaming recall, cost and latency

Primary component-clocks-OFF results, seven complete repetitions of 14,252
queries. Latencies below are microseconds. RANDOM here is the fixed mask0
actually timed; its 30-mask recall distribution is reported separately.

| Policy | Recall@10 | Exact evaluations/query | Total exact evaluations | Refined fraction | Mean | Median | p90 | p95 | p99 | Stream QPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PQ-NATIVE | 0.850309 | 0 | 0 | 0 | 161.650 | 163.587 | 183.213 | 189.410 | 203.592 | 6184.42 |
| GAP-L16 | 0.880501 | 3.992141 | 56896 | 0.249509 | 182.427 | 184.777 | 208.302 | 215.582 | 231.184 | 5480.24 |
| RANDOM-L16 mask0 | 0.872523 | 3.992141 | 56896 | 0.249509 | 171.107 | 169.445 | 207.130 | 217.673 | 235.087 | 5842.68 |
| ALL-L16 | 0.939973 | 16 | 228032 | 1 | 184.651 | 186.993 | 210.750 | 218.145 | 234.124 | 5414.31 |
| Full PQ candidate oracle (offline) | 0.950365 | — | — | — | — | — | — | — | — | — |

Service QPS (inverse mean API latency) is respectively 6186.21, 5481.64,
5844.29 and 5415.63; actual stream QPS also includes the output-collection
loop. Individual repetition mean latency standard deviations are 0.441,
0.140, 0.254 and 0.298 microseconds. All repetitions are retained.

GAP selected **3556 queries (24.950884%)**, a deviation of −0.049116
percentage points from 25%, without recalibration. RANDOM's 30-mask mean
recall is **0.872784**, with empirical central 95% interval
**[0.872184, 0.873504]**. GAP exceeds that mean by **0.007717** and exceeds
the interval's upper endpoint. This interval measures random allocation
variability, not uncertainty over independent machines or query populations.

Relative to ALL-L16, GAP saves **75.0491% exact evaluations** but only
**1.2042% mean latency**, retaining **33.6724% of ALL-L16's recall gain**
over native. Both frozen systems targets (20% latency, 60% gain) fail.
ALL-L16 costs only 2.224 additional microseconds while adding 0.059472 recall.
Repetition-to-repetition standard deviation of latency saving is 0.130
percentage points. GAP is formally nondominated among these discrete mean-
and p95-latency reference points, but this small tradeoff is not evidence of
the preregistered useful practical frontier improvement.

### Threshold generalization and paired outcomes

Spearman(gap, offline ranking loss) = **−0.356737**. Refined versus unrefined
queries have mean ranking loss 0.139370 versus 0.086986, and harmful-query
rates 83.6895% versus 65.5666%. Their candidate-oracle recalls are 0.946091
and 0.951786. These are descriptive held-out comparisons, not policy inputs.

Among the 3556 selected queries, 2823 (79.387%) improve recall, 733 (20.613%)
are neutral, and zero lose recall. Per-query gap, decisions, native/GAP/ALL16/
oracle recall and repeated GAP latency are preserved in
`runs/phase4b_streaming_feasibility_v1/analysis/queries.jsonl.gz`.

No FP32 graph traversal was run for these new queries. Therefore this phase
does **not** newly estimate exact-minus-PQ discovery degradation. The measured
full PQ candidate oracle is the relevant recovery upper bound; Phase4A's
approximately 0.00093 discovery delta remains a historical result only.

## Where online time goes

Component-clocks-ON diagnostic means, microseconds:

| Policy | Search including retention | Gap/extraction/branch | Exact refinement | Final selection |
|---|---:|---:|---:|---:|
| NATIVE | 161.672 | 0.015 | 0.016 | 0.210 |
| GAP | 178.329 | 3.056 | 0.571 | 0.278 |
| RANDOM | 169.675 | 0.803 | 0.568 | 0.279 |
| ALL16 | 178.719 | 3.056 | 2.166 | 0.401 |

The pure score subtraction is not separately isolated. Making its inputs
available is **not negligible**: retention and top16 extraction are paid for
every GAP query. ALL16 has almost identical extraction time without needing
a gap decision. The whole GAP path is approximately 20.8 microseconds slower
than native, while the entire ALL16 exact-distance stage is only 2.17
microseconds. Avoiding most of that cheap stage cannot offset the shared
candidate accessibility cost. Search-time differences also include adapter/
split-path and storage effects; no precise attribution to individual buffers
or memory bandwidth is claimed without profiling.

Clock-ON minus OFF mean changes are +0.263, −0.193, +0.218 and −0.308
microseconds respectively (magnitudes below 0.17%). Negative differences
reflect order/cache/frequency noise, not negative timestamp cost. Primary
claims use OFF timings. GAP p95/p99 are 215.582/231.184 microseconds, slightly
below ALL16's 218.145/234.124 and above native's 189.410/203.592. No new large
tail amplification is apparent in this workload; without an application SLA
this is not a universal statement that tail latency is acceptable.

## Machine and memory accounting

Intel Core i9-10920X @3.50GHz; one socket, 12 physical/24 logical cores;
32,273,192 KiB RAM. GCC 11.5.0 (Red Hat 11.5.0-14). Application flags:
`-O3 -DNDEBUG -std=c++20 -Wall -Wextra -Wpedantic -fopenmp`;
existing generic CPU FAISS: `-O3 -DNDEBUG -std=gnu++20 -fPIC -fopenmp`,
`FINTEGER=int`. All timed paths use one OpenMP/OpenBLAS thread, CPU2 affinity.
Performance governor, turbo enabled, neither changed. Recorded CPU2 frequency
snapshots (approximately 1.2 GHz) were idle observations, **not** locked or
measured active benchmark frequency. Host load/process snapshots and compiler
information are in `benchmark_environment.json`. No heavy competing process
was observed; the shared host was not reserved exclusively.

Refinement reads 512 nominal FP32 payload bytes per candidate. GAP/RANDOM
read 29,130,752 bytes over the workload (2043.98/query), versus ALL16's
116,752,384 bytes (8192/query). These are vector payload estimates, not
measured DRAM transactions. A PQ-only deployment additionally needs the
512,000,000-byte FP32 database for either refinement policy; in this benchmark
the same frozen FP32 storage is resident for all policies. GAP does not save
that resident memory relative to ALL16.

Reusable scratch reaches 4,024,576 bytes: four-million-byte epoch tags plus
capacity2048 ID/ADC/order arrays. Mean pool size is 1034.584, range216–1920;
live arrays use approximately 12.4 KB/query, with small top16/result stacks.
The threshold itself is eight bytes. No short pools or capacity growth beyond
2048 occur. Candidate retention is paid online and not hidden in preprocessing.

## Secondary batch observations

After the primary run, scalar-deferred32 and indexed-batch32 mean amortized
whole-path costs are **182.37694** and **182.37550 microseconds/query**.
Refine/final costs are 0.57280 and 0.65689 respectively; batching does not
provide meaningful end-to-end acceleration here. Mean repetition p95 batch
completion times are 6083.51 and 6070.97 microseconds. Those approximately
6 ms batch completions are not individual streaming service latencies.
All results and exact counts equal the scalar streaming policy.

## Anomalies, limitations and reproducibility

271 queries have exhaustive GT distance ties at ranks10/11. Primary ID-based
recall is unchanged. A separate capped boundary-tie-aware sensitivity gives
native/GAP/ALL16/oracle recall 0.850344/0.880676/0.940535/0.951017;
GAP minus RANDOM mean remains 0.007724 and selected harmful count remains0.
No test labels changed policy, threshold, L, model or measured implementation.

76 Python metric tests and five C++ tests pass. All timed output IDs and
decisions are checked after complete streams, preventing validation-label
traffic from entering the inter-query benchmark loop. Raw per-query timings,
output IDs, candidate arrays, 30 masks, environment, immutable source/model/
vector checksums, and all 56 primary/14 batch run records are retained.
Full analysis requires no new searches or training. `audit_phase4b.py` checks
raw recall/counts, regenerated masks and byte-identical tables/figures against
an independently regenerated output directory. Commands are in the experiment
log. Final audit **PASS**: 798,112 timed query traces, 57,008 policy/query recall
checks, 30 regenerated masks, identical derived query rows, and all29 table/
figure files byte-identical after independent regeneration. The artifact
manifest is `runs/phase4b_streaming_feasibility_v1/final_audit.json`.
NFS build clock-skew warnings and a temporary matplotlib cache fallback
did not prevent successful builds/tests/figure output.

Inference is limited to this one machine, fixed query order, warm resident
index, single-thread integration and one model. There are no cold-cache,
network, concurrency or deployment SLA claims. Exact-evaluation efficiency
and wall-clock efficiency must not be conflated.

## Decision and exactly one next experiment

The threshold and GAP-over-random recall advantage generalize. Actual selection
is 24.950884%; exact work falls75.0491% relative to ALL16, but latency falls
only1.2042% and retained recall gain is33.6724%. The measured policy therefore
fits **“generalizes but has little systems value.”** The risk signal is not
the immediate systems bottleneck. Candidate retention/extraction is meaningful;
the exact L16 kernel is already cheap, and batch32 does not repair the economics.
No learned uncertainty model is justified by this result.

**One next experiment:** a fixed-output ALL-L16 candidate-accessibility
ablation, comparing current full L0 materialization with bounded top16
retention during the same search. Freeze graph/PQ/queries/L/tie convention,
validate identical native and refined IDs against the current simple reference
before timing, and measure full latency plus scratch memory. This tests whether
candidate-level retention/selection overhead, rather than better query-risk
detection, is the actionable systems bottleneck; it does not propose a new
quantizer or reopen navigability.
