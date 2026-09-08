# Phase 4D — multithread bounded ALL-L16 validation

Status: complete. Results support preregistered **Case A** on this machine and
workload: the recall improvement survives1–8 physical-core query concurrency,
without a materially widening throughput gap or disproportionate tail problem.
Frozen design: `runs/phase4d_multithread_scalability_v1/preregistration.md`;
config: `configs/indexes/phase4d_multithread_scalability.conf`.

## Read-only / thread-local audit

Phase4C `bounded_retention.{h,cpp}` and `streaming_refinement.{h,cpp}` are
unchanged. Each worker owns CandidateAccess (no FULL storage), each query owns
its bounded heap, native result heap and PQ DistanceComputer. The latter's ADC
table is a local vector; ProductQuantizer::compute_distance_table reads the
same immutable codebook and writes that local table. No PQ model/code mutation.
VisitedTable::get_reusable uses C++ thread_local storage, not a global shared
visited array; its periodic epoch resets remain unchanged. HNSW::search and
search_level_0 keep local stats; the shared high-level IndexHNSW hnsw_stats
aggregation wrapper is not invoked. No graph storage pointer swapping.

The harness changes concurrency only. It uses the same low-level adapter,
parameters, codebook, base/query identity, tie handling and final16 refinement
as4C. Race-freedom is supported by source ownership review plus deterministic
and exhaustive-output stress checks, not claimed as a formal proof or TSAN run.

## Closed-loop workload and measurement

Workers1/2/4/8 use CPU prefixes[2,3,4,5,6,7,8,9], all distinct physical cores;
coordinator CPU0. Each worker issues one synchronous query at a time with no
think time, no dispatch queue, no query batches and no per-pass synchronization.
Worker w traverses IDs w,w+T,... and repeats its shard four times. Every cell
therefore covers each of the14252 queries exactly4 times (57008 actual query
measurements), preserving the original within-shard stream order. This is
intentional systems reuse, not another query-generalization experiment.

Before each cell, workers create private state and output buffers, warm512 old
Phase4A queries, then wait at a common launch barrier. Query-worker creation,
model/graph loading and warmup are excluded from timing. End-to-end QPS uses
all queries divided by time from release to the last worker's final output
collection. Per-query latency uses the **actual** steady_clock interval inside
the unchanged Phase4C API, including preemption during that call; it is not
computed from QPS. Post-stream ID comparisons, graph/model hashes and file I/O
are excluded. Static shards allow a short finite-workload drain phase, recorded
per worker; throughput is not labeled infinite steady-state capacity.

Each policy/concurrency has5 measured repetitions, all retained. Eight cells
are randomly reordered within each repetition (PCG64 seed94300011). Across
repetitions report mean, sample SD, min/max of every required metric. Main
latency points are means of per-repetition latency summaries; pooled ECDFs
are separate. Scaling uses ratios of mean QPS to the same policy's1-worker
mean. These repeated-query samples are not independent population observations.

OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, MKL_NUM_THREADS=1,
OMP_DYNAMIC=FALSE, OMP_MAX_ACTIVE_LEVELS=1. Each worker also explicitly sets
OpenMP thread settings, checks it is not in a parallel region, and verifies
single-CPU affinity. The ready gate checks live process thread count=T+1;
visited-table addresses must differ across workers. The coordinator is outside
measured query cores; source code launches no nested per-query workers.

## Correctness gate

Stress at all four worker counts covers61440 alternating native/ALL16 calls,
with same-query concurrent calls, worker-specific calls and consecutive query
repetitions. The deterministic512-query sample includes all210 known4C PQ
rank16-cutoff ties plus302 seeded queries. All returned IDs and ALL16 top16
ID/order/PQ-score bits match frozen4B/4C references; no observable state leak,
thread sharing or index mutation. This stress does not replace checking every
measured output: each primary/profile query is also checked after its stream.

One hundred thousand two-clock iterations measured0.03590133us/iteration,
including loop/output storage overhead. This diagnostic is not subtracted from
latencies. Main queries have the same one clock pair as4C, no stage/callback
timers. Six existing C++ tests also pass; Python scaling/latency/sharding tests
are added separately.

## Frozen interpretation and optional hardware diagnostics

Case A requires <=15% throughput penalty through8 workers, controlled p95/p99
and broadly comparable scaling efficiency. Case B is a widening throughput/
efficiency gap consistent with shared-resource pressure, not proof of memory
bandwidth causation. Case C is disproportionate tails despite attractive means.
No new numeric tail SLA is invented after results. Mixed outcomes are allowed.

Nominal additional FP32 payload is8192 bytes per refined query; this is not
measured DRAM bandwidth. Worker thread CPU time and per-cell load/CPU/memory/
frequency snapshots provide inexpensive diagnostics. A separate perf phase
after primary attempts1/8 workers for each policy. Counters are enabled only
after warmup/ready, disabled after measured workers finish and before validation
or file I/O; controller acknowledgements are required. It measures user-space
cycles/instructions/generic cache references/misses/task-clock, including small
final worker teardown. It is not an isolated refinement-kernel profile, LLC
occupancy measurement or bandwidth measurement. Unsupported counters will be
reported without blocking the primary experiment.

## Primary results

All40 cells completed, with **2,280,320** individually timed queries and exact
output checks. Every policy/worker point contains285040 measurements across
five repeats. Recall is unchanged: NATIVE **0.85030873**, bounded ALL16
**0.93997334**, improvement **0.08966461** (8.966 recall percentage points).
The tiny last-digit floating reduction differences across row orders are not
ID/recall-count differences. No outcomes or repeats were removed.

The following are means of the five repetition metrics; ± denotes sample SD,
not population confidence intervals. Latencies are microseconds.

| Workers | Policy | QPS mean ± SD | Mean latency ± SD | Median | p90 | p95 ± SD | p99 ± SD |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | NATIVE | 6118.50 ± 14.93 | 163.312 ± 0.399 | 165.235 | 185.144 | 191.491 ± 0.751 | 205.558 ± 0.956 |
| 1 | ALL16 | 5687.78 ± 32.39 | 175.698 ± 1.001 | 177.521 | 200.252 | 207.768 ± 2.414 | 223.714 ± 2.953 |
| 2 | NATIVE | 11892.92 ± 22.07 | 167.839 ± 0.248 | 169.894 | 190.015 | 196.074 ± 0.293 | 209.776 ± 0.411 |
| 2 | ALL16 | 10988.41 ± 21.11 | 181.497 ± 0.287 | 183.299 | 207.095 | 214.710 ± 0.732 | 230.820 ± 1.410 |
| 4 | NATIVE | 23076.26 ± 52.15 | 171.779 ± 0.310 | 173.935 | 194.337 | 200.470 ± 0.917 | 214.253 ± 1.149 |
| 4 | ALL16 | 21218.67 ± 80.96 | 186.388 ± 0.308 | 188.217 | 212.575 | 220.449 ± 0.611 | 236.823 ± 0.825 |
| 8 | NATIVE | 45165.50 ± 33.61 | 175.208 ± 0.083 | 177.674 | 197.506 | 203.366 ± 0.163 | 216.942 ± 0.511 |
| 8 | ALL16 | 41682.04 ± 216.52 | 189.198 ± 0.427 | 191.413 | 214.862 | 222.563 ± 1.400 | 240.508 ± 1.910 |

`results/tables/phase4d_aggregate.csv` provides **mean, SD, min and max for
every required metric**, including median/p90 and recall; repetitions and raw
query records are separate. For example8-worker QPS ranges45127.46–45203.79
for NATIVE and41434.71–41921.78 for ALL16; p99 ranges216.433–217.715 and
238.754–243.011us. These ranges are observed repeats, not guaranteed bounds.

### Scaling and relative costs

| Workers | NATIVE speedup / efficiency | ALL16 speedup / efficiency | Throughput penalty | Mean overhead | p95 overhead | p99 overhead / ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.000 / 100% | 1.000 / 100% | 7.040% | 7.584% | 8.500% | 8.833% / 1.0883 |
| 2 | 1.944 / 97.188% | 1.932 / 96.597% | 7.605% | 8.138% | 9.504% | 10.032% / 1.1003 |
| 4 | 3.772 / 94.289% | 3.731 / 93.264% | 8.050% | 8.505% | 9.966% | 10.534% / 1.1053 |
| 8 | 7.382 / 92.272% | 7.328 / 91.604% | 7.713% | 7.984% | 9.439% | 10.863% / 1.1086 |

All four throughput targets pass. Even the largest paired-repeat penalty is
8.347%, below15%. At8 workers, ALL16 retains92.287% of native throughput;
its parallel efficiency is only0.668 percentage points lower. Throughput
penalty rises0.673 points from1 to8, rather than progressively diverging.
The4-worker penalty is slightly higher than8-worker penalty, so a monotonic
contention claim is not supported.

The fresh1-worker mean overhead is **7.58%**, not Phase4C's5.85%. Engine/input
hashes and all result IDs are unchanged. Phase4D uses newly created pinned
workers, longer repeated streams and a different measurement session; allocator,
cache and frequency state may differ. Startup is excluded, so its direct time
cannot explain this difference. We do not identify a specific cause or combine
old native timing with new ALL16 timing. All concurrency claims use the
contemporaneous1-worker reference, showing only a small additional change.

### Tails, utilization and finite workload effects

The8-worker p99 cost increase is10.86%, versus7.98% mean increase, a modest
additional tail cost, not zero. Across paired8-worker repeats the p99 overhead
ranges9.66–12.03%. The largest latency in all primary raw samples is563.510us;
it is retained, not trimmed. These observations do not support Case C's
disproportionate tail breakdown. They do not certify an application SLA or
predict queuing latency under open-loop overload: this is closed-loop service
latency with at most one outstanding query per worker.

Average assigned-core utilization is approximately99.8% at1 worker and
98.79%/98.44% for native/ALL16 at8. No hidden nested query threads were seen:
each ready barrier has T+1 live threads, and all workers report OpenMP1 and
their assigned physical CPU. Maximum start-release lag is14.91us per **cell**,
not per query. The mean finite-stream last-worker drain fraction at8 workers
is1.56% native versus2.45% ALL16. Reported QPS includes this imbalance; it is
not corrected into an idealized throughput estimate.

## Shared-resource diagnostics: observations versus interpretation

ALL16 nominally reads8192 additional FP32 payload bytes/query. At measured
throughput this corresponds to0.0466/0.0900/0.1738/0.3415 nominal GB/s at
1/2/4/8 workers. This excludes cache-line effects, other search traffic and
evictions; it is **not measured DRAM bandwidth**, nor proof that bandwidth
cannot bottleneck another workload. The512MB FP32 side database is still
resident/required. Bounded heaps remain thread/query private; visited arrays
add approximately1MB per worker in both policies, with unchanged periodic resets.

All four optional gated perf runs succeeded, with100% event-running time
(no reported multiplexing loss). Primary results do not use these runs.
Representative user-space generic counters per query:

| Workers | Policy | Instructions/query | Cycles/query | Cache references/query | Cache misses/query |
|---:|---|---:|---:|---:|---:|
| 1 | NATIVE | 1,137,376 | 736,289 | 5457.6 | 3174.1 |
| 1 | ALL16 | 1,184,523 | 784,976 | 5782.6 | 3442.2 |
| 8 | NATIVE | 1,137,377 | 750,936 | 5670.0 | 3428.6 |
| 8 | ALL16 | 1,184,524 | 800,187 | 5952.7 | 3687.9 |

Additional instructions are approximately4.15% at either concurrency; ALL16
cache misses are approximately8.45% above native at1 and7.56% at8. Both
policies' cycles/query and cache misses increase with concurrency; ALL16 does
not show a uniquely widening counter gap. Gated task-clock totals closely
match summed worker CPU time (approximately0.2% difference, including
coordinator/teardown and scope differences). Perf's printed default “CPUs
utilized” divides by whole-process lifetime including disabled load/warmup;
**it is not our hot-loop utilization metric**. Use the thread CPU/makespan
measurements instead.

Interpretation: extra work/cache pressure is observable, but this is **not
evidence that refinement becomes a new memory/cache bottleneck through8 cores**.
Nearly matching efficiency and bounded relative penalties argue against
Case B here. These generic counters cannot isolate FP32 traffic or prove the
absence of bandwidth/LLC limitations. All-core frequency/power and common
search-memory effects remain possible contributors to both policies' slowdown.
No precise bandwidth claim is made and no privileged counter setup was needed.

## Environment and reproducibility

Execution Git `9a0c375c9151ec45bf7866c8d5b14e23198b48f6` plus archived harness/
binary hashes; FAISS `20f14b31a6d54e243a3d1de6ae193fc4c3ec18ed`. Original
Phase4C engine source and frozen graph/model/input hashes verified before
and after. Graph fingerprint:
`1b3c18f6e2dd3ccec56e74eed30458f60a65f01234bf51478af25d84b5c02d16`.
Shared in-memory PQ codebook/codes/cache-array hash also stays identical in
stress, primary and profiles. SIFT1M base1M x128 FP32; same Phase4A PQ64x8
model and HNSW M16/efConstruction80, search ef64, final k10/L16.
Inherited model provenance (no retraining): training65536 vectors, learning
split seed94000011, PQ initialization94000037,25 PQ iterations and24 historical
training threads, as in `configs/indexes/phase4a_gap_guided_refinement.conf`.
Historical graph construction seed20260907/24 construction threads is unchanged;
these are not online thread settings. Online order/stress seeds are94300011/
94300037, and actual query IDs/checksums remain those of4B.

Intel Core i9-10920X @3.50GHz,12 physical/24 logical CPUs, single NUMA node.
L1d/L1i32KiB/core; L2 1024KiB/core; shared L3 19712KiB; RAM32273192KiB.
Workers2–9, coordinator0; SMT siblings14–21 not assigned. GCC11.5.0, same
Release app flags `-O3 -DNDEBUG -std=c++20 -Wall -Wextra -Wpedantic -fopenmp`;
generic CPU FAISS `-O3 -DNDEBUG -std=gnu++20 -fPIC -fopenmp`, FINTEGER=int.
Threads library links the harness; the query engine is unchanged.

Performance governor/turbo settings unchanged. Frequency snapshots are not
active frequency locks. Per-cell one-minute host load observations range
1.77–3.13; load averages are not instantaneous active-worker counts. A
host-visible precheck observed no heavy competitor, but per-phase process
snapshots are namespace-local and the host is not exclusively reserved.
No agent-owned heavy task ran during primary/profile measurement. Build NFS
clock-skew warnings are retained; no compile error or measured-run discard.

86 Python tests,6 C++ tests and the61440-call concurrency stress pass. Every
primary/profile ID and ALL16 top16 reference is validated after the stream.
Raw per-query CSVs, returned ID arrays,40 primary cell metadata/snapshots,
stress/profile records and all commands are preserved under
`runs/phase4d_multithread_scalability_v1/`. Tables report repetitions plus
mean/SD/min/max; seven figures include all required comparisons. Analysis
regenerates from raw runs without new searches. The experiment log records
exact commands, including actual perf FIFO command paths in environment JSON.

Scope limitations: one dataset/model/machine, warm resident index, intentional
query reuse, static deterministic sharding, no multi-NUMA/SMT/open-loop network
load, and no formal data-race detector. Repeated query passes improve tail
measurement density but are not independent external-validity replications.
No GAP policy, learned predictor or further risk optimization was run.

Final independent audit **PASS**:61440 stress,2,280,320 primary and228032
profile returned-ID/count checks; deterministic order regenerated; all frozen
input/source/binary hashes and shared in-memory graph/PQ hashes unchanged;
all28 table/figure files byte-identical after independent regeneration.
Artifact manifest: `runs/phase4d_multithread_scalability_v1/final_audit.json`.

## Mandatory decision and exactly one next experiment

1. **Similar scaling:**8-worker speedup7.382 native versus7.328 ALL16,
   efficiency92.27% versus91.60%.
2. **Overhead stays small:** throughput penalty7.04–8.05%;8-worker7.71%.
   Mean latency overhead7.58–8.50%, with no sustained widening.
3. **No demonstrated new memory/cache bottleneck:** counters show extra
   work/misses but no growing ALL16-specific scaling penalty. Not proof of
   immunity on larger/high-dimensional or multi-socket workloads.
4. **Tails remain controlled in this closed-loop test:** p99 ratios1.088–1.109;
  8-worker p99 overhead is modestly above its mean overhead, not pathological.
   Application acceptability still depends on an explicit SLA and arrival model.
5. **Recall/throughput tradeoff remains attractive:** at8 workers, recall
  0.93997 versus0.85031 for41.7k versus45.2k QPS. Approximately9 recall points
   survive at approximately7.7% throughput cost, not a free improvement.
6. **Case A is supported:** bounded shallow exact refinement is a viable
   systems primitive worth further external validation, not yet a general
   production or novelty claim. Do not return to query-level risk prediction.
7. **One next experiment:** an external-validity gate on one preselected
   high-dimensional real embedding dataset, comparing NATIVE and unchanged
   bounded ALL-L16 at1/8 workers after independently fixing a reasonable PQ/HNSW
   operating point. Preserve offline candidate-oracle recall as a control and
   jointly measure recall,QPS,p95/p99. This tests whether both shallow-refinement
   recovery and its low concurrent cost survive beyond low-dimensional SIFT,
   without selecting a dataset or tuning the policy to favorable test outcomes.
