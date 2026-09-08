# Phase 4D preregistration — before concurrent results

Only NATIVE and BOUNDED_ALL16, unchanged Phase4C CandidateAccess / BoundedTop16
code, same SIFT1M graph/PQ64x8/ef64/k10/L16 and Phase4B14252 queries. No GAP,
training, GT recalculation or query-risk optimization. Same compiler/flags.
Existing output IDs, top16 score/ID/order references are mandatory controls.

Hardware audit: i9-10920X,12 physical cores,24 logical CPUs, one NUMA node.
Workers1/2/4/8 pinned to prefixes of physical CPU list[2,3,4,5,6,7,8,9];
SMT siblings14–21 are not assigned. Coordinator CPU0. No frequency/governor
changes or unrelated process termination. Shared host, not exclusive isolation.
One thread per query worker, own CandidateAccess, per-call PQ distance table,
thread_local FAISS visited table. Shared graph/PQ/base are read-only. No calls
to the high-level wrapper's shared global hnsw_stats updates. OMP/OpenBLAS1,
OMP_DYNAMIC=FALSE, OMP_MAX_ACTIVE_LEVELS=1, worker omp_get_max_threads1.
Verify live thread count at the ready barrier equals workers+coordinator.

Closed loop: at most one outstanding query per worker, no think time. Worker w
receives query IDs w,w+T,... in original order, repeating the shard four times.
Across all workers every query is executed exactly four times per repetition;
unequal final shard lengths (at most one original query) are retained. No queue
dispatcher, within-stream barriers or query batches. Warm512 old queries/worker
before a synchronized measurement start. Pool construction, index load, warmup,
validation and output writing are outside measurement. Worker-local output
collection is inside the aggregate throughput window. No label checks between
timed queries. Shared completion window ends at last worker's last output.
Record worker drain/imbalance instead of pretending infinite steady state.

Five repetitions of all eight policy/worker conditions, independently shuffled
within repetition with PCG64 seed94300011. Four query passes per cell gives
57008 actual latency samples/cell,285040 per policy/concurrency. These are
repeated queries, not independent query-population samples. Preserve every
repetition (not fastest), actual per-query steady-clock latency from unchanged
Phase4C run(clock_level0), worker elapsed/thread CPU time, overall QPS.
Calibrate a100000-pair steady_clock loop separately; primary contains no
component/per-candidate timers. Do not estimate latency as reciprocal QPS.

Correctness pre-gate: deterministic512-query sample seed94300037 plus known
PQ cutoff-tie queries where possible; all worker counts execute4096 alternating
native/all16 calls/worker, with identical simultaneous queries, different
queries and consecutive repeated queries. Compare to frozen4B native/all16
and4C top16 references. Retain stress IDs and schedules. Every primary output
also validated after completion. Stop on semantic mismatch, no altered policy.

Primary summaries: every run's recall,QPS,mean/median/p90/p95/p99; across runs
mean/sampleSD/min/max for EACH metric. Scaling uses ratio of mean QPS to that
policy's one-worker mean; efficiency=speedup/T. Relative throughput penalty
and mean/p95/p99 overhead compare contemporaneous policy means at each T.
Also retain paired-repetition relative comparisons and pooled ECDFs, without
calling repeated samples population CIs.

Case A: retain most native throughput; <=15% throughput penalty at every
worker count through8, controlled p95/p99, broadly similar scaling efficiency.
Case B: low single-thread overhead but widening penalty and materially worse
ALL16 efficiency: shared-resource hypothesis, not proven memory attribution.
Case C: attractive mean/QPS but disproportionate tails; require tail-focused
work before claims. No new numeric tail cutoff is invented after results:
report p95/p99 ratios against mean ratios, repeat spread and no universal SLA.
Mixed/indeterminate outcomes are allowed. No threshold changes after results.

Memory diagnostics: nominal ALL16 additional FP32 payload8192 bytes/query,
not measured DRAM traffic; thread CPU utilization and machine snapshots.
perf is installed and a read-only task-clock/cache-misses smoke test succeeded.
After primary, attempt one separately labeled gated perf diagnostic for each
policy at1 and8 workers (same four passes). Enable after warmup/ready barrier,
disable after measured workers finish before validation/file I/O, using perf's
control FIFO acknowledgement. User-space cycles,instructions,cache-references,
cache-misses,task-clock only; not precise bandwidth counters. Skip/record failure
if unsupported; counter collection must never block primary conclusions.
Counter-run latency does not replace primary wall-clock samples. No tuning.
