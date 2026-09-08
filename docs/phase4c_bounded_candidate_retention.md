# Phase 4C — bounded top16 candidate retention

Status: complete. Candidate-access optimization passes its overhead-reduction
gate; query-level selective refinement still fails its practical-value gate.
Frozen design/thresholds: `runs/phase4c_bounded_candidate_retention_v1/preregistration.md`.

## Source audit before optimization

`src/instrumentation/streaming_refinement.cpp`: Capture wraps native PQ
DistanceComputer scalar/batch4 calls, epoch-tags deduplicate L0 IDs, ID/score
vectors append only first arrivals. Upper greedy navigation uses unwrapped ADC;
its final seed/score is inserted once into the L0 set. ALL16 uses the full
unique evaluated population, not popped nodes, candidate-queue membership or
native returned k10. Thus bounded retention targets exactly the existing pool.

After search, a preallocated ordinal vector is filled with iota and partial
sorted for16; ties use first-arrival ordinal. Selected IDs are copied into
fixed output arrays,16 direct FP32 rows are evaluated,16 exact values sorted
for final10 with ordinal tie-breaking. No full FP32 candidate matrix copy.
Full scratch:4MB epoch tags plus three capacity2048 four-byte arrays. Vectors
are reserved once, then reused; actual growth will be checked. Shared FAISS
search/DC/heap allocations remain in every mode, not attributed to the recorder.

Pinned FAISS HNSW.cpp search_from_candidates_impl marks seed visited and calls
vt.set on neighbors BEFORE query distances_batch_4/scalar calls. This prevents
duplicate callbacks for this L0 search path. The collector still handles
consistent-score duplicate streams defensively with resident-ID checks.
Bounded cutoff only improves: rejected/evicted IDs cannot later qualify with
the same score. First-arrival ordering among surviving items is preserved.

Candidate storage writes can affect cache behavior during search. The new
paired FULL versus BOUNDED versus NATIVE measurements will capture combined
path overhead, not claim precise DRAM/cache attribution. Callback timers will
be explicitly separated from primary no-internal-timer results.

## Implementation and equivalence gate

The historical `StreamingRefinement` source is unchanged. A separate
`CandidateAccess` research adapter in `bounded_retention.cpp` uses the same
FAISS APIs and native k10 heap: upper greedy navigation, then search_level_0,
with the cached seed admitted once. NATIVE still calls ordinary HNSW::search.
The FULL collector reproduces the epoch-tags/vector/partial-sort reference;
the BOUNDED collector changes only storage of already-computed ADC results.
FAISS source, graph topology, PQ models/codes and efSearch are unmodified.

`BoundedTop16` uses a fixed16-element max-heap of (float score, int32 ID,
uint32 arrival ordinal). The worst score/ordinal is the root. A candidate
which cannot improve the root is discarded; eligible resident duplicate IDs
are ignored. Late equal scores cannot displace earlier scores. Sorting a
copy of these16 entries materializes final ascending order; there is no
second sort over the full population or per-candidate dynamic allocation.
This is defined for deterministic finite per-ID ADC, as in the frozen index.

Uniqueness proof has two levels: FAISS's visited-before-distance path prevents
duplicates here; defensively, the heap rejects resident duplicates and a
previously rejected/evicted ID cannot improve a monotonically improving cutoff
when its score is unchanged. Synthetic tests compare every prefix against a
simple first-ID-deduplicated full stable sort, including dense ties, repeat
IDs, evicted/reappearing IDs and reset/short-pool handling (200 fixed seeds).

Before timing, all14252 queries were checked in six modes and three clock
levels (256536 checks), against both the untouched4B executable logic and
the stored4B candidate arrays/results. Ordered top16 IDs, PQ score bits,
arrival ordinals, native ID/score bits, ALL16 exact score bits and final IDs
match. All210 rank16/17 PQ boundary-tie queries match without a tie exemption.
Full pool ID/score order is unchanged; bounded callback counts equal unique
full-pool sizes. Frozen GAP scores, decisions and results also match in
correctness checks; its **performance diagnostic** runs only after primary.
Graph fingerprint matches before/after; heap size208 bytes, full scratch
4,024,576 bytes, no candidate-capacity growth in validation.

## Timed design and interpretation rules

Modes FULL/BOUNDED both extract top16 but return native top10 without exact
refinement. FULL_ALL16/BOUNDED_ALL16 perform the same16 direct FP32 distances
and final exact top10. All extra top16 extraction/materialization is timed.
FULL remains a semantic and implementation-pattern reference rather than
using an old timing number: all five modes are measured contemporaneously
through the new shared adapter. Both collectors expose the same output fields.

Primary:7 repetitions x5 modes x2 clock levels, randomized order seed94200011;
512 old-query warmups precede each whole14252-query stream. No labels, result
comparisons or file writes occur between measured calls; validation follows
the completed stream. Per-query API latency and whole-stream QPS are retained.
All runs remain in summaries, without trimming or result-driven revisions.

OFF has only an outer total-clock pair. StageON times search (including
retention), candidate finalize, exact16, and final topk/cleanup. Detail clocks
around each ADC-result capture batch/scalar insertion are deliberately
intrusive, run separately after primary and GAP diagnostic. `t_search_core`
is directly observable for NATIVE; search-minus-callback time for FULL/BOUNDED
still includes timer/cache effects and is **not** a counterfactual pure core.
Speedup and overhead-reduction claims use OFF, never these subtractions.

Secondary frozen GAP uses seven randomized repetitions (seed94200037) with
contemporaneous NATIVE/BOUNDED_ALL16 controls, OFF and StageON. Fixed threshold
710.40625, L16 and all inputs remain unchanged. No threshold calibration.

Optimization passes if bounded-overhead <=50% full-overhead (positive
denominator required), OR bounded ALL16 mean latency improves >=10%.
Selectivity is useful only if bounded GAP saves >=10% mean latency versus
bounded ALL16 AND retains >=60% of ALL16's gain over native. These are
exploratory engineering gates, not statistical significance claims.

The preregistered cases are A (candidate access was a major bottleneck), B
(bounded access makes GAP practically useful), or C (little timing effect).
Intermediate/mixed outcomes will be reported without forcing B.

## Results: primary five-mode comparison

All numbers below use component clocks OFF, pooling equal-size seven repeats.
Latency units are microseconds. QPS is measured whole-stream throughput,
including output collection but excluding post-stream validation and file I/O.

| Mode | Recall@10 | Mean | Median | p90 | p95 | p99 | Stream QPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| NATIVE | 0.850309 | 162.832 | 164.732 | 184.347 | 190.871 | 205.501 | 6139.30 |
| FULL (no exact refinement) | 0.850309 | 184.732 | 187.114 | 211.488 | 218.790 | 235.285 | 5411.95 |
| BOUNDED (no exact refinement) | 0.850309 | 169.109 | 171.157 | 191.943 | 198.537 | 213.117 | 5911.81 |
| FULL_ALL16 | 0.939973 | 186.708 | 189.197 | 212.870 | 220.047 | 236.492 | 5354.64 |
| BOUNDED_ALL16 | 0.939973 | 172.359 | 174.083 | 196.069 | 203.597 | 219.408 | 5799.83 |

Service QPS (inverse mean API latency) is respectively 6141.31,5413.24,
5913.35,5355.96,5801.86. Repetition-mean SD is 0.275,0.785,0.763,0.463,
0.429 microseconds respectively; raw distributions and repetition ranges are
in `phase4c_policies.*` / `phase4c_repetitions.*`.

Measured paired overheads:

```
full_overhead    = 184.732080 - 162.831814 = 21.900266 us
bounded_overhead = 169.108819 - 162.831814 =  6.277005 us
overhead_reduction = 71.338225%

ALL16 mean latency: 186.707853 -> 172.358667 us
ALL16 saving = 7.685368%
ALL16 speedup = 1.083252x
```

The **<=50% residual-overhead gate passes** (28.66% remains). The alternative
**>=10% end-to-end ALL16-saving gate does not pass**. The registered OR criterion
therefore passes, without changing either threshold. Per-repeat overhead
reduction ranges64.48–73.75% (SD3.25 percentage points); ALL16 latency saving
ranges7.28–8.00% (SD0.25 points). No repeat was removed.

Bounded ALL16 costs 9.527 microseconds, approximately **5.85%**, above native
while improving recall by **0.089665**. It is close in cost, not literally free.
Compared with FULL_ALL16, its p95 falls220.047->203.597 and p99
236.492->219.408 microseconds. No new tail blowup appears in this workload;
this is not an application-SLA guarantee.

## What costs were removed?

Stage-clock diagnostic means (not the primary speedup estimator):

| Mode | Search including collector | Candidate finalize | Exact16 | Final topk/cleanup |
|---|---:|---:|---:|---:|
| NATIVE | 162.874 | 0.016 | 0.019 | 0.211 |
| FULL | 181.568 | 3.152 | 0.019 | 0.225 |
| BOUNDED | 168.468 | 0.245 | 0.017 | 0.215 |
| FULL_ALL16 | 181.103 | 3.142 | 2.122 | 0.396 |
| BOUNDED_ALL16 | 168.769 | 0.245 | 2.093 | 0.384 |

The non-refining modes' small exact-stage clock intervals contain **zero**
exact distance calls; they are stage-boundary/branch overhead, not refinement.
Full extraction/iota/partial sort/materialization costs approximately3.15us;
heap final materialization costs approximately0.245us. The ALL16 exact-distance
stage remains approximately2.1us in either implementation. Most measured
reduction occurs inside search-with-collector, consistent with avoiding full
tags/vector writes and changing retention work. Without hardware profiling,
their individual contributions or cache-bandwidth causes are not identifiable.

Intrusive callback-clock diagnostics record FULL capture23.593us/query and
BOUNDED heap maintenance8.977us/query, but raise total latency to200.048 and
181.130us, versus stage-only184.965 and168.944. These are **timer-perturbed
diagnostics**, not estimates of uninstrumented exclusive recording costs.
Search minus those measured intervals is173.002/171.665us and still contains
clock/cache effects; it cannot be relabeled pure FAISS core latency. Native's
search stage is the uncontaminated *no-collector reference*, not a subtraction
proof of full-path core time. This distinction prevents attributing high-frequency
timestamp overhead to the collector.

StageON−OFF mean changes across primary modes are +0.288,+0.233,−0.165,
+0.056,−0.868us (largest magnitude0.504%). Their repeat-paired SDs are
0.544–0.868us. Negative changes reflect run-order/cache/frequency variability,
not negative clock cost. OFF confirmation shows the optimization without
internal timers; the conclusions do not depend on intrusive instrumentation.

## Memory and allocation accounting

The unique evaluated pool has mean1034.584, median1076.5, p951337,
minimum216 and maximum1920. These count actual L0 evaluations, not popped
or queued nodes. In NATIVE timing rows candidate count0 means **not recorded**,
not zero search distance evaluations.

| Incremental collector state | FULL | BOUNDED |
|---|---|---|
| Retained IDs/scores | all pool members (mean1034.584 each) | at most16 each |
| Live ID+score bytes | mean8276.67; p9510696 | 128 |
| Ordering metadata | full ordinal array, mean4138.34 bytes | 64 bytes |
| Heap including counters/padding | not applicable | 208 bytes total |
| Final sorted temporary | full reusable ordering array | 192-byte fixed16 copy |
| Extra all-database duplicate tags | 4,000,000 bytes | none |
| Reserved full arrays+tags | 4,024,576 bytes/worker | none |
| Incremental collector dynamic allocations/query | 0 after reserve | 0 |

The heap is208 bytes, not the entire process or query frame: its192-byte sorted
copy and common fixed output arrays also exist. Shared FAISS visited-table,
result-heap and DistanceComputer setup/allocation remain in every mode. FULL
allocates its tag array and three vectors during setup; none grow across this
stream. Allocation counts are source/capacity accounting, not malloc profiling.
The benchmark process retains a dormant FULL reference object even in bounded
runs, so these numbers are avoidable per-worker collector state, **not a
measured RSS drop** of the benchmark process.

Both ALL16 implementations physically read the same16 FP32 vectors/query:
8192 nominal payload bytes/query,228032 evaluations over the stream.
Refinement still needs the512,000,000-byte FP32 database side storage; no
resident-vector-memory saving is claimed. No full candidate-vector matrix is
materialized or second graph search performed. GAP makes56896 exact calls,
3.992141/query; these counts are unchanged from4B.

## Secondary: frozen GAP after primary completion

Separate contemporaneous controls (OFF, seven repetitions):

| Mode | Recall@10 | Mean us | Median | p90 | p95 | p99 | Stream QPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| NATIVE | 0.850309 | 162.829 | 164.731 | 184.578 | 190.992 | 205.221 | 6139.45 |
| GAP_BOUNDED | 0.880501 | 169.908 | 171.844 | 192.831 | 199.643 | 214.806 | 5884.09 |
| BOUNDED_ALL16 | 0.939973 | 172.556 | 174.277 | 196.260 | 203.698 | 219.276 | 5793.79 |

GAP still selects exactly3556/14252 queries (24.950884%) with the original
710.40625 threshold. Its mean latency is only **1.534510% lower** than bounded
ALL16. It retains **33.672431%** of ALL16's recall gain. Both selective-value
conditions (10% and60%) fail. Output equivalence fixes the recall fraction;
implementation optimization could not change that fact without changing the
method, which this phase expressly forbids.

The gap/finalize stage is0.245us for bounded GAP, exact stage0.550us averaged
over all queries, and final selection0.267us. Making the uncertainty score
accessible is cheaper now, but skipping exact16 still saves little total time.
Uniform ALL16 adds only2.648us over GAP while adding0.059472 recall.

All three diagnostic points are mathematically nondominated: GAP remains an
intermediate recall/latency point. **Do not say ALL16 strictly dominates GAP.**
Practically, native is the minimum-latency endpoint, and bounded ALL16 is the
strong high-recall reference at approximately6% latency premium over native.
The frozen GAP middle point does not meet the registered useful tradeoff.

## Environment, artifacts and limitations

Execution Git `a9591c9` plus immutable source/binary hashes; FAISS
`20f14b31a6d54e243a3d1de6ae193fc4c3ec18ed`. Frozen graph fingerprint
`1b3c18f6e2dd3ccec56e74eed30458f60a65f01234bf51478af25d84b5c02d16`.
Dataset/query/model paths, checksums and full revision are in provenance.
Index remains HNSW M16/efConstruction80, PQ64x8, standard SIFT1M dimension128.

Intel Core i9-10920X,12 physical/24 logical cores,32,273,192 KiB RAM;
CPU2 affinity, one OMP/OpenBLAS thread. GCC11.5.0; app Release
`-O3 -DNDEBUG -std=c++20 -Wall -Wextra -Wpedantic -fopenmp`;
generic CPU FAISS `-O3 -DNDEBUG -std=gnu++20 -fPIC -fopenmp`, FINTEGER=int.
Performance governor and enabled turbo unchanged. Approximate1.2GHz snapshots
are **idle**, not measured active frequency locks. Load before1.20/1.24/1.39,
after primary1.98/1.62/1.52. Host-visible precheck found no heavy competitor;
runner process snapshots are namespace-local, not continuous host monitoring.
No agent-owned heavy worker ran alongside the benchmark. No exclusivity claim.

81 Python tests and6 C++ tests pass. No full trajectory logging, quantizer
training, learned model, new GT or search change. All raw per-query latencies,
returned IDs, top16 references,114 schedules/run records, source/input hashes
and environment are under `runs/phase4c_bounded_candidate_retention_v1/`.
Analysis consumes raw runs and historical immutable inputs only; commands and
reproduction audit are in the experiment log. Original4B raw results/source
remain intact. Primary recall retains the existing ID-based GT convention;
the210 PQ cutoff ties are distinct from GT ties and are handled identically.

Limitations: single machine/model/warm resident index, fixed query order and
intentional workload reuse. No new held-out inference or production concurrency
claim. Stage timing differences cannot precisely identify cache effects.
Existing FAISS per-query overhead remains; this is not optimization of its
search algorithm. A premature build before asynchronous CMake configure finished
reported no target; the subsequent build succeeded before all validations.
NFS clock-skew warnings were retained. No scientific run was discarded and no
implementation was revised after seeing performance outcomes.

Final audit **PASS**: all1,624,728 timed query outputs/counts,14252 independent
stable-full-sort top16 references, unchanged input/source/binary hashes,
identical derived query rows, and all21 table/figure files byte-identical after
independent regeneration. Manifest:
`runs/phase4c_bounded_candidate_retention_v1/final_audit.json`.

## Decision and exactly one next experiment

1. FULL candidate access cost21.900us above native; full finalization alone
   approximately3.15us. Its cache/recording effects are included, not isolated.
2. Bounded16 reproduces every ordered top16 and ALL16 final ID, including ties.
3. It removes71.34% of overhead: a meaningful candidate-access optimization.
4. New primary ALL16 latency172.36us (same recall0.939973); saving7.69%, below
   the alternative10% end-to-end threshold.
5. Frozen GAP is169.91us in its separately matched diagnostic; same recall0.880501.
6. Query selectivity remains not worthwhile by the frozen thresholds: only1.53%
   faster than uniform,33.67% gain retained. **Do not invest further in query-level
   risk prediction on this evidence.** Uniform refinement is close to native's
   cost, though its approximately6% premium is not zero.
7. Evidence supports **Case A for candidate-access overhead**, with modest
   whole-query improvement; not Case B. Native and uniform bounded ALL16 are
   useful low-latency/high-recall endpoints. GAP is formally nondominated but
   a weak practical middle point, not grounds to restart risk-model development.
8. **One next experiment:** a pinned-worker concurrency scaling gate comparing
   NATIVE and BOUNDED_ALL16 at1/2/4/8 independent single-thread query workers,
   same frozen graph/PQ/query IDs and exact outputs. Measure whole-workload QPS
   and per-query p95/p99 with paired repeated runs. This tests whether uniform
   refinement's small single-thread cost survives shared memory/cache pressure
   before treating it as a practical deployment choice. No new risk model or
   policy tuning.
