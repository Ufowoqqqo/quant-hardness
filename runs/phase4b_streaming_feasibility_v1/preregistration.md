# Phase 4B preregistration, before new query measurements

Frozen L16, raw g9_12_pq, threshold **710.40625**, inclusive <= comparison.
This value comes only from Phase4A: both its linear 25th percentile and its
last selected rank2500 score equal 710.40625. A boundary tie includes rank2501
under <= (25.01% in Phase4A). Do NOT reproduce Phase4A's ID-based quota or
recompute a percentile on Phase4B. Numeric threshold, source query-score and
selection hashes are recorded before preparing or scoring B queries.

Remaining 14252 IDs are the suffix of Phase4A's exact eligible-ID permutation
after its 65536 training and 10000 query records. Verify those two prefixes
against recorded IDs. No new training, graph construction or query tuning.
Warmup uses first512 Phase4A queries, not any of the measured B queries.

Correctness stage first executes actual streaming GAP logic to establish its
real count and validate against native FAISS/legacy corrected L0 semantics.
This is not an offline score-mask policy: every timed GAP query repeats its
own search, score extraction, threshold comparison and exact refinement;
never reads an offline GAP decision. Only RANDOM uses an independent mask.
RANDOM masks: PCG64 SeedSequence([94100037,replicate]), 30 permutations,
prefix matching actual GAP count. Latency mask is fixed replicate0; selection
does not use content. Recall for all30 is replayed from validated per-query
native/top16 results. No mask or policy chosen for its measured recall.

Single-thread, CPU2 pinned. Seven repetitions; shuffle eight conditions
(four policies x component-clocks on/off) independently per repetition with
PCG64 seed94100011. Warm up512 old queries before every condition. Same B
query order for every policy. Primary latency uses component clocks OFF,
two steady-clock calls bracketing the whole online call. Component ON runs
provide t_search,t_gap,t_refine,t_final_selection and overhead control. No
logging, validation, graph hashes, index loading, training, ground truth,
oracle reranking or file IO in timed sections. Result/policy extraction,
scratch growth, retention, top16 selection, branching, exact16 distances,
final top10 and query-local cleanup ARE included. Report repetition variation,
per-query distributions and service QPS; not network/request queue latency.

Native API retains k10 result heap, not all evaluated candidate scores. The
new adapter calls original FAISS upper greedy and L0 search implementations;
it records unique L0 IDs/scores with reusable epoch tags and vectors. Upper
evaluations excluded, cached L0 seed admitted exactly once (same candidate
semantics as previous re-evaluation-based recorder). No second traversal or
ADC rescoring. Keep k10 even for ALL16; do not replace with k16 search.
Simple full-sort reference verifies partial top16 ordering on warmup and all
untimed test exports before benchmarking. No performance changes after
observing timings. Native uses same low-level FAISS single-query interface
without retention; public IndexHNSW::search identity is checked separately.
RANDOM unselected queries skip retention/sorting, as their content-independent
decision is available before search. GAP must retain candidates on ALL queries;
ALL always retains, but does not need the gap/threshold decision. Charge this
asymmetric, required online overhead honestly. Buffers/epoch tags are reused
and their sizes reported, not treated as free storage.

Short pools <16: stop before policy interpretation rather than silently change
L/budget. Native/global score and exact GT boundary ties are kept; primary
ID Recall@10 unchanged. Separately named tie-aware sensitivity uses strict
hits plus capped boundary ties, as Phase4A. Any strict-distance discrepancy
stops interpretation. Full oracle is offline only; primary raw candidates are
PQ L0 evaluated IDs and original ADC scores, not FP32 traversal pools.

Offline exhaustive GT uses existing FAISS IndexFlat direct FP32 kernel
(distance_compute_blas_threshold=INT_MAX, 24 threads) rather than the slow
reference-BLAS route. Same squared-L2 metric, all base vectors; explicitly
validate deterministic100 queries against direct FP64 all-base distances.
This offline backend choice never enters benchmark latency. No competing
agent-owned heavy jobs during timings; record load/context/frequency before
and after; do not kill unrelated jobs or change governor/turbo. If a shared
machine remains noisy, report all repetitions rather than cherry-pick.

Success criteria unchanged: useful systems signal requires GAP recall above
matched RANDOM, >=20% mean latency saving vs ALL16 AND >=60% of ALL16 gain
over native. Strong additionally improves recall/latency Pareto frontier and
has nonpathological tail latency (qualitative, report actual quantiles).
Generalized recall but small runtime saving or ALL16 much better for a small
latency premium means little systems value. No p-value/publication claims.
Use arithmetic mean of equal-length repetition mean latencies for saving;
report per-repetition ratios as variability, no trimming slow repetitions.
Recall comparison uses random mean and empirical 95% interval (30 masks).

Only after primary benchmark completes: batch32 diagnostic, seven repetitions,
randomized scalar-deferred vs FAISS pairwise_indexed_L2sqr order (94100083),
same threshold/model/L and single thread. Both modes retain decisions from
fresh searches of that batch and defer exact16 refinement, then return
results. Include buffer/index-list construction and final selection. Compare
whole-batch amortized service time and refine stage; explicitly NOT single
query response latency. Last short batch retained. Validate batched exact
distances against scalar reference on old warmup queries before any timing.
No new learned model, optimized weights, candidate-level policy or quantizer.
