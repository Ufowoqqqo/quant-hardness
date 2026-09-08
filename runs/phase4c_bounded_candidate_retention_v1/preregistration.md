# Phase 4C preregistration — before implementation benchmarks

Use unchanged Phase4B graph/PQ/query/warmup arrays, ef64/k10/L16, frozen
GAP threshold710.40625 inclusive. No training, new GT, or search algorithm.
Intentional systems reuse of all14252 Phase4B queries, not a generalization test.

Candidate semantic population: unique distance-evaluated L0 nodes plus cached
L0 entry, excluding upper-only nodes. Score ties use first-evaluation order.
PQ ADC calls, batches and their ordering remain FAISS's own. Bounded top16
must reproduce ordered reference top16 IDs/scores and final ALL16 IDs on
EVERY query, with no relaxed tie exception if exact agreement is possible.
If equivalence fails, stop before timings and report, not benchmark a changed
method. Native IDs/scores and graph/model hashes must also match.

Modes: NATIVE, FULL (record+extract without refinement, native output), BOUNDED
(heap+materialize without refinement, native output), FULL_ALL16, BOUNDED_ALL16.
Seven repetitions each,512 old warmups before each, fixed stream order, CPU2,
one OMP/OpenBLAS thread. Randomize five modes x clocksOFF/stageON within each
repetition using PCG64 seed94200011. No result-driven implementation revisions.
Primary timings OFF include all required retention/extraction/refinement,
query-local setup/cleanup. StageON uses only boundary clocks.

Per-callback retention-clock diagnostics run only AFTER primary, one whole
stream for FULL and BOUNDED, labeled intrusive/not used for speedup claims.
Their subtraction from stage search time is descriptive, not a counterfactual
native-core estimate; cache effects and timer costs cannot be subtracted away.
Native/full/bounded paired end-to-end overhead is the primary measurement.

Only AFTER primary completion run frozen GAP_BOUNDED (OFF/stageON) with
contemporaneous NATIVE and BOUNDED_ALL16 controls, seven repetitions and
independently randomized order seed94200037. No new policy selection.

Case A: most full overhead removed and bounded ALL16 materially faster,
outputs unchanged. Case B: bounded GAP becomes practically useful relative
to bounded ALL16. Case C: little timing improvement, deprioritize query gating.
Do not force B; quantify intermediate results if no case exactly fits.

Meaningful retention optimization if bounded_overhead <= .50*full_overhead
(only if full overhead safely positive), OR bounded ALL16 mean latency falls
>=10%. Selectivity worthwhile only if GAP mean latency falls >=10% versus
bounded ALL16 AND retains >=60% of ALL16 gain over native. Thresholds will not
change after results. All costs use contemporaneous controls, not old4B timings.

Before implementation audit: FULL epoch-tags vector (4*1M bytes), preallocated
2048 ID/score/order arrays; unique inserts only, order_.iota+partial_sort16;
no vector data materialization, just16 FP32 row reads. Candidate pushes cause
capacity growth only above2048; check actual capacities and counts. Extra tag
and array writes can affect traversal caches; no precise attribution without
profiling. Phase4B means: search178.719us, extraction3.056us, exact16 2.166us.
These historical observations
motivate but do not replace new matched measurements.

Implementation: a separate research adapter preserves old Phase4B source.
Bounded heap stores(score,ID,first-arrival ordinal), max16 entries; compare
score then ordinal. Equal-score late arrivals cannot replace earlier ones.
No full tags or full candidate vectors in bounded execution. Resident-ID
duplicate checks plus monotonic cutoff suffice for deterministic per-ID ADC:
once evicted/rejected, the same score cannot re-enter. FAISS already marks
neighbors via vt.set before batch/scalar query ADC, so actual L0 callbacks
are unique in this frozen path. Seed admitted exactly once. Test both this
actual invariant and synthetic duplicate streams with consistent scores.

Preserve per-query latency and IDs, result equivalence evidence, raw benchmark
order, hashes, environment, per-repetition statistics and all prior runs.
