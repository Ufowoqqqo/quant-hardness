# Candidate-set instrumentation semantics

## Why the definition changed

Phase 1 and Phase 2A recorded every database ID passed to the FAISS
`DistanceComputer` during a complete HNSW query. That legacy all-level set
includes the entry-point and greedy navigation at levels greater than zero as
well as the level-0 search. In Phase 2A, two of 16,000 fixed-ef queries had an
exact all-level oracle Recall@10 that exceeded exact native Recall@10 by 0.1.

The two rows are preserved unchanged:

- replicate 0, mean shift alpha=1, query 342;
- replicate 1, radial shift sigma=2, query 447.

They are not tie cases. A ground-truth node evaluated during upper-level
greedy descent was available to an oracle over the all-level union but was not
retained by the native level-0 result heap. Thus the old set was too broad for
the intended candidate-discovery oracle.

## Corrected definitions

Starting with Phase 2B:

- `V_L0` is the unique candidate set evaluated by FAISS level-0 search,
  including the selected level-0 seed. The seed's stored upper-navigation
  distance is re-evaluated once by the recording distance computer solely to
  assign it to the level-0 candidate set. That repeated value is asserted to
  be bit-identical and does not affect traversal or output.
- `V_upper_only` is the set of IDs evaluated during entry/upper-level greedy
  navigation that are absent from `V_L0`.
- The discovery oracle is exact FP32 reranking of `V_L0`, not the all-level
  union.

The legacy Phase 1 and Phase 2A raw fields retain their original meaning. They
were not renamed or overwritten. Reports referring to those results must call
them legacy/all-level metrics when contrasting them with Phase 2B.

## Implementation

The research-only adapter is implemented in
`src/instrumentation/faiss_level0_recorder.cpp`. It does not patch or fork the
pinned FAISS submodule and does not copy the HNSW search algorithm.

For each query it:

1. calls FAISS `hnsw_detail::greedy_update_nearest` for levels `max_level`
   through 1, using the selected exact or PQ distance computer;
2. records the resulting level-0 entry point and distance;
3. re-evaluates that seed through the level-0 recorder and asserts identical
   distance;
4. calls FAISS `IndexHNSW::search_level_0` with the native parameters and
   selected entry point;
5. compares IDs and distances against an unmodified full native
   `IndexHNSW::search` call;
6. checks the graph fingerprint before and after; and
7. computes `V_upper_only = V_upper_phase - V_L0`.

`src/instrumentation/paired_decomposition_l0.cpp` computes L0 coverage,
oracles, deltas, and set overlap. The existing
`paired_decomposition.cpp` remains the explicit legacy/all-level path needed
to validate old raw results.

## Validation

`decomposition_correctness` checks on a small deterministic graph that:

- upper-only and L0 sets are disjoint and upper-only evaluations occur;
- exact and PQ native IDs and distances are unchanged;
- the graph fingerprint is unchanged;
- L0 oracle recall equals L0 coverage; and
- exact L0 oracle recall equals exact native recall.

`candidate_semantics_regression` regenerates the Phase 2A indexes and queries
for both known anomalies. Both reproduce `delta_exact_control=0.1` under the
legacy all-level definition and become exactly zero under the L0 definition.
It also reruns complete 500-query deterministic IID and mean-shift alpha=1
subsets from replicate 0:

| Subset | Mean L0 discovery delta | L0 rerank recovery |
| --- | ---: | ---: |
| IID | 0.0042 | 0.970629 |
| mean alpha=1 | approximately 0 | 1.000000 |

The correction therefore does not materially change the Phase 1 or Phase 2A
aggregate conclusion: discovery degradation remains near zero and exact
reranking recovers at least 97% of the gap in these deterministic controls.
Phase 2B was allowed to proceed.
