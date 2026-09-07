# Phase 1 plan: paired HNSW traversal baseline

## Status as of 2026-09-07

The synthetic shared-topology proof of concept and the first paired Recall@10
experiment are complete. The operating-regime calibration is also complete;
it selected PQ32×8 at `efSearch` 160, 256, and 384 without using concentration
as a selection criterion. Raw per-query results and observations are documented
in `docs/phase1_paired_recall.md` and `docs/phase1_calibration.md`. Trajectory,
visited-node, distance-count, and latency instrumentation remain intentionally
deferred. The next minimal measurement is exact reranking of the unchanged
retained PQ candidate set to separate candidate discovery from final
approximate ranking.

## Selected backend and inspected revision

Phase 1 uses Meta FAISS CPU at tag `v1.15.0`, pinned to full commit:

```text
20f14b31a6d54e243a3d1de6ae193fc4c3ec18ed
```

The tag and commit were inspected before implementation. The dependency is
recorded as a Git submodule, and the superproject must pin the submodule gitlink
to this commit rather than a branch.

The following experiment inputs still need to be specified before a real
baseline is executable; the synthetic proof-of-concept fixes small local values
explicitly in its test configuration:

- dataset, vector dimension, distance metric, and preprocessing;
- query set and exact FP32 ground-truth top-10 IDs;
- HNSW construction parameters, construction seed, and persisted graph;
- `efSearch`, result count (`k = 10`), entry-point policy, and tie-breaking;
- quantizer type, training data, parameters, seed, and whether queries are
  quantized;
- final-result policy: traversal-distance ranking or exact FP32 reranking.

The final-result policy is scientifically material. A later baseline should
log both the traversal-ranked top-10 and an exact-FP32-reranked top-10
from the retained candidate set. The former measures the end-to-end mode; the
latter helps separate candidate-discovery damage from final ranking error.
Neither should silently replace the other as the primary metric.

## Concrete FAISS source map

The relevant code at the pinned revision is:

- `faiss/IndexHNSW.h`
  - `IndexHNSW` owns the `HNSW hnsw` topology and an independent `Index*
    storage`.
  - `IndexHNSWFlat` constructs that storage as `IndexFlatL2` for L2.
  - `IndexHNSWPQ` constructs an independent HNSW over `IndexPQ`; it is not used
    for the paired experiment because its graph would be built independently.
- `faiss/IndexHNSW.cpp`
  - `IndexHNSW::add()` calls `storage->add()` and then
    `hnsw_add_vertices()`; during construction, `hnsw_add_vertices()` obtains a
    distance computer from the FP32 storage. This is the only graph-construction
    path used by Phase 1.
  - `storage_distance_computer()` delegates to
    `storage->get_distance_computer()`.
  - the internal `hnsw_search()` obtains that distance computer, calls
    `set_query()`, and invokes the shared `HNSW::search()`.
  - `IndexHNSW::search()` selects L2 ordering and uses the same internal search
    path regardless of storage representation.
- `faiss/impl/HNSW.h` and `faiss/impl/HNSW.cpp`
  - `HNSW::search()` accepts a `DistanceComputer&` and performs upper-layer
    greedy descent plus base-layer search.
  - `search_impl()`, `greedy_update_nearest_impl()`, and
    `search_from_candidates_dispatch()` consume that same distance-computer
    abstraction without owning vector storage.
  - `HNSW` stores `entry_point`, `max_level`, `levels`, `offsets`,
    `neighbors`, and `cum_nneighbor_per_level`, which define the search
    topology.
- `faiss/impl/DistanceComputer.h`
  - `DistanceComputer::set_query()`, `operator()`, and
    `distances_batch_4()` are the query-to-node distance seam already used by
    HNSW traversal.
- `faiss/IndexFlat.cpp`
  - `FlatL2Dis`, returned by the flat storage distance-computer factory,
    computes exact squared FP32 L2 distances.
- `faiss/IndexPQ.cpp` and
  `faiss/impl/pq_code_distance/PQDistanceComputer_impl.h`
  - `IndexPQ::get_FlatCodesDistanceComputer()` returns FAISS's existing PQ
    distance computer.
  - `PQDistanceComputer::set_query()` builds the query lookup table and
    `distance_to_code()`/`distance_to_code_batch_4()` perform asymmetric L2
    distance computation against stored PQ codes.

## Concrete shared-topology design

The proof-of-concept uses three objects:

1. one `IndexFlatL2` storage holding the FP32 base vectors;
2. one `IndexHNSW` owning the sole HNSW topology and initially pointing to the
   flat storage;
3. one trained `IndexPQ` holding PQ codes for the same base vectors in exactly
   the same insertion/ID order, but no HNSW topology.

`IndexHNSW::add()` is called exactly once while its storage is the
`IndexFlatL2`. Exact search uses the normal `IndexHNSW::search()` path. For PQ
search, a repository-owned, single-threaded RAII guard temporarily changes only
the public `IndexHNSW::storage` pointer to the ID-aligned `IndexPQ`, invokes the
same `IndexHNSW::search()` function, and restores the original pointer even if
search throws. Before switching, it requires equal dimension, metric, and
`ntotal` and a frozen graph fingerprint. It does not call `add()`, `train()`,
`reset()`, link-reordering, or any HNSW mutator.

This is preferable to constructing `IndexHNSWPQ`, which would violate graph
identity. It also avoids copying the HNSW algorithm or patching upstream FAISS.
The guard is deliberately research-only and not thread-safe; concurrent search
on the guarded `IndexHNSW` is forbidden. If later experiments require
concurrency, the smallest upstream patch should instead add a const search entry
point accepting an explicit traversal `Index` or `DistanceComputer` factory.

The only intended query-time difference is therefore:

```text
IndexFlatL2::get_distance_computer()  versus
IndexPQ::get_distance_computer()
```

Both feed the identical `IndexHNSW::search()` and `HNSW::search()` code path.

## Graph fingerprint design

Define a versioned fingerprint over a canonical byte encoding named
`faiss-hnsw-struct-v1`. Encode field names, lengths, and integer values in a
fixed byte order, then hash the resulting bytes. Include:

- `entry_point` and `max_level`;
- the complete `levels` vector;
- the complete `offsets` vector;
- the complete `neighbors` vector, including unused `-1` slots;
- the complete `cum_nneighbor_per_level` vector.

These are the structural fields used to locate every node's neighbor range at
every HNSW level. Construction RNG state and `assign_probas` are not part of the
query topology; query parameters such as `efSearch`, bounded-queue mode, and
relative-distance checks are recorded and compared separately.

Compute the fingerprint immediately after FP32 construction, before and after
each exact search, while PQ storage is selected, after each PQ search, and after
the FP32 storage is restored. Every value must be byte-for-byte identical. The
POC also checks that a deliberate mutation of a copied structural field changes
the fingerprint, guarding against a constant or incomplete implementation.

## Experimental invariant

Build one HNSW graph from the FP32 base vectors, persist it, load it once, and
run two modes through the same search implementation:

- **A — `exact_fp32`:** FP32 query-to-node distance during traversal.
- **B — `quantized`:** the selected approximate distance during traversal.

Only the distance evaluator may differ. Both modes must share the same in-memory
graph topology, graph levels, entry point, neighbor iteration order, query
vectors and ordering, `k`, `efSearch`, filtering, termination condition,
candidate-queue behavior, visited-set implementation, and deterministic
tie-breaking. Graph construction is outside the paired measurement and must
never run between modes.

Recall@10 is computed against independent exact FP32 brute-force ground truth,
not against mode A. Mode A is still an approximate graph search and therefore
is not ground truth.

## Minimal experiment architecture

Keep the experiment runner and measurement layer outside `third_party/`:

```text
fixed FP32 graph + queries + ground truth + resolved config
                         |
                  paired run controller
                         |
             one shared HNSW search loop
                  /              \
       FP32 distance oracle   quantized distance oracle
                  \              /
            read-only search observer
                         |
               paired per-query JSONL
```

The minimum repository-owned components are:

1. **HNSW adapter (`src/graph/`)** — loads the pinned graph once, exposes its
   immutable metadata/fingerprint, and calls the upstream search loop. It must
   not contain measurement definitions.
2. **Distance oracles (`src/quantization/`)** — a common interface with an FP32
   reference implementation and one quantized implementation. Oracle calls are
   the unit counted as distance evaluations. Quantizer training/building occurs
   before the paired search run.
3. **Search observer (`src/instrumentation/`)** — receives events or reads
   counters for node discovery, distance evaluation, and query timing. It must
   not alter queues, branching, termination, or neighbor ordering.
4. **Metrics (`src/metrics/`)** — computes recall from result IDs and stored
   ground-truth IDs after search. It must not be called from the ANN traversal.
5. **Paired runner (`scripts/run_search.py`)** — validates invariants, applies
   warm-up policy, executes the same ordered query list in both modes, and
   writes a run manifest plus one paired record per query.

Use one search loop parameterized by a distance-oracle interface rather than
two copied implementations. If the selected library cannot support this at its
public API, make the smallest upstream patch possible, keep it in
`third_party/`, record the patch and revision, and validate it against the
unmodified implementation.

## Smallest instrumentation points

Exact locations depend on the selected HNSW implementation. Instrument only
these boundaries:

1. **Distance-call boundary:** increment `distance_evaluations` exactly once
   whenever traversal requests a query-to-node distance, including the entry
   point and upper-layer descent. State explicitly whether any final reranking
   evaluations are reported separately; do not mix them into traversal counts.
2. **Visited-set insertion boundary:** increment `visited_nodes` only on the
   first successful insertion of a node ID into the per-query visited set.
   Also consider logging `expanded_nodes` separately, because “visited” and
   “popped/expanded” are not interchangeable.
3. **Search-call boundary:** start the timer immediately before entering the
   common search function and stop it immediately after returned IDs/distances
   are materialized. Quantizer training, graph loading, logging, recall
   computation, and exact reranking are excluded and timed separately.
4. **Return boundary:** copy the traversal-ranked result IDs, retained candidate
   IDs if available, and exact-reranked result IDs without modifying the search
   state.

Prefer an optional observer/counter object whose disabled form is a no-op. Do
not add logging or file I/O inside the ANN search loop.

## Proposed raw output layout

```text
runs/<run_id>/
├── manifest.json
├── resolved_config.yaml
├── queries.jsonl
└── command.txt
```

`manifest.json` records run-level provenance:

```json
{
  "schema_version": 1,
  "run_id": "<stable unique ID>",
  "created_at_utc": "<ISO-8601>",
  "git_commit": "<full SHA>",
  "dirty_worktree": false,
  "exact_implementation": "<name and revision>",
  "third_party_patch_sha256": "<SHA-256 or null>",
  "dataset": {
    "name": "<name>",
    "base_vectors_sha256": "<SHA-256>",
    "queries_sha256": "<SHA-256>",
    "ground_truth_sha256": "<SHA-256>",
    "dimension": 0,
    "distance_metric": "l2|inner_product|cosine",
    "preprocessing": "<explicit description>"
  },
  "random_seeds": {
    "graph_construction": 0,
    "quantizer_training": 0,
    "experiment": 0
  },
  "graph": {
    "artifact_sha256": "<SHA-256>",
    "topology_fingerprint": "<SHA-256>",
    "M": 0,
    "ef_construction": 0,
    "entry_point": 0,
    "max_level": 0
  },
  "search": {
    "k": 10,
    "ef_search": 0,
    "query_order_sha256": "<SHA-256>",
    "tie_breaking": "<rule>",
    "final_result_policy": "<explicit policy>",
    "warmup_policy": "<explicit policy>",
    "repetitions": 0
  },
  "quantizer": {
    "type": "<type>",
    "parameters": {},
    "training_data_sha256": "<SHA-256>",
    "query_representation": "<FP32 or quantized>"
  },
  "machine": {
    "hostname": "<host>",
    "os": "<OS and kernel>",
    "cpu": "<model and core count>",
    "memory_bytes": 0,
    "gpu": "<model or null>",
    "compiler": "<compiler and flags>"
  }
}
```

`queries.jsonl` contains one paired record per query. Arrays preserve ordered
IDs; counts and latency remain raw per repetition rather than being averaged:

```json
{
  "schema_version": 1,
  "run_id": "<run ID>",
  "query_index": 0,
  "query_id": "<dataset query ID>",
  "query_order": 0,
  "ground_truth_ids": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
  "repetition": 0,
  "exact_fp32": {
    "recall_at_10": 0.0,
    "visited_nodes": 0,
    "expanded_nodes": 0,
    "distance_evaluations": 0,
    "rerank_distance_evaluations": 0,
    "latency_ns": 0,
    "traversal_result_ids": [],
    "exact_reranked_result_ids": [],
    "retained_candidate_ids": []
  },
  "quantized": {
    "recall_at_10": 0.0,
    "visited_nodes": 0,
    "expanded_nodes": 0,
    "distance_evaluations": 0,
    "rerank_distance_evaluations": 0,
    "latency_ns": 0,
    "traversal_result_ids": [],
    "exact_reranked_result_ids": [],
    "retained_candidate_ids": []
  }
}
```

Here, `exact_fp32.traversal_result_ids` are the exact-search result IDs and
`quantized.traversal_result_ids` are the quantized-search result IDs. A generic
“search result IDs” field is intentionally not duplicated; the selected primary
result policy must be declared in the manifest. If retained candidates are not
available from the chosen implementation, record `null`, not an empty list.

## Fairness and correctness unit tests

### Reference metric tests

1. **Recall@10 reference cases:** compare the production recall function with a
   simple set-intersection reference for perfect, disjoint, partial, duplicated,
   and fewer-than-10 results. Specify how duplicate IDs and short lists are
   rejected or handled.
2. **FP32 distance reference:** compare the exact distance oracle against a
   scalar implementation on fixed vectors, random seeded vectors, zeros, and
   large/small finite values using a declared tolerance.
3. **Quantized distance reference:** compare the optimized quantized oracle
   against a simple decoder/scalar reference on fixed codes and seeded random
   inputs before any benchmark.

### Shared-traversal tests

4. **Identity-oracle equivalence:** substitute a quantized-oracle test double
   that returns the exact FP32 distance. Modes A and B must produce identical
   ordered result IDs, visited-node counts, expanded-node counts, distance-call
   counts, and traversal event traces. Latency is explicitly exempt.
5. **Single-loop proof:** assert both mode entry points invoke the same search
   function/binary code path and differ only in the oracle instance/config.
6. **Parameter snapshot equality:** serialize all effective search parameters
   immediately before each mode and require byte-for-byte equality except for
   the allowed distance-oracle descriptor.
7. **Graph immutability:** hash serialized topology, levels, entry point, and
   neighbor lists before A, between A/B, and after B; hashes must match. Also
   assert both runs reference the same loaded graph object where the language
   permits.
8. **Query-order equality:** hash the ordered query-ID sequence for each mode
   and require equality. Assert each paired record contains the same query ID
   and vector checksum.
9. **Deterministic tie test:** use a tiny graph with equal distances and verify
   identical neighbor iteration and tie-breaking across modes when their
   oracle outputs are equal.
10. **Counter semantics:** on a hand-constructed tiny graph, compare observer
    counts with a scalar traced reference, including entry-point evaluation,
    duplicate neighbor encounters, upper layers, and termination.
11. **Instrumentation transparency:** with the FP32 oracle, compare search with
    observation enabled and disabled. Ordered results and traversal trace must
    match.
12. **Log round-trip:** write and reload paired records without loss of ID
    order, integer precision for nanosecond timings, or distinction between
    unavailable (`null`) and empty (`[]`) fields.

Use small deterministic graphs for unit tests. End-to-end tests on a real
dataset supplement but do not replace these tests.

## Possible confounders and controls

- **Different graphs or implicit rebuilds:** load one FP32-built artifact once;
  record topology and artifact fingerprints.
- **Different effective search parameters:** snapshot resolved parameters and
  fail the run on any unapproved difference.
- **Ground truth confused with exact HNSW:** compute ground truth independently
  by exhaustive FP32 search.
- **Final-ranking error mixed with traversal damage:** preserve traversal-ranked
  and exact-reranked outputs and declare the primary policy.
- **Quantizer training leakage:** record training set identity and keep query and
  evaluation data out of training unless explicitly studying transductive use.
- **Metric/preprocessing mismatch:** record normalization, dtype conversion,
  dimensional transforms, and distance convention; share them across modes.
- **Tie-breaking and floating-point behavior:** use deterministic ID-based tie
  rules and record compiler/SIMD settings, tolerances, and NaN policy.
- **Ambiguous counters:** define unique visited nodes, expanded nodes, traversal
  distance calls, and rerank distance calls separately.
- **Caching, warm-up, CPU frequency, and thermal drift:** use a declared warm-up,
  raw repetitions, fixed thread affinity where possible, and record execution
  order. Do not interpret a single latency pass as traversal hardness.
- **Mode-order effects:** preserve query ordering within both modes and use
  separate, predeclared AB and BA repetitions (or paired blocks) to detect
  systematic order effects.
- **Logging overhead:** keep file I/O outside the timed region and test that
  instrumentation does not change traversal.
- **Threading and nondeterminism:** begin single-threaded; record thread counts
  and deterministic seeds before studying parallel search.
- **Result truncation:** retain ordered top-10 IDs and, where available, the full
  `efSearch` candidate set needed for exact reranking.
- **ID mapping errors:** test the mapping among graph-internal, dataset, and
  ground-truth IDs.
- **Latency implementation-cost confounding:** quantized arithmetic may be
  slower or faster independently of graph hardness. Interpret visited nodes and
  distance evaluations separately from latency.
- **Multiple simultaneous changes:** do not change topology, graph construction,
  query representation, quantizer, and search policy in one comparison.

## Execution sequence

1. Pin the HNSW dependency and inspect its actual search loop. **Completed for
   FAISS v1.15.0 at the commit recorded above.**
2. Validate exact and PQ distance computers on a shared synthetic topology,
   including the graph fingerprint invariant. **Completed by the initial POC;
   see `docs/faiss_backend_validation.md`.**
3. Freeze metric, preprocessing, graph, query order, counter definitions, and
   final-result policy in a versioned configuration.
4. Implement and test scalar FP32 distance and recall@10 references.
5. Add the common distance-oracle seam and no-op observer; prove FP32 behavior
   matches the unmodified upstream implementation.
6. Validate FAISS's PQ distance computer against a simple decode/reference path
   before any optimization or performance claim.
7. Run deterministic tiny-graph fairness tests. The initial POC includes an
   identity-distance control through an ID-aligned alternate FP32 storage;
   additional hand-constructed counter tests remain pending.
8. Run one small paired smoke experiment, preserve all per-query records, and
   append its exact command/configuration to `docs/experiment_log.md`.

The smallest next experiment is the synthetic paired recall@10 test described
in `docs/faiss_backend_validation.md`. It must precede SIFT1M and trajectory
instrumentation.
