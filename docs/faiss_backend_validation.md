# FAISS backend validation

## Scope

This is a synthetic proof-of-concept, not a benchmark and not evidence that
quantization induces query hardness. It validates only that Meta FAISS can run
exact FP32 L2 and PQ asymmetric-distance traversal over one shared HNSW
topology.

No trajectory logging, SIFT1M run, latency comparison, or performance
optimization was performed.

## Dependency

- Repository: Meta FAISS CPU
- Tag: `v1.15.0`
- Pinned commit: `20f14b31a6d54e243a3d1de6ae193fc4c3ec18ed`
- Integration: Git submodule at `third_party/faiss`
- Upstream modifications: none

The top-level CMake configuration fails if the initialized FAISS submodule does
not resolve to the pinned full commit.

## Design validated

One `faiss::IndexHNSW` was constructed with `faiss::IndexFlatL2` storage. Its
only call to `add()` therefore constructed all levels and links using exact
FP32 squared-L2 distances. A separate `faiss::IndexPQ` was trained and populated
with the same vectors in the same ID order; it owns PQ codes but no HNSW graph.

Both modes invoke the same `faiss::IndexHNSW::search()` and
`faiss::HNSW::search()` implementation. A single-threaded RAII guard changes
only `IndexHNSW::storage` for the duration of a search:

- exact traversal obtains `IndexFlatL2`'s FP32 distance computer;
- PQ traversal obtains `IndexPQ`'s existing asymmetric-distance computer.

The guard requires equal dimension, L2 metric, and `ntotal`, and restores the
FP32 storage even if search throws. No `IndexHNSWPQ` graph is constructed.

## Graph identity check

The versioned `faiss-hnsw-struct-v1` fingerprint is SHA-256 over a canonical
encoding of:

- entry point;
- maximum level;
- complete per-vector levels;
- complete neighbor offsets;
- complete neighbor array, including unused `-1` slots;
- complete cumulative neighbor-capacity vector.

The test checks the fingerprint after FP32 construction, after PQ training,
before and after exact traversal, while PQ storage is selected, after PQ
traversal, and after FP32 storage restoration. It also deliberately changes and
restores each included structural field to confirm that every field affects the
fingerprint. As an identity control, the test also searches through a second
`IndexFlatL2` populated with the same vectors and ID order; its ordered IDs and
distances must exactly match both native search and guarded FP32 search.

## Exact command and configuration

Base superproject commit at execution time:
`aa860b6ec42e8695daccdabb9114eaa6fc130fec`.
The worktree was dirty with the implementation and documentation described in
this file.

```bash
cmake -S . -B /tmp/quant-hardness-build-v4 -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/quant-hardness-build-v4 --target faiss_backend_validation -j 8
ctest --test-dir /tmp/quant-hardness-build-v4 --output-on-failure
/tmp/quant-hardness-build-v4/faiss_backend_validation configs/indexes/faiss_hnsw_poc.conf
```

Resolved configuration:

```text
seed=12345
base_vectors=4096
queries=32
dimension=32
k=10
hnsw_m=16
ef_construction=40
ef_search=32
pq_m=8
pq_nbits=4
metric=squared L2
threads=1
base/query distribution=independent standard normal FP32 vectors
PQ training data=all 4096 synthetic base vectors
```

Machine and build information:

```text
host=rwcpu8.cse.ust.hk
kernel=Linux 5.14.0-687.24.1.el9_8.x86_64
cpu=Intel Core i9-10920X, 12 cores / 24 hardware threads
memory_bytes=33047748608
compiler=GCC 11.5.0 20240719 (Red Hat 11.5.0-14)
cmake=4.0.3
build_type=Release
faiss_opt_level=generic
```

## Observed results

CTest result:

```text
1/1 Test #1: faiss_backend_validation ......... Passed
100% tests passed, 0 tests failed out of 1
```

Program output:

```text
faiss_commit=20f14b31a6d54e243a3d1de6ae193fc4c3ec18ed
graph_fingerprint=f3d781d785ae54b4291b73b026e446ccd455744e58922b5878e8f455201ad270
base_vectors=4096
queries=32
dimension=32
pq_distance_differences=4096/4096
pq_mean_absolute_distance_difference=11.5816
identity_storage_control=PASS
exact_first_result_id=3551
pq_first_result_id=4092
status=PASS
```

Observations only:

- The FAISS checkout matched the pinned commit.
- One FP32-built HNSW graph was used for both searches.
- The graph fingerprint remained
  `f3d781d785ae54b4291b73b026e446ccd455744e58922b5878e8f455201ad270`
  across both traversal modes and storage restoration.
- For the first synthetic query, all 4096 tested PQ ADC distances differed
  from their FP32 squared-L2 counterparts by more than `1e-5`.
- The mean absolute distance difference for that query was `11.5816` in the
  dataset's squared-L2 units.
- Both searches returned 10 valid in-range IDs for each of 32 queries.
- Native FP32 search, guarded FP32 search, and the alternate-`IndexFlatL2`
  identity control produced exactly identical ordered IDs and distances.
- The first returned ID was `3551` for exact traversal and `4092` for PQ
  traversal.

## Anomalies and possible confounders

- The first exact and PQ result IDs differed. This is recorded as an
  observation; the POC does not establish why they differed.
- The vectors are synthetic standard-normal data and are not representative of
  SIFT1M or any target distribution.
- PQ was trained on the same synthetic base set and tested at one seed with
  4-bit subquantizers. No training-set or seed sensitivity was tested.
- Results are traversal-distance ranked. Exact reranking and recall against
  exhaustive ground truth were not part of this backend validation.
- The storage switch is intentionally single-threaded and is unsafe for
  concurrent searches on the same `IndexHNSW` object.
- The mean absolute distance difference is scale-dependent and is not a quality
  or hardness metric.
- No per-query latency or traversal counters were collected.

## Smallest distinguishing next experiment

Before SIFT1M, add an exhaustive FP32 ground-truth calculation for this same
synthetic dataset and preserve paired per-query exact/PQ top-10 IDs and
recall@10. The completed identity-distance control rules out the storage-switch
mechanism as an explanation when the alternate storage contains identical FP32
vectors and IDs; the next test should determine the paired distribution of
result-quality changes without adding trajectory instrumentation.
