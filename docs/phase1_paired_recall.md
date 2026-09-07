# Phase 1 synthetic paired-recall experiment

## Scope and question

This experiment asks whether per-query Recall@10 differs when a single HNSW
graph constructed with FP32 squared-L2 distances is searched with either FP32
or FAISS PQ asymmetric distances. It is a synthetic phenomenon check, not a
benchmark and not evidence of novelty or causality.

No trajectory logging, optimization, new quantizer, or SIFT1M data was used.

## Reproducible setup

- Database: 20,000 independent standard-normal FP32 vectors, dimension 64,
  seed 1729.
- Queries: 500 separately generated independent standard-normal FP32 vectors,
  dimension 64, seed 2718. Query IDs and execution order are 0 through 499 in
  both modes.
- Ground truth: exhaustive FP32 squared-L2 top-10 using `IndexFlatL2`.
- Graph: one FP32-built HNSW, `M=16`, `efConstruction=80`, construction seed
  31415.
- Search: `k=10`, `efSearch=16`, bounded queue and relative-distance check
  enabled, one thread.
- PQ: FAISS `IndexPQ`, eight 4-bit subquantizers, trained on all database
  vectors with seed 16180; FP32 queries use asymmetric distance.
- FAISS: `v1.15.0`, commit
  `20f14b31a6d54e243a3d1de6ae193fc4c3ec18ed`.
- Superproject base commit:
  `aa860b6ec42e8695daccdabb9114eaa6fc130fec`, with the recorded implementation
  present as a dirty worktree.

The complete resolved configuration is stored with the raw run at
`runs/phase1_synthetic_paired_recall_v1/resolved_config.conf`. Standard-normal
generation uses the recorded GCC/libstdc++ implementation; C++ does not
guarantee that `std::normal_distribution` is bit-identical across different
standard-library implementations.

## Fair-comparison checks

The HNSW graph was built once while the only attached storage was
`IndexFlatL2`. A separate `IndexPQ` contains codes for the same vectors in the
same ID order and has no HNSW graph. Exact and PQ modes call the same FAISS
HNSW search implementation and differ only in the attached query-time distance
computer.

The `faiss-hnsw-struct-v1` fingerprint was
`5eeca90445dd29413ba0ba7bceb4be18258ef5580174050fbaf087e2f57791e7`.
Runtime assertions verified this value after graph construction, after PQ
preparation, before and after exact search, and after PQ search and storage
restoration. Query-order equality is also asserted before either run.

Correctness tests cover a hand-checkable exhaustive ground truth, perfect,
partial, and disjoint Recall@10, positive and negative delta sign, accepted and
rejected paired query orders, and stable/sensitivity checks for the graph
fingerprint. Both test executables passed.

## Outputs

- Raw paired rows: `runs/phase1_synthetic_paired_recall_v1/queries.jsonl`
- Run metadata: `runs/phase1_synthetic_paired_recall_v1/manifest.json`
- Summary table:
  `results/tables/phase1_synthetic_paired_recall_v1_summary.json`
- Sorted per-query table:
  `results/tables/phase1_synthetic_paired_recall_v1_delta_sorted.csv`
- Figures: histogram, empirical CDF, sorted delta, and exact-recall/delta
  scatter under `results/figures/`, all prefixed
  `phase1_synthetic_paired_recall_v1_`.

The analysis script independently checks sequential paired query order,
top-10 list size and uniqueness, and the recorded delta sign/value before
producing derived artifacts. Existing raw or derived outputs are never
overwritten.

## Observed results

Recall is measured against exhaustive FP32 ground truth, and
`delta_recall_at_10 = recall_exact_at_10 - recall_pq_at_10`.

| Measurement | Observed value |
| --- | ---: |
| Exact mean Recall@10 | 0.4772 |
| Exact median Recall@10 | 0.5 |
| PQ mean Recall@10 | 0.0708 |
| PQ median Recall@10 | 0.0 |
| Mean delta | 0.4064 |
| Median delta | 0.4 |
| Minimum / maximum delta | -0.2 / 0.9 |
| Delta p90 / p95 / p99 | 0.6 / 0.7 / 0.8 |
| Fraction delta = 0 | 0.026 (13/500) |
| Fraction delta > 0 | 0.966 (483/500) |
| Fraction delta < 0 | 0.008 (4/500) |

The modal delta was 0.4 (100 queries), followed by 0.5 (99 queries). Among
the 483 positive-delta queries, 168 (34.78%) accounted for at least 50% of
total positive recall loss, 309 (63.98%) accounted for at least 80%, and 374
(77.43%) accounted for at least 90%.

Thus nonzero per-query degradation exists in this run. The positive loss is
not confined to a small set of queries: 96.6% of queries have positive delta,
and more than three quarters of the positive-delta queries are needed to
account for 90% of total positive loss. This run therefore does not show loss
that is restricted to a rare heavy tail. That observation is specific to this
single synthetic distribution, graph/search setting, seed set, and coarse PQ
configuration.

Four queries had negative delta, including a minimum of -0.2, meaning PQ
returned more ground-truth neighbors than exact HNSW for those queries. This is
possible because both modes are approximate graph searches; no causal
explanation was tested.

## Anomalies, limitations, and unresolved confounders

- PQ top-10 ranking is its native approximate-distance ranking. The observed
  difference combines PQ-induced traversal decisions with approximate ranking
  of retained candidates. This experiment cannot attribute loss specifically
  to traversal divergence.
- The 4-bit PQ baseline has very low median recall in this configuration. One
  PQ setting and one set of seeds do not establish behavior for other
  quantization strengths or distributions.
- Exact HNSW itself has mean Recall@10 0.4772 because `efSearch=16` is
  deliberately approximate. It is not used as ground truth.
- Four negative deltas and thirteen zero deltas are preserved in the raw rows;
  they were not filtered from the descriptive statistics.
- The storage-switch adapter is single-threaded. Concurrent use would be
  unsafe, but this run explicitly used one thread.
- The graph fingerprint covers HNSW structural search state, not vector storage
  contents. ID alignment and equal storage sizes are checked separately.
- Synthetic normal-vector generation may not be bit-identical with another C++
  standard-library implementation even with the same seeds.

## Smallest distinguishing next experiment

For the same frozen graph, queries, PQ codes, and search settings, expose the
retained candidate set from each PQ search and recompute its ordering with
exact FP32 distances. Record both native PQ-ranked Recall@10 and exact-reranked
Recall@10 from that unchanged candidate set. The gap removed by exact reranking
is final-ranking damage; any remaining gap against the FP32 traversal result is
candidate-discovery/traversal damage. This is a measurement extension, not a
new search algorithm, and does not require trajectory logging.
