# Phase 3C raw precision experiment

`preregistration.md` is the protocol written before score export.
`primary_checkpoint/` preserves primary tables and observations before the
optional control. All prior runs are untouched.

`exact_pool/` is the primary 10,000-query FP32-traversal pool.
`pq64_pool_control/` is the separately labeled 1,000-query sample of existing
Phase 3B PQ64 candidate sets. `control_candidates.bin` and its provenance JSON
preserve the imported IDs; no quantized traversal generated these again.

Each raw condition contains:

- `candidate_scores_N.bin`: packed 16-byte little-endian records:
  int32 ID, float32 exact squared L2, float32 PQ32 ADC, float32 PQ64 ADC.
  One shared ID column guarantees identical membership under all scorers.
- `queries.jsonl`: query IDs, chunk name, byte offset, candidate count, SHA-256
  of candidate IDs as little-endian int64 in first unique evaluation order,
  GT, and exact-native control IDs/scores (null for the optional pool).
- `resolved_config.conf`, `phase3a_config.conf`, `manifest.json`: configuration,
  dataset checksums/seeds, frozen graph/PQ settings and hashes, code revision,
  machine and executable/source hashes.

`analysis/CONDITION/` contains per-query metrics (`queries.jsonl`), every
selected displaced/intruder pair (`replacement_pairs.jsonl.gz`), every pair
inverted by either scorer (`inversion_transitions.jsonl.gz`), and random
candidate-position pairs (`random_pair_indices.bin`: int32,int32, 512 per
query). Gzip headers have deterministic time/name fields. Candidate score
errors are float64(PQ)-float64(exact), recoverable for every stored record.

Stable/Improved/Persistent/Regressed flags overlap as defined in the request.
`fully_rescued` is separately recorded. Exact/PQ top-k lists use stable score
sorts in candidate order, and membership sets are also retained explicitly.
No native-result substitution is used for score comparisons.

Regenerate analysis into a new directory with
`scripts/analyze_phase3c_precision_transition.py derive`, then `summarize`.
Database/FAISS access is unnecessary for these steps. Final verification
checks independent regeneration and raw input hashes. See the experiment log
for exact commands. The `*_oracle1_exploratory.csv` tables are a post-hoc
geometry sensitivity check, not a changed primary definition.
