# Phase 3B raw data

`preregistration.md` preserves the protocol before primary score analysis.
`primary_analysis/` preserves the completed ef64 checkpoint before ef128.

For each `ef64/` and `ef128/`:

- `manifest.json`: repository/FAISS revisions, executable and source hashes,
  immutable graph/PQ hashes and machine information.
- `resolved_config.conf` and `phase3a_config.conf`: all experiment, dataset,
  construction, seed and analysis settings.
- `queries.jsonl`: ascending query IDs, native IDs/scores, provided GT, binary
  chunk name, byte offset and number of candidate records.
- `candidate_scores_N.bin`: packed little-endian records, 12 bytes each:
  int32 database ID, float32 exact squared L2, float32 PQ ADC. Records are in
  unique first L0 evaluation order. Errors are derived as float64(PQ)-float64(exact).
  The byte offset in queries.jsonl is relative to its named file.

`analysis/ef*/queries.jsonl` contains all query metrics and the three top-k
lists. `harmful_pairs.jsonl` contains all displaced/intruder cross products
for harmful queries. `random_pair_indices.bin` contains little-endian int32
candidate-position pairs, 512 pairs per query, query-major; query offsets
are recorded in the derived JSONL. Scores for a sampled pair are recovered
from the corresponding candidate chunk. Native tie audit cases are retained
in `native_tie_cases.json` and aggregate validation in `native_heap_audit.json`.

`analysis/failed_ef64_numpy_bool/` is an excluded partial diagnostic artifact;
do not include it when reading completed ef64/ef128 analyses.

Regenerate with `scripts/analyze_phase3b_fixed_candidate_ranking.py derive`
into a **new** output directory, then `summarize`. Run
`scripts/audit_phase3b_native_ties.py` for the independent result-heap audit.
These commands need the raw files and Phase 3A query logs, not the dataset or
FAISS. See docs/experiment_log.md for exact commands.

`final_verification.json` is produced by `scripts/verify_phase3b_artifacts.py`;
it records raw-input integrity, byte-identical re-derivation, final artifact
hashes, and exact Python package versions. The original per-stage analysis
source hashes precede additions to the aggregate tables/plots; final source
hashes are recorded separately without overwriting stage provenance.
