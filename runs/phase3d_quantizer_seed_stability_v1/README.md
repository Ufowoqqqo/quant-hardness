# Phase 3D raw data

Pre-registration and resolved config are recorded before training. No old
raw results are overwritten. Source exact pools are content-addressed by
`input_provenance.json`; they remain in the committed Phase 3C run.

- `candidate_requests.tsv`: query ID, Phase 3C candidate chunk filename,
  byte offset in that chunk, candidate count, SHA256 of little-endian int64
  ordered IDs. All five score arrays follow exactly this concatenated order.
- `seed_0` through `seed_4`: complete FAISS `pq64.index` (codebooks and 1M
  codes), ordered `training_ids.i32`, `scores.f32`, config and manifest.
  Scores are little-endian float32 ADC values for 10,353,047 candidates;
  exact scores/GT/ID order are taken without alteration from Phase 3C.
- `analysis/queries.jsonl.gz`: one row per query, exact geometry, ordered/set
  exact top-10, five model records with ordered/set PQ top-10, signed losses,
  inversions, errors and descriptive loss allocation, harmful category,
  across-seed mean/sample-SD/max, pool hash and global score offset.
- `analysis/replacement_pairs.jsonl.gz`: all actual selected displaced/
  intruder Cartesian pairs with model ID, query ID, ranks, exact margin,
  error difference, violation, strict-inversion flag, GT membership.
- `analysis/inversion_pairs.jsonl.gz`: every exact A x B pair strictly
  inverted under >=1 model; query/IDs/ranks/margin, five-bit mask, inversion
  count/frequency/class, signed violations for all five models. Exact ties
  are excluded from strict inversions, not from results or replacements.
- `analysis/pair_summary.json`: pooled pair frequency, pairwise intersection
  and union counts. `validation.json` records source/score/metric controls.
- `primary_checkpoint/`: primary results frozen BEFORE ensemble analysis.
- `ensemble/`: post-primary uniform score-average diagnostic, float64
  accumulation; per-query records plus summary. No new search or benchmark.
- Execution logs retain commands' stdout/stderr. `final_verification.json`
  will record hash validation and independent raw-data re-derivation checks.

Random pair indices are exactly the committed Phase 3C PCG64 samples, reused
across all five models; derivation checks them byte-for-byte. Strict distance
ties use first recorded candidate order. The saved exact oracle is .95315,
not the tie-different native .95319. All five models share this same oracle.

Loss allocations to pair-frequency classes are descriptive equal-credit
bookkeeping, NOT unique causal effects; tie/unassigned debits and GT-intruder
credits are explicit. Pair-frequency categories are one-off=1, minority=2,
majority=3 or 4, persistent=5. Query-category losses are disjoint signed sums.
All SDs use ddof=1; quantile bins preserve tied descriptor values.
