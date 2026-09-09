# Phase 5A pre-run dataset amendment — 2026-09-09

This amendment is made **before any Phase 5A ANN retrieval result**. The reason
is data availability, not ANN outcomes. The earlier DBPedia ada-002 artifact
requires a long download; a local cache of a modern embedding version was
found. Do not describe this as the same artifact or only a renamed split.

## Source and identity

Repository: `Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M`.
Cached source revision: `4a9731217921bc476a0f03544f11f22ae4903fa5`, as recorded
in the local dataset metadata's source URLs. There are26 numbered Arrow
shards; source column `text-embedding-3-large-1536-embedding` stores float64
lists. The cache describes1000000 entities. Complete local validation scans
every row for1536 components, nulls, nonfinite values and zero norms, records
all shard byte sizes/SHA-256 and row ranges. These are local integrity
fingerprints, not independently publisher-authenticated hashes.

Machine-readable source/split evidence:
`runs/phase5a_highdim_external_validity_v1/amendment/dataset_manifest.json`.
Original source row identity is the concatenation of numerically ordered
shards, preserving record-batch and row order. Original text/entity IDs are
not used to reorder vectors. This source has no official990k/10k role split.

Validation completed: all26 shards decoded, exactly1000000 rows of dimension
1536; no null, nonfinite or zero vectors. Maximum normalized FP32 norm error
was `2.995396664040584e-08`. Source files were not changed.

## Frozen roles

NumPy1.26.4 `Generator(PCG64(20260909)).permutation(1000000)` defines order:

- First990000 row IDs: base database, in permutation order. HNSW/PQ database
  ID is the position in this saved base list.
- Last10000 row IDs: evaluation queries, in permutation order.
- PQ training: `Generator(PCG64(95000011)).choice(990000,65536,replace=False)`
  gives positions within the base list; save corresponding source IDs in
  that sampling order. This is a subset of base, not a held-out training pool.

Save each ordered list as little-endian int64 and record its SHA-256. No
evaluation-query row can occur in base or PQ training. Row-disjointness does
not imply semantically unrelated entities or unique embedding contents;
do not claim that stronger guarantee. This is explicitly a **held-out
entity-to-entity nearest-neighbor workload**, not natural-language-query
retrieval. Ground truth must be newly computed against the saved base.

Frozen SHA-256 values:

| Ordered source IDs | Count | SHA-256 |
|---|---:|---|
| Base | 990000 | `0b65c6b76ebcb691fdfe199eb0c642ee556f0878f01f17a1006d6b0d16e36a40` |
| Query | 10000 | `4bc58919e88432cfa96f8450ed52616ffb95585a486977d9edf39ff8bb0f8797` |
| PQ training | 65536 | `c348d26740f319d8061549e51b12b21d9c6fd09701c4a80782fabea24f208204` |

Manifest SHA-256:
`98aa2949f8faf56aea0e3c35da9f73c549a32a1f5b4ba4280e343dff25b36ebd`.
Full-size ID regeneration matched all three saved files exactly.

## Frozen preprocessing

Implementation: `scripts/phase5a_dataset.py::normalize_fp32_once`.
For each source vector, first cast values to FP32. Compute its norm in FP64
from those FP32 values, divide in FP64, then store contiguous little-endian
FP32. Reject nonfinite and zero vectors. This specifies precision and order;
it is not normalization of the original float64 source before casting.

Archive normalized vectors once and use those identical bytes for graph,
PQ training, GT, traversal and exact refinement. Never renormalize separately
in search backends. Use squared L2 consistently, mathematically equivalent
to cosine ranking on unit vectors; retain planned independent finite-precision
and exact-ranking checks before scientific interpretation. The amendment scan
validates normalization but does not persist ANN vectors or run retrieval.

## Unchanged experimental design

Primary PQ768×8, 2 dimensions/subquantizer,768B code versus6144B FP32,8×
compression. Training count65536, seed95000037,25 iterations unchanged.
M16/efConstruction80, fixed FP32 topology, ef32/64/128, k10, boundedL16,
corrected L0 semantics, all hypotheses and benchmark criteria unchanged.
No uncertainty model, selective query policy or retrieval-based tuning.

## Commit gate and commands

Commit this amendment, frozen config/preprocessing, source manifest and role
ID files BEFORE running any Phase 5A retrieval evaluation. Record that commit
as experimental provenance when the subsequent experiment begins.

```bash
/rwproject/kdd-db/kluaq/miniconda3/envs/build_env/bin/python -m unittest discover -s tests -p test_phase5a_dataset.py
/rwproject/kdd-db/kluaq/miniconda3/envs/build_env/bin/python scripts/freeze_phase5a_amendment.py
/rwproject/kdd-db/kluaq/miniconda3/envs/build_env/bin/python scripts/audit_phase5a_amendment.py
git diff --check
```

The freezer refuses an existing output directory, to avoid silently changing
the frozen split or overwriting provenance. Inputs remain read-only; partial
older downloads and earlier registration remain preserved.
