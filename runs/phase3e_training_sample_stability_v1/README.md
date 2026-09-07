# Phase 3E raw data: balanced ABC3x3, secondary O1..O5

The pre-registration predates all nine new trainings and result inspection.
Resolved config fixes sampling seeds, nine initialization seeds, ensemble
composition and the original Phase 3C source paths. No graph traversal.

- `input_provenance.json`: hashes of original candidate chunks, exact scores,
  GT/metadata, graph file, config, pre-registration and request index; Git,
  dirty state and machine metadata. Sources stay in committed Phase 3C.
- `secondary_provenance.json`: compatibility/hash evidence for original
  Phase 3D O models and raw per-query metrics. They are read-only inputs,
  never pooled into balanced ABC variance/frequency metrics.
- `candidate_requests.tsv`: query ID, original candidate chunk, byte offset,
  count and little-endian int64 ordered-ID SHA256. Identical to Phase 3D.
- `A1` through `C3`: full standard FAISS `pq64.index` (codebook+1M codes),
  `training_ids.i32`, concatenated ADC `scores.f32`, resolved config and
  model manifest with actual subset/sample/init identity, checksums,
  code parameters, FAISS/Git/source/executable and machine information.
- `analysis/queries.jsonl.gz`: one row/query with all9 model metrics, full
  top-10 ordered lists and sets, signed loss, exact geometry and oracle,
  score offset/pool hash, variance components, harmful counts/frequencies,
  disjoint category and subset-persistent flag.
- `analysis/replacement_pairs.jsonl.gz`: all actually displaced/selected
  intruder Cartesian pairs, model/query IDs, GT flags, exact margins,
  errors and strict-inversion flags. Includes exact/PQ-tied cases.
- `analysis/inversion_pairs.jsonl.gz`: every strict inversion in >=1 model,
  with exact pair IDs/ranks/margin, nine-bit model mask in A1..C3 order,
  nine signed violations, model count/frequency, subset count/frequency,
  per-subset init counts, numbers of subsets with majority/all3 inversions.
- `analysis/pair_summary.json`: aggregate pair intersections/unions,
  model x subset frequency table and within-subset repetition table.
- `analysis/sample_overlap.json`: actual independent sample overlaps,
  including historical O; no enforced disjointness.
- `analysis/validation.json`: every-query exact/GT/order and variance
  controls; random pair identity; historical raw score audit100 queries/model.
- `primary_checkpoint`: results frozen before the optional two diagnostics.
- `ensemble`: per-query outputs for fixed A1/A2/A3 and A1/B1/C1 score
  averages (float64 uniform accumulation); no fitted weights or benchmark.

Reuse exactly512 Phase 3C random candidate pairs/query and stable first-
evaluation-order tie breaking. Exact scores remain FP32 squared L2; no
rescaling/preprocessing. Global MAE is query-weighted; all metrics retain
their Phase 3D definitions. Maximum violation includes exact-tied A x B
opportunities; strict inversion counts exclude exact ties.

Variance units are recall squared: W_pop + B_pop = T_pop (ddof0), with
B_pop representing OBSERVED variation of subset means, including finite-init
noise. Bessel-corrected W_sample/B_sample and signed B_sample-W_sample/3
are separately recorded, not clipped or claimed as formal random effects.
Different from Phase 3D, `Robust` means <=2 harmful models out of9, not zero.
Query subset-persistence requires >=2/3 harmful models in every subset;
PAIR subset-frequency instead requires >=1 inverted model in a subset.

Analysis/table/figure regeneration needs saved scores and historical runs,
not dataset vectors or FAISS search/training. The separate verification
additionally validates model/source files. No previous raw run is overwritten.
