# Phase 5A preregistration — before retrieval outcomes

## Binding pre-run amendment — 2026-09-09

The user-approved amendment in `docs/phase5a_dataset_amendment.md` supersedes
the original dataset, role split and preprocessing paragraphs below. They
remain here as historical evidence, not active instructions. All H1–H4,
ef32/64/128, PQ768×8, k10, L16, fixed topology and systems criteria remain
unchanged. No Phase 5A ANN retrieval result exists at amendment time.

Active source: Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M,
revision4a9731217921bc476a0f03544f11f22ae4903fa5. NumPy PCG64 permutation,
seed20260909, first990000 source row IDs are base (in permutation order),
last10000 queries (in permutation order). Sample65536 training rows without
replacement from that base only with independent PCG64 seed95000011, retain
sampling order. Training is a subset of base, NOT a third disjoint pool.
Evaluation queries are excluded from both by source row identity. This is
held-out entity-to-entity nearest-neighbor evaluation, not natural-language
query retrieval. Cast source values to FP32 BEFORE normalization; the frozen
implementation computes FP64 norm/division from those FP32 values and stores
FP32 once. Every downstream operation must reuse that archived representation.
No graph/PQ training, GT or ANN result is inspected before committing this
amendment, config, role IDs/checksums, local shard manifest and implementation.

## Historical initial registration — superseded where specified above

Recorded 2026-09-08; parent commit
`d56f1b2c8edfba8e7dc9a53eab46096b2b4d9198`;
FAISS `20f14b31a6d54e243a3d1de6ae193fc4c3ec18ed`.
No Phase 5A vector preprocessing, PQ training, graph building, ground-truth
evaluation, retrieval or latency experiment has run. Dataset acquisition is
incomplete. This document freezes design, not unverified artifact counts.

## Dataset and splits

Use DBPedia/OpenAI 1536-dimensional text embeddings, not a lower-dimensional
substitute. Preferred accessible object is the ANN-Benchmarks-format HDF5
mirror in the config. Verify full object MD5/size against server metadata,
compute SHA-256, and inspect all HDF5 shapes/attributes before using it.
The root ANN-Benchmarks URL currently returns 404. Qdrant's same-workload
archive is an alternative source, NOT permission to silently change splits.

Use all official `test` queries. From official `train`, reserve 65,536
uniformly sampled IDs without replacement (NumPy PCG64 seed 95000011) solely
for PQ training; remaining IDs, in original order, are the graph database.
Thus training/base/query roles are disjoint by source ID. Check cross-role
vector-content duplicates too; stop for a documented decision if any query
content overlaps training. This explicitly changes the benchmark base, so
provided GT is not valid for the primary experiment: recompute all GT.
Expected, NOT yet validated: 990,000 original train and 10,000 test, yielding
924,464 base after reservation. Do not claim these are measured counts.
Persist every role's original ID list and checksum and the actual counts.
Warmup uses only reserved training vectors, never a test-label-selected set.

Normalize each vector by its FP64 norm, store FP32, reject zero/nonfinite
vectors. All graph construction, PQ training, exact scoring, GT and reranking
use squared L2 on these same normalized FP32 arrays. Mathematically unit-L2
equals 2−2cosine; finite FP32 norm/rounding deviations must be measured.
Validate deterministic samples with independently calculated FP64 L2 and
cosine, documenting tolerance/boundary ties rather than silently replacing
rankings. Provisional numerical check: norm error <=2e-6, distance absolute
error <=2e-5; any non-tie top-10 mismatch stops interpretation. Archive
strict ID recall and explicit tie diagnostics separately.

## Fixed architecture and precision

One exact FP32 graph, M16, efConstruction80, seed20260907, 24 construction
threads. Save its exact serialized topology and fingerprint; parallel
construction is not promised bitwise seed-reproducible. Never rebuild across
conditions. Standard PQ768×8, dsub2, 25 training iterations, seed95000037.
This preserves SIFT's storage ratio: 6,144 FP32 bytes versus 768 PQ bytes,
8× compression. FAISS ProductQuantizer accepts positive M dividing d; there
is no M64 ceiling. Its per-query ADC table is 786,432 bytes, a real systems
cost to measure, not hide. No secondary quantizer is selected.

Before ANN: deterministic reconstruction and pairwise relative-distance /
inversion sanity checks; independent reference ADC validation. No retrieval
outcome-based quantizer selection. If implementation/resource constraints
prevent this config, report them before training/retrieval with a changed one.

## Comparisons and correctness

Only ef32/64/128, k10, L16. Exact traversal, PQ native, full PQ L0 oracle,
bounded ALL16 share graph and queries. L0 includes the admitted level-0 seed,
not upper-only evaluations. Full recording is offline diagnosis only.
Compare every bounded top16 ID/order/score and final top10 with the existing
full reference, first-evaluation-order score ties unchanged. Verify ordinary
native IDs against instrumented native IDs. Stop on unverified mismatch.
Preserve per-query GT/native/exact/oracle/ALL16 IDs, signed recalls/deltas,
L0 counts and raw candidates where modest, graph/model/input hashes.
Check exact-control oracle discrepancies and numerical ties before analysis.

The existing CandidateAccess guard allows only ef64: the future minimal
implementation change is to permit preregistered ef32/128, retaining the
same HNSW algorithm and comparator. Test at1536 dimensions against the
simple full reference before benchmark; do not optimize the new workload.

## Hypotheses and interpretation (unchanged user criteria)

H1: for practical ef with reasonable exact recall, full candidate reranking
recovers >=90% of exact-vs-PQ loss; report exact recall explicitly rather
than interpreting a low-quality exact control as navigability evidence.
H2: at least one practical ef has L16 ranking-gap recovery >=60%.
H3: bounded ALL16 lies on or improves the observed Recall/QPS frontier in
at least one useful region. Report all six points, not only favorable ones.
H4: exact refinement cost may be higher than SIFT without falsifying H1.

Use signed discovery=exact−oracle, ranking=oracle−PQ, total=exact−PQ;
undefined ratios for denominators <=1e-12. No negative-value clipping.
Distinguish A scientific+systems validity, B mechanism only with workload-
dependent refinement, C discovery mechanism fails. Do not force A.

## Systems

Same Release/generic FAISS flags as4D. Single-query worker pinned CPU2;
coordinator CPU0; internal OMP/BLAS threads1, no nested parallelism. Five
randomly ordered repetitions of all six single-thread cells; four complete
query passes/repetition, 512 training-vector warmups/worker. After these,
five repetitions of native/ALL16 ef64 with1/8 physical workers pinned2..9.
Recheck topology/availability before execution. Record actual per-query
latency, all repetitions, wall-clock throughput, machine/load/affinity and
compiler flags. Do not include load/train/build/GT in query time. Record
recall, mean/median/p95/p99, relative penalty and efficiency. No GAP policy.
ALL16 approximate extra vector reads: 98,304B/query versus8,192B on SIFT;
12×, excluding cache reuse and query data. No bandwidth-causality claim.

## Reporting and post-primary diagnostics

All raw outputs, summary tables, figures, report and exact commands follow
the requested phase5a paths. Analysis must regenerate from raw measurements.
SIFT4D reference uses its measured QPS/tails and recall; do not borrow a
different SIFT query set's discovery value without an explicit label.
Only if the frozen primary result has strong oracle but weak L16, inspect
missed GT ranks and L32/64 offline recovery. Requested L8 is invalid for
top10 among L candidates: label unavailable, do not redefine L or k.
If discovery is weak, report and stop optimization. Exactly one next
experiment after evidence; no learned/query-selective risk model.
