# Experiment log

Append one entry after every meaningful experiment. Do not overwrite prior
entries.

## Entry template

### YYYY-MM-DD — short experiment name

**Objective**

State the smallest question this experiment tests.

**Provenance**

- Git commit:
- Dataset and version/path:
- Random seed(s):
- Machine/OS/CPU/GPU/memory:
- Third-party implementation and revision:

**Parameters**

- Configuration file(s):
- Graph parameters:
- Search parameters:
- Quantizer parameters:
- Dataset preprocessing:
- Metrics and exact definitions:

**Exact command**

```bash
# Paste the non-interactive command exactly as executed.
```

**Raw outputs**

- Run directory:
- Per-query measurement file(s):

**Observed results**

Summarize measurements without interpretation. Include paired distributions or
quantiles; do not report only averages.

**Anomalies and possible confounders**

- None recorded yet.

**Smallest distinguishing next experiment**

Describe the minimum experiment that separates the leading explanations.

---

## 2026-09-04 — FAISS shared-topology synthetic backend validation

**Objective**

Test whether one HNSW graph constructed with exact FP32 squared-L2 distances
can execute both FP32 and FAISS PQ asymmetric-distance traversal without any
change to its structural fingerprint.

**Provenance**

- Git commit: `aa860b6ec42e8695daccdabb9114eaa6fc130fec`
  with a dirty worktree containing the POC under validation
- Dataset: deterministic synthetic standard-normal FP32; 4096 base vectors and
  32 independent queries, dimension 32
- Random seed: `12345`
- Machine: `rwcpu8.cse.ust.hk`; Linux 5.14.0-687.24.1.el9_8.x86_64;
  Intel Core i9-10920X; 12 cores/24 threads; 33047748608 bytes RAM
- Third-party implementation: Meta FAISS `v1.15.0`, commit
  `20f14b31a6d54e243a3d1de6ae193fc4c3ec18ed`

**Parameters**

- Configuration: `configs/indexes/faiss_hnsw_poc.conf`
- Graph: HNSW `M=16`, `efConstruction=40`, exact FP32 `IndexFlatL2` storage,
  one construction, one thread
- Search: `k=10`, `efSearch=32`, bounded queue and relative-distance check
  enabled, identical query order
- Quantizer: FAISS `IndexPQ`, `M=8`, `nbits=4`, trained on all synthetic base
  vectors with seed `12345`; FP32 queries with asymmetric distance
- Preprocessing: none
- Metric: squared L2; this validation does not compute recall

**Exact commands**

```bash
cmake -S . -B /tmp/quant-hardness-build-v4 -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/quant-hardness-build-v4 --target faiss_backend_validation -j 8
ctest --test-dir /tmp/quant-hardness-build-v4 --output-on-failure
/tmp/quant-hardness-build-v4/faiss_backend_validation configs/indexes/faiss_hnsw_poc.conf
```

**Raw outputs**

- Validation record: `docs/faiss_backend_validation.md`
- Per-query measurement files: not produced; this POC validates backend
  mechanics and deliberately does not yet add traversal or latency logging

**Observed results**

- CTest: 1/1 passed.
- Graph fingerprint for construction and both traversal modes:
  `f3d781d785ae54b4291b73b026e446ccd455744e58922b5878e8f455201ad270`.
- PQ distances differed from FP32 distances for 4096/4096 base vectors tested
  against the first query at threshold `1e-5`.
- Mean absolute distance difference: `11.5816` squared-L2 units.
- Both modes returned valid top-10 IDs for all 32 queries.
- Native FP32, guarded FP32, and alternate-storage FP32 identity-control
  searches returned exactly identical ordered IDs and distances.
- First result ID: exact `3551`; PQ `4092`.

**Anomalies and possible confounders**

- The exact and PQ first result IDs differed; cause was not tested.
- Synthetic distribution, one seed, one small 4-bit PQ configuration, no
  ground-truth recall, and no exact reranking.
- The research-only traversal-storage switch is single-threaded.
- The build used a dirty worktree; all source changes are listed by Git status
  and documented in `docs/faiss_backend_validation.md`.

**Smallest distinguishing next experiment**

On the same synthetic data, add exhaustive FP32 ground truth and preserve paired
per-query top-10 IDs and recall@10 before running SIFT1M.

---

## 2026-09-07 — Fixed-graph synthetic paired Recall@10

**Objective**

Measure the per-query distribution of exact-HNSW Recall@10 minus PQ-distance
HNSW Recall@10 while holding a single FP32-built HNSW topology and every search
parameter fixed.

**Provenance**

- Git commit: `aa860b6ec42e8695daccdabb9114eaa6fc130fec`, with a dirty
  worktree containing the implementation under validation
- Dataset: deterministic synthetic independent standard-normal FP32; 20,000
  base vectors, 500 separate queries, dimension 64
- Random seeds: database 1729, query 2718, graph 31415, PQ 16180
- Machine: `rwcpu8.cse.ust.hk`; Linux
  5.14.0-687.24.1.el9_8.x86_64; 24 hardware threads; GCC 11.5.0
- Third-party implementation: Meta FAISS `v1.15.0`, commit
  `20f14b31a6d54e243a3d1de6ae193fc4c3ec18ed`

**Parameters**

- Configuration: `configs/indexes/faiss_hnsw_paired_recall.conf`
- Graph: one HNSW construction with exact FP32 squared L2, `M=16`,
  `efConstruction=80`, one thread
- Search: `k=10`, `efSearch=16`, bounded queue and relative-distance check
  enabled; ascending query IDs in both modes
- Quantizer: FAISS `IndexPQ`, `M=8`, `nbits=4`, trained on every base vector;
  asymmetric distance from FP32 queries
- Dataset preprocessing: none
- Metric: Recall@10 against exhaustive FP32 squared-L2 top-10;
  `delta_recall = recall_exact - recall_pq`

Resolved configuration:

```text
database_seed=1729
query_seed=2718
graph_seed=31415
pq_seed=16180
base_vectors=20000
queries=500
dimension=64
k=10
hnsw_m=16
ef_construction=80
ef_search=16
pq_m=8
pq_nbits=4
```

**Exact commands**

```bash
cmake -S . -B /tmp/quant-hardness-build-v5 -G Ninja -DCMAKE_BUILD_TYPE=Release -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=ON
cmake --build /tmp/quant-hardness-build-v5 -j 8
ctest --test-dir /tmp/quant-hardness-build-v5 --output-on-failure
/tmp/quant-hardness-build-v5/faiss_paired_recall configs/indexes/faiss_hnsw_paired_recall.conf runs/phase1_synthetic_paired_recall_v1
python scripts/analyze_paired_recall.py runs/phase1_synthetic_paired_recall_v1/queries.jsonl results/tables results/figures --prefix phase1_synthetic_paired_recall_v1
```

**Raw outputs**

- Run directory: `runs/phase1_synthetic_paired_recall_v1/`
- Per-query measurements: `runs/phase1_synthetic_paired_recall_v1/queries.jsonl`
- Metadata: `runs/phase1_synthetic_paired_recall_v1/manifest.json`
- Graph fingerprint:
  `5eeca90445dd29413ba0ba7bceb4be18258ef5580174050fbaf087e2f57791e7`

**Observed results**

- Correctness tests: 2/2 passed.
- Exact Recall@10 mean/median: 0.4772/0.5. PQ Recall@10 mean/median:
  0.0708/0.0.
- Delta mean/median/min/max: 0.4064/0.4/-0.2/0.9. Empirical
  nearest-rank p90/p95/p99: 0.6/0.7/0.8.
- Delta was zero for 13/500 queries, positive for 483/500, and negative for
  4/500.
- Of positive-delta queries, 168/483, 309/483, and 374/483 accounted for at
  least 50%, 80%, and 90% of total positive recall loss, respectively.
- The graph fingerprint remained unchanged at all asserted checkpoints.

**Anomalies and possible confounders**

- Four queries had negative delta; PQ found more ground-truth IDs than exact
  HNSW for these queries.
- Native PQ results combine traversal-decision error and approximate final
  ranking error. No traversal-specific causal claim is supported.
- The single coarse 4-bit PQ setting produced low recall. This may not describe
  other PQ strengths, seeds, or distributions.
- C++ standard-normal generation is tied to the recorded standard-library
  implementation for bit-level reproducibility.

**Smallest distinguishing next experiment**

Expose the retained PQ candidate set for the same searches and exact-rerank
that unchanged set. Compare native PQ Recall@10 with reranked PQ-candidate
Recall@10 to separate final approximate-ranking loss from candidate-discovery
loss without adding trajectory logging.

---

## 2026-09-07 — Fixed-topology synthetic operating-regime calibration

**Objective**

Determine whether the previous low exact recall was an `efSearch` effect and
identify neutral high-recall operating points across standard PQ code rates.

**Provenance**

- Git commit: `aa860b6ec42e8695daccdabb9114eaa6fc130fec`, dirty worktree
  containing the implementation and earlier Phase 1 outputs
- Dataset: 20,000 base vectors and 500 independent queries, dimension 64,
  independent standard-normal FP32, no preprocessing
- Seeds: database 1729, query 2718, graph 31415, PQ 16180, distance sampling
  424242
- Machine: `rwcpu8.cse.ust.hk`; Linux
  5.14.0-687.24.1.el9_8.x86_64; Intel Core i9-10920X; 24 hardware threads;
  GCC 11.5.0
- Meta FAISS `v1.15.0`, commit
  `20f14b31a6d54e243a3d1de6ae193fc4c3ec18ed`

**Parameters**

- Configuration: `configs/indexes/faiss_hnsw_calibration.conf`
- Graph: constructed once with exact FP32 squared L2, `M=16`,
  `efConstruction=80`
- Search: `k=10`; `efSearch` 16, 32, 64, 96, 128, 160, 192, 256, 384,
  512; bounded queue and relative-distance check; one thread
- PQ: 8×4, 8×8, 16×8, and 32×8; each trained on all 20,000 base vectors
- Distance diagnostics: first 100 queries; 100 query-database pairs and 100
  query/database-pair comparisons per query and PQ
- Metric: Recall@10 against exhaustive FP32 ground truth;
  `delta_recall = recall_exact - recall_pq`

**Exact commands**

```bash
cmake -S . -B /tmp/quant-hardness-build-v5 -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/quant-hardness-build-v5 --target faiss_calibration -j 8
ctest --test-dir /tmp/quant-hardness-build-v5 --output-on-failure
/tmp/quant-hardness-build-v5/faiss_calibration configs/indexes/faiss_hnsw_calibration.conf runs/phase1_calibration_v1
python scripts/analyze_calibration.py runs/phase1_calibration_v1 results/tables results/figures --selected 160:pq_m32_nbits8,256:pq_m32_nbits8,384:pq_m32_nbits8
```

The complete resolved configuration is preserved at
`runs/phase1_calibration_v1/resolved_config.conf`.

**Raw outputs**

- Run directory: `runs/phase1_calibration_v1/`
- Exact per-query rows: 5,000 in `exact_queries.jsonl`
- Paired per-query rows: 20,000 in `paired_queries.jsonl`
- Distance/order samples: 40,000 rows each in `distance_samples.jsonl` and
  `order_samples.jsonl`
- Graph fingerprint:
  `5eeca90445dd29413ba0ba7bceb4be18258ef5580174050fbaf087e2f57791e7`

**Observed results**

- Correctness tests: 2/2 passed.
- Exact mean Recall@10 increased monotonically from 0.4772 at `efSearch=16`
  to 0.9864 at 512. Selected exact values were 0.9206, 0.9572, and 0.9784 at
  `efSearch` 160, 256, and 384.
- At the selected points, PQ32×8 recall was 0.8070, 0.8222, and 0.8328; mean
  delta was 0.1136, 0.1350, and 0.1456.
- PQ8×4 mean absolute relative distance error/inversion rate was
  0.2865/0.3265; PQ32×8 was 0.01665/0.0452.
- Full per-combination statistics are in
  `results/tables/phase1_calibration.csv` and `.json`.

**Anomalies and possible confounders**

- PQ recall saturates well below exact recall as `efSearch` increases; this
  calibration does not separate traversal and final-ranking effects.
- PQ8×4 is globally destructive in this setting and is retained as a negative
  calibration control, not a selected operating point.
- Selected PQ32×8 loss is heterogeneous but affects a majority of queries; it
  is not confined to a rare heavy tail.
- One synthetic distribution and one seed set do not establish generality.

**Smallest distinguishing next experiment**

At the three selected fixed-graph points, preserve the PQ-retained candidate
set and exact-rerank that unchanged set to decompose final-ranking loss from
candidate-discovery loss, without trajectory logging.

---

## 2026-09-07 — PQ32×8 candidate-discovery/ranking decomposition

**Objective**

Determine whether calibrated PQ32×8 Recall@10 loss comes from failure to
distance-evaluate useful ground-truth nodes or from PQ-based selection/ranking
after those nodes have been evaluated.

**Provenance**

- Git commit: `776bc6aab58856cfe346364674116c2fa6309cbb`, with a dirty
  worktree containing this decomposition implementation and outputs
- Dataset: 20,000 independent standard-normal FP32 base vectors and 500
  separately generated queries, dimension 64, no preprocessing
- Seeds: database 1729, query 2718, graph 31415, PQ 16180
- Machine: `rwcpu8.cse.ust.hk`; Linux
  5.14.0-687.24.1.el9_8.x86_64; Intel Core i9-10920X; 24 hardware threads;
  GCC 11.5.0
- Meta FAISS `v1.15.0`, commit
  `20f14b31a6d54e243a3d1de6ae193fc4c3ec18ed`

**Parameters**

- Configuration: `configs/indexes/faiss_hnsw_decomposition.conf`
- Graph: constructed once using FP32 squared L2, `M=16`,
  `efConstruction=80`; fingerprint
  `5eeca90445dd29413ba0ba7bceb4be18258ef5580174050fbaf087e2f57791e7`
- Search: `k=10`, `efSearch` 160/256/384, bounded queue and relative-distance
  check enabled, one thread, ascending query IDs
- Quantizer: FAISS PQ32×8, 32 bytes/vector, trained on all 20,000 base vectors
- Ground truth: exhaustive FP32 squared-L2 top-10
- Instrumentation: unique IDs supplied to scalar or batch-4 query-to-node
  distance-computer calls; database-to-database distances excluded

**Exact commands**

```bash
cmake -S . -B /tmp/quant-hardness-build-v5 -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/quant-hardness-build-v5 --target faiss_decomposition decomposition_correctness -j 8
ctest --test-dir /tmp/quant-hardness-build-v5 --output-on-failure
/tmp/quant-hardness-build-v5/faiss_decomposition configs/indexes/faiss_hnsw_decomposition.conf runs/phase1_decomposition_v1
python scripts/analyze_decomposition.py runs/phase1_decomposition_v1 runs/phase1_calibration_v1/paired_queries.jsonl results/tables results/figures
```

**Raw outputs**

- Run directory: `runs/phase1_decomposition_v1/`
- Per-query rows and full sorted candidate sets: `queries.jsonl` (1,500 rows)
- Manifest/config: `manifest.json`, `resolved_config.conf`
- Graph fingerprint matched calibration at every checkpoint.

**Observed results**

- Correctness tests: 3/3 passed. All 1,500 instrumented native results matched
  calibration IDs; exact-control discrepancy was zero for every query.
- Mean exact/PQ-native/PQ-oracle recalls were 0.9206/0.8070/0.9214 at
  `ef=160`, 0.9572/0.8222/0.9572 at 256, and 0.9784/0.8328/0.9792 at 384.
- Mean discovery delta was -0.0008/0/-0.0008. Mean ranking delta was
  0.1144/0.1350/0.1464. Rerank recovery was 1.0070/1.0000/1.0055.
- PQ candidate coverage exceeded exact coverage for 55, 35, and 21 queries;
  exact coverage exceeded PQ for 54, 34, and 18 queries.
- Mean evaluated-set Jaccard increased from 0.8224 to 0.8493 to 0.8710.

**Anomalies and possible confounders**

- Recovery slightly above one reflects negative mean discovery delta, not an
  exact-control discrepancy.
- “Ranking” includes PQ-based retention/selection after distance evaluation;
  it is not limited to a distinct terminal sorting pass.
- Exact reranking uses every evaluated node (roughly 3,000–5,600 per query), an
  oracle diagnostic rather than a practical reranking budget.
- One IID Gaussian dataset and one seed tuple do not establish generality.

**Smallest distinguishing next experiment**

At `efSearch=256`, repeat the identical PQ32×8 decomposition for several
independent database/query/graph/PQ seed tuples to test whether near-zero mean
discovery delta and ranking dominance are seed-stable before changing the data
distribution.
