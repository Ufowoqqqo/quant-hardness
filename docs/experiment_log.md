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

---

## 2026-09-07 — IID-Gaussian decomposition seed robustness

**Objective**

Test the pre-registered stability of near-zero mean discovery degradation and
at least 90% exact-rerank recovery over eight independent Gaussian seed tuples.

**Pre-registered criterion**

At least 7/8 replicates must simultaneously satisfy
`abs(mean delta_discovery) <= 0.01` and `rerank recovery >= 0.90`. This was
recorded in `configs/indexes/faiss_hnsw_seed_robustness.conf` before execution.

**Provenance**

- Execution-time Git HEAD: `04e4f134cb9d7459e0ee94b4a8e84690e540cab5`,
  with a dirty worktree containing the robustness runner and outputs. The
  reused CMake cache embeds its earlier configure-time base commit
  `776bc6aab58856cfe346364674116c2fa6309cbb`; replicate manifests record that
  value together with `dirty_worktree=1`.
- Eight seed tuples: database/query/graph/PQ values fully recorded in the
  resolved config and root run manifest
- Dataset per replicate: 20,000 base vectors, 500 separate queries, dimension
  64, independent isotropic standard-normal FP32, no preprocessing
- Machine: `rwcpu8.cse.ust.hk`; Linux
  5.14.0-687.24.1.el9_8.x86_64; Intel Core i9-10920X; 24 hardware threads;
  GCC 11.5.0
- Meta FAISS `v1.15.0`, commit
  `20f14b31a6d54e243a3d1de6ae193fc4c3ec18ed`

**Parameters**

- HNSW: `M=16`, `efConstruction=80`, exact FP32 construction, one graph per
  replicate
- Search: `efSearch=256`, `k=10`, bounded queue, relative-distance check, one
  thread, ascending query IDs
- Quantizer: FAISS PQ32×8, trained on all 20,000 replicate database vectors
- Ground truth: exhaustive FP32 squared-L2 top-10
- Instrumentation and decomposition definitions unchanged from
  `docs/phase1_decomposition.md`

**Exact commands**

```bash
cmake --build /tmp/quant-hardness-build-v5 --target faiss_decomposition decomposition_correctness -j 8
ctest --test-dir /tmp/quant-hardness-build-v5 --output-on-failure
python scripts/run_seed_robustness.py configs/indexes/faiss_hnsw_seed_robustness.conf /tmp/quant-hardness-build-v5/faiss_decomposition runs/phase1_seed_robustness_v1
python scripts/analyze_seed_robustness.py runs/phase1_seed_robustness_v1 results/tables results/figures
```

**Raw outputs**

- Root: `runs/phase1_seed_robustness_v1/`
- Eight replicate directories, each preserving 500 per-query rows and all
  evaluated candidate ID sets
- Replicate/aggregate tables: `results/tables/phase1_seed_robustness.*`

**Observed results**

- Pre-registered criterion: **PASS, 8/8 replicates**.
- Mean discovery delta across replicates: mean 0.000625, sample SD 0.002251,
  range [-0.0022, 0.0042].
- Rerank recovery: mean 0.995752, sample SD 0.016055, range
  [0.970629, 1.016224].
- Mean ranking delta: mean 0.137000, sample SD 0.002894, range
  [0.1340, 0.1416].
- Exact-control discrepancy was zero for all 4,000 queries. All eight graph
  fingerprints were distinct.

**Anomalies and possible confounders**

- Recovery exceeds one in four replicates because mean PQ candidate coverage
  slightly exceeds exact candidate coverage; negative discovery deltas are
  preserved.
- This check covers only IID isotropic Gaussian data and is not
  publication-quality evidence of generality.
- Standard-normal reproducibility remains tied to the recorded C++ standard
  library implementation.

**Smallest distinguishing next experiment**

Do not tune the IID baseline further. Execute the separately prepared,
currently unrun `docs/phase2_distribution_plan.md` comparison to determine
whether structured, anisotropic, or OOD query distributions produce candidate-
discovery degradation.

---

## 2026-09-07 — Phase 2A controlled query-distribution shift

**Objective and pre-registration**

Hold each Gaussian database, exact-FP32 HNSW graph, and PQ32x8 model fixed
while changing only the query mean or radial scale. Before running, the config
recorded criteria A/B/C, including no-evidence thresholds
`abs(mean delta_discovery) <= 0.01` and rerank recovery `>= 0.90`, and emerging
thresholds mean discovery delta `>= 0.03` and recovery `<= 0.80` in at least
3/4 replicates with exact Recall@10 `>= 0.90`.

**Configuration and provenance**

- Git HEAD recorded by the binary:
  `f56bdcbf462193b1f9d3e733e35a9b5d05401b0e`, with the Phase 2A worktree
  dirty; FAISS commit
  `20f14b31a6d54e243a3d1de6ae193fc4c3ec18ed`
- Four seed tuples for database/query/graph/PQ/direction are in
  `configs/indexes/faiss_hnsw_phase2a_query_shift.conf`
- Per replicate: 20,000 database vectors from `N(0,I_64)`, 500 paired latent
  query vectors, dimension 64, no preprocessing
- Mean shift: alpha `{0,.25,.50,.75,1}` along one fixed random unit direction;
  radial shift: sigma `{1,1.25,1.5,2}`
- HNSW: exact construction, `M=16`, `efConstruction=80`; fixed search
  `efSearch=256`, `k=10`, bounded queue, relative-distance check, one thread
- Exact calibration grid, used only when fixed-ef mean exact recall was below
  0.90: `{256,384,512,768,1024}`; first ef reaching 0.95 selected
- PQ: 32 subquantizers, 8 bits, trained on and encoding all 20,000 database
  vectors once per replicate
- Machine: `rwcpu8.cse.ust.hk`; Linux
  5.14.0-687.24.1.el9_8.x86_64; GCC 11.5.0; 24 hardware threads

**Exact commands**

```bash
cmake -S . -B /tmp/quant-hardness-build-v5 -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/quant-hardness-build-v5 --target faiss_query_shift decomposition_correctness -j 8
ctest --test-dir /tmp/quant-hardness-build-v5 --output-on-failure
/tmp/quant-hardness-build-v5/faiss_query_shift configs/indexes/faiss_hnsw_phase2a_query_shift.conf runs/phase2a_query_shift_v1
python -m py_compile scripts/analyze_phase2a_query_shift.py
python scripts/analyze_phase2a_query_shift.py runs/phase2a_query_shift_v1 results/tables results/figures
```

**Observed results**

- Criterion A held for all 21 fixed-ef shifted replicate/condition rows with
  exact recall at least 0.90. No condition met criterion B or C.
- Across conditions, mean discovery delta was -0.0002 to 0.0020 and mean
  rerank recovery was 0.9860 to 1.0022.
- At fixed ef, exact recall declined to 0.9070 at alpha=1 and 0.8353 at
  sigma=2, averaged across replicates. The seven triggered calibrations all
  restored exact recall to at least 0.95 using ef512 or ef1024.
- At matched points, discovery delta was -0.0018 to 0.0026 and recovery was
  0.9779 to 1.0158.
- IID raw outputs exactly reproduced Phase 1 robustness replicates 0--3 for
  ground truth, native/oracle results, recalls, discovery delta, and Jaccard.

**Anomalies and possible confounders**

- `delta_exact_control` was +0.1 for 2 of 16,000 fixed rows and zero otherwise
  (overall mean 0.0000125). The all-level distance recorder includes upper
  greedy evaluations not necessarily retained by the level-0 result heap.
  These raw observations are preserved and the tiny positive oracle bias is
  not silently corrected.
- Positive and negative discovery deltas both became more frequent with OOD
  severity, while their mean stayed near zero.
- More severe fixed-ef shifts made exact HNSW harder; only separately reported
  matched-recall results support mechanism comparisons there.
- The experiment changes queries around an isotropic Gaussian database and
  does not address structured database geometry.

**Single next experiment**

Run a four-replicate Phase 2B clustered/mixture database experiment with IID
queries and the same decomposition plus matched-exact-recall control. Do not
add trajectories unless a reproducible discovery signal first emerges.

---

## 2026-09-07 — L0 candidate semantics correction and Phase 2B clustered geometry

**Instrumentation correction**

The legacy recorder mixed upper-level greedy evaluations with level-0 search.
The new phase-separated adapter calls FAISS's existing
`greedy_update_nearest` and `search_level_0` implementations and defines the
discovery oracle over `V_L0`. It records `V_upper_only` separately and
re-evaluates the selected L0 seed solely for phase attribution.

Both known Phase 2A anomalies reproduced all-level
`delta_exact_control=0.1` and became exactly zero with L0 semantics. Native
IDs/distances and graph fingerprints were unchanged. Complete 500-query IID
and alpha=1 regressions yielded mean discovery 0.0042 and approximately zero,
with recovery 0.970629 and 1.0. The correction did not materially change the
prior conclusion, so Phase 2B proceeded.

**Phase 2B pre-registration and parameters**

- Four seed tuples, 20,000 database vectors, 500 separate queries, dimension
  64, 16 uniform mixture components
- Generator: `sqrt(1-rho) z + sqrt(rho*64) c_j`, using fixed random unit
  centers and rho `{0,.10,.25,.50,.75,.90}`
- Query and database distributions identical within every condition; latent
  vectors, centers, and labels paired across rho
- One condition-specific FP32 HNSW graph: `M=16`, `efConstruction=80`
- Search: PQ32x8 versus exact FP32, `efSearch=256`, `k=10`, bounded queue,
  relative-distance check, one thread, ascending query IDs
- Prescribed exact-only calibration grid `{256,384,512,768,1024}` if exact
  mean recall fell below 0.90; no condition triggered it
- PQ trained on and encoding all 20,000 condition-specific database vectors
- PQ quality: first 100 queries, 100 sampled distances and 100 sampled pairs
  per query, using fixed per-replicate sample IDs across rho
- Thresholds were recorded in
  `configs/indexes/faiss_hnsw_phase2b_clustered_geometry.conf` before execution
- Git HEAD embedded by the binary:
  `cbc944c773ac7d38d1f514d66aa46d87341e3cc9`, dirty worktree; FAISS commit
  `20f14b31a6d54e243a3d1de6ae193fc4c3ec18ed`
- Machine: `rwcpu8.cse.ust.hk`, Linux
  5.14.0-687.24.1.el9_8.x86_64, GCC 11.5.0, 24 hardware threads

**Exact commands**

```bash
cmake -S . -B /tmp/quant-hardness-build-v5 -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/quant-hardness-build-v5 --target decomposition_correctness candidate_semantics_regression faiss_clustered_geometry -j 8
/tmp/quant-hardness-build-v5/decomposition_correctness
/tmp/quant-hardness-build-v5/candidate_semantics_regression
/tmp/quant-hardness-build-v5/faiss_clustered_geometry configs/indexes/faiss_hnsw_phase2b_clustered_geometry.conf runs/phase2b_clustered_geometry_v1
python -m py_compile scripts/analyze_phase2b_clustered_geometry.py
python scripts/analyze_phase2b_clustered_geometry.py runs/phase2b_clustered_geometry_v1 results/tables results/figures
ctest --test-dir /tmp/quant-hardness-build-v5 --output-on-failure
```

**Observed results**

- All 24 conditions passed the no-discovery-breakdown criterion; none met the
  emerging or strong geometry criteria.
- Mean discovery delta stayed between -0.0003 and 0.0020. Rerank recovery was
  0.9860--1.0018 and converged to approximately one at high rho.
- Exact recall increased from 0.9649 at rho zero to about 0.9997; no matched
  run was needed.
- PQ-native recall fell from 0.8253 to 0.7117, but PQ L0-oracle recall tracked
  exact recall. Ranking delta, not discovery delta, increased to 0.2881.
- Random-pair PQ mean absolute error fell from 2.1316 to 1.2939 and strict
  inversion rate from 0.0425 to 0.0320 as rho increased.
- `delta_exact_control` was exactly zero for all 12,000 queries. Rho zero
  exactly reproduced Phase 1 native baselines for replicates 0--3.

**Anomalies and possible confounders**

- At rho at least 0.5, nearly every ground-truth top-10 lies within one
  generating component. Global random-pair error sampling increasingly
  measures easy cross-cluster comparisons and may not reflect fine local
  ranking quality.
- L0 evaluation counts decrease strongly with rho, but exact and PQ counts
  remain matched and PQ L0-oracle recall stays near one.
- Several high-rho per-query Spearman correlations are undefined because
  discovery delta is constant at zero.
- Balanced spherical mixtures do not represent every structured or real
  embedding distribution.

**Single next experiment**

Run one SIFT1M fixed-topology L0 decomposition at calibrated high exact recall,
with PQ32x8 and the same exact-rerank and PQ-quality controls. Do not add full
trajectory logging unless a reproducible discovery signal appears.

---

## 2026-09-07 — Phase 3A SIFT1M real-data gatekeeper

**Pre-registration and configuration**

The grids, selection rule, and gatekeeper thresholds were written to
`configs/indexes/faiss_hnsw_phase3a_sift1m.conf` and
`docs/phase3a_sift1m.md` before decomposition. The actual primary PQ choice was
made from native calibration only.

- Standard SIFT1M: 1,000,000 base, 10,000 query, dimension 128, 100 provided
  GT neighbors; all queries used
- Data revision: Hugging Face `qbo-odp/sift1m` commit
  `bd8ccad6c2a0a0a3a7519f6d37c0e5a2d59fe55b`
- HNSW: `M=16`, `efConstruction=80`, seed 20260907; one FP32 graph
- Exact efSearch grid: `{16,32,48,64,96,128,192,256,384,512}`
- PQ grid: `{16,32,64} x 8` bits, seed 30260907, trained with the full database
  as input; primary selected as PQ64x8
- Selected decomposition efSearch: 32, 64, 128; `k=10`, bounded queue,
  relative-distance check, one search thread
- Candidate semantics: corrected L0-only; first 100 candidate sets retained
- Git HEAD embedded by the binary:
  `4e5137dd71e64df669eeb4c18b9f4af3aee7a556`, dirty worktree; FAISS
  `20f14b31a6d54e243a3d1de6ae193fc4c3ec18ed`
- Machine: `rwcpu8.cse.ust.hk`, Linux 5.14.0-687.24.1.el9_8.x86_64, GCC
  11.5.0, 24 hardware threads

**Exact commands**

```bash
# Data acquisition used the fixed repository revision shown above.
HF_HOME=/tmp/qh_hf_cache HF_XET_CACHE=/tmp/qh_hf_cache/xet \
  /tmp/qh_hf_download/bin/hf download qbo-odp/sift1m sift_base.fvecs \
  --repo-type dataset \
  --revision bd8ccad6c2a0a0a3a7519f6d37c0e5a2d59fe55b \
  --local-dir /rwproject/kdd-db/kluaq/dataset/sift1m/hf_download \
  --max-workers 8

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target faiss_sift1m dataset_loader_correctness decomposition_correctness candidate_semantics_regression faiss_backend_validation phase1_correctness -j4
ctest --test-dir build --output-on-failure

./build/faiss_sift1m prepare \
  configs/indexes/faiss_hnsw_phase3a_sift1m.conf \
  runs/phase3a_sift1m_v1
./build/faiss_sift1m calibrate \
  configs/indexes/faiss_hnsw_phase3a_sift1m.conf \
  runs/phase3a_sift1m_v1
./build/faiss_sift1m decompose \
  configs/indexes/faiss_hnsw_phase3a_sift1m.conf \
  runs/phase3a_sift1m_v1
python scripts/analyze_phase3a_sift1m.py \
  runs/phase3a_sift1m_v1 results/tables results/figures
```

**Observed results**

- A fixed 100-query GT sample matched exhaustive FP32 top-10 exactly.
- Exact recall was 0.88629, 0.95319, and 0.98379 at selected efSearch 32, 64,
  and 128.
- PQ64 native recall was 0.80964, 0.85245, and 0.86867. Exact reranking of
  `V_L0_pq` produced 0.88402, 0.95200, and 0.98287.
- Mean discovery delta was 0.00227, 0.00119, and 0.00092; mean ranking delta
  was 0.07438, 0.09955, and 0.11420.
- Rerank recovery was 0.97038, 0.98819, and 0.99201.
- At the exact-recall-qualified points, the no-real-data-discovery-signal
  criterion passed; emerging and strong criteria failed.

**Anomalies and possible confounders**

- efSearch 32 was selected by the pre-registered closest-target rule but exact
  recall was 0.88629; it is excluded from criteria requiring at least 0.90.
- Exact distance ties at the top-10 boundary invalidated the synthetic-only
  equality between provided-GT coverage and ID-based oracle recall. The raw
  metrics remain separate. Every mismatch was verified as an exact boundary
  tie, and exact-native was independently verified as a valid exact rerank of
  `V_L0`. Two 93-row failed-guard files were retained, not overwritten. Three
  completed files with the superseded per-query field name
  `rerank_recovery_query` were retained under `superseded_schema/` and excluded
  from analysis; replacement files use `rerank_recovery`. The run-level
  metadata amendment records the stage history and completed binary/source
  hashes.
- One dataset and one frozen graph/PQ seed do not establish broad real-data
  external validity. Random-pair PQ diagnostics underweight local errors.
- Build tooling reported NFS/host clock skew of up to about four minutes;
  clean compilation and all five tests passed.

**Single next experiment**

Pivot to a reference-only SIFT1M ranking/selection analysis on fixed recorded
PQ candidate sets, relating exact-versus-PQ selection errors to local distance
margins and PQ distance errors. Do not add trajectory instrumentation or a new
algorithm.
# Phase 3B — SIFT1M fixed-candidate ranking (completed)

**Objective and pre-registration**

Test native/global PQ selection equivalence first, then characterize fixed-
candidate score errors. H1–H4 and tie/bin/sampling conventions were written to
`docs/phase3b_fixed_candidate_ranking.md` before results and preserved as
`runs/phase3b_fixed_candidate_ranking_v1/preregistration.md`. Full ef64 analysis
and heap audit preceded ef128; the checkpoint is retained in `primary_analysis/`.

**Exact configuration**

- `configs/indexes/phase3b_fixed_candidate_ranking.conf`, inheriting dataset,
  graph and PQ provenance from `configs/indexes/faiss_hnsw_phase3a_sift1m.conf`.
- SIFT1M: 1M database, 10k queries, d=128, provided top-10 GT; unchanged files
  with SHA-256 verification on both exports. All scores use FP32 squared L2.
- Existing frozen M16/efConstruction80 HNSW, graph seed 20260907, PQ64x8 seed
  30260907. No rebuilding/training/encoding. efSearch=64 primary, then 128.
- k=10, one search thread, bounded_queue=true, check_relative_distance=true.
- 512 distinct-item candidate pairs/query, PCG64 seed=60300001+query_id;
  ten quantile bins, ties unsplit; zero-margin ratios in a separate bin.
- Git HEAD during execution: `9a42b60540f90a90aaacacc71232f13569e9d8b1`, dirty
  Phase 3B sources. FAISS `20f14b31a6d54e243a3d1de6ae193fc4c3ec18ed`.
- Host rwcpu8.cse.ust.hk, GCC 11.5.0, Linux 5.14.0-687.24.1.el9_8.x86_64.
  Per-export manifests record machine and executable/source/frozen-index hashes.

**Exact commands, in execution order**

```bash
python -m venv --system-site-packages /tmp/phase3b-plot-env
/tmp/phase3b-plot-env/bin/python -m pip install matplotlib==3.9.4
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target faiss_sift1m -j2
python -m unittest discover -s tests -p test_phase3b_metrics.py -v
ctest --test-dir build --output-on-failure
./build/faiss_sift1m ranking-export configs/indexes/phase3b_fixed_candidate_ranking.conf runs/phase3b_fixed_candidate_ranking_v1 64
cp docs/phase3b_fixed_candidate_ranking.md runs/phase3b_fixed_candidate_ranking_v1/preregistration.md
python scripts/analyze_phase3b_fixed_candidate_ranking.py derive configs/indexes/phase3b_fixed_candidate_ranking.conf runs/phase3b_fixed_candidate_ranking_v1 runs/phase3b_fixed_candidate_ranking_v1/analysis 64
```

The first derivation stopped on a non-JSON-serializable NumPy Boolean in the
harmful-pair output. After casting that value to a Python bool and adding a
serialization regression assertion, execution continued with:

```bash
mv runs/phase3b_fixed_candidate_ranking_v1/analysis/ef64 runs/phase3b_fixed_candidate_ranking_v1/analysis/failed_ef64_numpy_bool
python -m unittest discover -s tests -p test_phase3b_metrics.py -v
python scripts/analyze_phase3b_fixed_candidate_ranking.py derive configs/indexes/phase3b_fixed_candidate_ranking.conf runs/phase3b_fixed_candidate_ranking_v1 runs/phase3b_fixed_candidate_ranking_v1/analysis 64
MPLCONFIGDIR=/tmp/phase3b-mpl /tmp/phase3b-plot-env/bin/python scripts/analyze_phase3b_fixed_candidate_ranking.py summarize configs/indexes/phase3b_fixed_candidate_ranking.conf runs/phase3b_fixed_candidate_ranking_v1/analysis results/tables results/figures
python scripts/audit_phase3b_native_ties.py runs/phase3b_fixed_candidate_ranking_v1/ef64 runs/phase3b_fixed_candidate_ranking_v1/analysis/ef64
mkdir -p runs/phase3b_fixed_candidate_ranking_v1/primary_analysis
cp results/tables/phase3b_*.csv results/tables/phase3b_*.json runs/phase3b_fixed_candidate_ranking_v1/primary_analysis/
cp docs/phase3b_fixed_candidate_ranking.md runs/phase3b_fixed_candidate_ranking_v1/primary_analysis/checkpoint.md
./build/faiss_sift1m ranking-export configs/indexes/phase3b_fixed_candidate_ranking.conf runs/phase3b_fixed_candidate_ranking_v1 128
python scripts/analyze_phase3b_fixed_candidate_ranking.py derive configs/indexes/phase3b_fixed_candidate_ranking.conf runs/phase3b_fixed_candidate_ranking_v1 runs/phase3b_fixed_candidate_ranking_v1/analysis 128
python scripts/audit_phase3b_native_ties.py runs/phase3b_fixed_candidate_ranking_v1/ef128 runs/phase3b_fixed_candidate_ranking_v1/analysis/ef128
MPLCONFIGDIR=/tmp/phase3b-mpl /tmp/phase3b-plot-env/bin/python scripts/analyze_phase3b_fixed_candidate_ranking.py summarize configs/indexes/phase3b_fixed_candidate_ranking.conf runs/phase3b_fixed_candidate_ranking_v1/analysis results/tables results/figures
```

Independent re-derivation used the directory returned by
`mktemp -d /tmp/phase3b-reproduce.XXXXXX`:

```bash
python scripts/analyze_phase3b_fixed_candidate_ranking.py derive configs/indexes/phase3b_fixed_candidate_ranking.conf runs/phase3b_fixed_candidate_ranking_v1 /tmp/phase3b-reproduce.Jd3t79/analysis 64
python scripts/analyze_phase3b_fixed_candidate_ranking.py derive configs/indexes/phase3b_fixed_candidate_ranking.conf runs/phase3b_fixed_candidate_ranking_v1 /tmp/phase3b-reproduce.Jd3t79/analysis 128
MPLCONFIGDIR=/tmp/phase3b-mpl /tmp/phase3b-plot-env/bin/python scripts/analyze_phase3b_fixed_candidate_ranking.py summarize configs/indexes/phase3b_fixed_candidate_ranking.conf /tmp/phase3b-reproduce.Jd3t79/analysis /tmp/phase3b-reproduce.Jd3t79/tables /tmp/phase3b-reproduce.Jd3t79/figures
```

Final summary generation was repeated after adding GT-exchange counts and a
threshold-curve figure; these additions did not change any query metric.
Comparison of all query JSONL, harmful-pair JSONL, random-pair binary, table and
figure files was byte-identical. Reproduction commands require new derivation
directories; existing raw or derived per-ef outputs are not overwritten.

```bash
python -m unittest discover -s tests -p test_phase3b_metrics.py -v
git diff --check
/tmp/phase3b-plot-env/bin/python scripts/verify_phase3b_artifacts.py runs/phase3b_fixed_candidate_ranking_v1 /tmp/phase3b-reproduce.Jd3t79 results/tables results/figures runs/phase3b_fixed_candidate_ranking_v1/final_verification.json
```

`final_verification.json` records checked raw input hashes, final code/report/
result hashes, Python package versions, and independently reproduced files.

**Observations**

- 10,343,205 / 18,520,015 candidate records at ef64/128.
- Native/global PQ membership differs in 74/75 queries; ordered lists differ
  in 697/677. All are exact PQ ties. Independent result-heap reference matches
  every native list, including all tied cases.
- Oracle/global-PQ recall: .95200/.85246 at ef64, .98287/.86869 at ef128.
- Stable/Benign/Harmful/Lucky: 19.33/10.52/70.15/0% at ef64,
  18.47/3.89/77.64/0% at ef128.
- Mean ranking loss .09954/.11418; disagreement .12018/.12259.
- Spearman with loss: candidate MAE -.0700/-.0550; relative boundary margin
  -.1970/-.2579; local error/margin .2851/.3287; cross-boundary inversion count
  .6998/.7981; random candidate-pair inversions .0617/.0653.
- All native/oracle IDs, counts and retained candidate sets reproduce Phase
  3A. Frozen graph/PQ hashes and scalar/batch ADC checks pass. Five CTests and
  six Python reference tests pass.

**Anomalies and confounders**

- Tie-dependent IDs are retained with separate set/list/modulo-tie metrics.
  Native versus global-PQ recall differs by .00001/.00002 due to ties.
- Adjacent k/k+1 overtake holds for fewer than half of harmful queries.
  Boundary-locality should not be reduced to that single adjacent pair.
- Critical inversions are post-scoring and structurally related to the outcome;
  their strong correlation does not establish independent predictive ability.
- Random pair rates use 512 samples/query and have sampling noise. One dataset,
  graph and PQ model do not establish external validity.
- The excluded partial serialization-failure output and all prior experiments
  are preserved. Existing filesystem clock-skew warnings occurred during build.

**One smallest next experiment**

Hold ef64 candidate sets fixed and score them with the already-trained Phase
3A PQ32x8 and PQ64x8 models, comparing within-query changes in cross-boundary
inversions and recall loss at identical exact margins and GT. This has not been
run here and proposes no new quantizer or traversal instrumentation.
# Phase 3C — fixed exact-candidate-pool precision transition (completed)

**Pre-registration and fixed inputs**

H1–H5 and category/tie/sampling/margin conventions were written before export
to `docs/phase3c_precision_transition.md`, with a preserved preregistration in
`runs/phase3c_precision_transition_v1/preregistration.md`.

Config: `configs/indexes/phase3c_precision_transition.conf`, inheriting the
unchanged Phase 3A dataset paths/checksums, graph construction and PQ seeds.
SIFT1M: 1,000,000 base vectors, 10,000 queries, 128 dimensions, k=10. One
exact-FP32-traversal L0 candidate pool per query, efSearch=64. Existing M16,
efConstruction80 graph, seed 20260907; existing PQ32x8/PQ64x8 models, seed
30260907. No building, training, encoding or new traversal instrumentation.
Search remains single-threaded with bounded_queue/check_relative_distance true.
Random pairs: 512/query, PCG64(seed=60400001+query_id), shared across PQ scores.
Margins are squared L2; small margin is pre-declared as pair gap/dk<=.05,
large as >.10, with .01/.05/.10 tables and rank-based controls.

Execution HEAD `170cf03a9f20ba1629dce2ed905e3ab06df15600` (dirty Phase 3C
implementation), FAISS `20f14b31a6d54e243a3d1de6ae193fc4c3ec18ed`.
Machine rwcpu8.cse.ust.hk, GCC 11.5.0, Linux 5.14.0-687.24.1.el9_8.x86_64.
NumPy 1.23.5, Matplotlib 3.9.4 in the existing Phase 3B plotting environment.
All binary/source/index hashes are recorded with the run.

**Exact commands — primary, then optional control**

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target faiss_sift1m -j2
python -m unittest discover -s tests -p test_phase3c_metrics.py -v
mkdir -p runs/phase3c_precision_transition_v1
cp docs/phase3c_precision_transition.md runs/phase3c_precision_transition_v1/preregistration.md
./build/faiss_sift1m precision-export configs/indexes/phase3c_precision_transition.conf runs/phase3c_precision_transition_v1 exact_pool
ctest --test-dir build --output-on-failure
python scripts/analyze_phase3c_precision_transition.py derive configs/indexes/phase3c_precision_transition.conf runs/phase3c_precision_transition_v1 runs/phase3c_precision_transition_v1/analysis exact_pool
MPLCONFIGDIR=/tmp/phase3b-mpl /tmp/phase3b-plot-env/bin/python scripts/analyze_phase3c_precision_transition.py summarize configs/indexes/phase3c_precision_transition.conf runs/phase3c_precision_transition_v1/analysis results/tables results/figures exact_pool
mkdir -p runs/phase3c_precision_transition_v1/primary_checkpoint
cp results/tables/phase3c_exact_pool_* runs/phase3c_precision_transition_v1/primary_checkpoint/
cp docs/phase3c_precision_transition.md runs/phase3c_precision_transition_v1/primary_checkpoint/checkpoint.md
python scripts/analyze_phase3c_precision_transition.py prepare-control configs/indexes/phase3c_precision_transition.conf runs/phase3c_precision_transition_v1
./build/faiss_sift1m precision-export configs/indexes/phase3c_precision_transition.conf runs/phase3c_precision_transition_v1 pq64_pool_control
python scripts/analyze_phase3c_precision_transition.py derive configs/indexes/phase3c_precision_transition.conf runs/phase3c_precision_transition_v1 runs/phase3c_precision_transition_v1/analysis pq64_pool_control
MPLCONFIGDIR=/tmp/phase3b-mpl /tmp/phase3b-plot-env/bin/python scripts/analyze_phase3c_precision_transition.py summarize configs/indexes/phase3c_precision_transition.conf runs/phase3c_precision_transition_v1/analysis results/tables results/figures pq64_pool_control
```

After the primary result revealed different oracle recall across stability
groups, an explicitly post-hoc geometry sensitivity table restricted the
comparison to oracle recall=1. The two summarize commands above were rerun
after this addition and completion of descriptive change/cutoff tables; no
per-query metric, threshold, candidate pool or score was changed.

Independent re-derivation used `mktemp -d /tmp/phase3c-reproduce.XXXXXX`, which
returned `/tmp/phase3c-reproduce.O5YT0s`:

```bash
python scripts/analyze_phase3c_precision_transition.py derive configs/indexes/phase3c_precision_transition.conf runs/phase3c_precision_transition_v1 /tmp/phase3c-reproduce.O5YT0s/analysis exact_pool
python scripts/analyze_phase3c_precision_transition.py derive configs/indexes/phase3c_precision_transition.conf runs/phase3c_precision_transition_v1 /tmp/phase3c-reproduce.O5YT0s/analysis pq64_pool_control
MPLCONFIGDIR=/tmp/phase3b-mpl /tmp/phase3b-plot-env/bin/python scripts/analyze_phase3c_precision_transition.py summarize configs/indexes/phase3c_precision_transition.conf /tmp/phase3c-reproduce.O5YT0s/analysis /tmp/phase3c-reproduce.O5YT0s/tables /tmp/phase3c-reproduce.O5YT0s/figures exact_pool
MPLCONFIGDIR=/tmp/phase3b-mpl /tmp/phase3b-plot-env/bin/python scripts/analyze_phase3c_precision_transition.py summarize configs/indexes/phase3c_precision_transition.conf /tmp/phase3c-reproduce.O5YT0s/analysis /tmp/phase3c-reproduce.O5YT0s/tables /tmp/phase3c-reproduce.O5YT0s/figures pq64_pool_control
python -m unittest discover -s tests -p 'test_phase3*_metrics.py' -v
git diff --check
/tmp/phase3b-plot-env/bin/python scripts/verify_phase3c_artifacts.py runs/phase3c_precision_transition_v1 /tmp/phase3c-reproduce.O5YT0s results/tables results/figures runs/phase3c_precision_transition_v1/final_verification.json
```

**Observations**

- Primary: 10,353,047 candidate records. Exact/PQ32/PQ64 mean recall
  .95315/.70398/.85313. Mean loss .24917 -> .10002; mean critical inversions
  33.3636 -> 5.5846.
- 78.71% improve, 27.31% become fully correct, 69.06% remain harmful under
  both scorers, 4.08% regress. Flags overlap exactly as requested.
- Change correlation with recall recovery: critical count .6020, global MAE
  .0503, random inversion rate .0829. All 408 regressions occur despite lower
  global MAE; 169 occur despite fewer total critical inversions.
- 310,999 corrected, 22,637 persistent and 33,209 new inversions. Corrected
  relative margin median .09449; only 19.87% are <=.05 and 46.54% exceed .10.
  H5 is not supported under its pre-registered interpretation. Persistent
  and new errors are more concentrated near the exact selection boundary.
- The 1,000-query recorded-PQ64-pool control gives .952/.7066/.8478 recall,
  with critical-change correlation .6080 versus global-MAE .0640, and only
  20.30% small-margin corrected pairs. The same-query exact-pool subset is
  reported separately to control sampling differences.
- 13 Python tests (7 Phase 3C, 6 Phase 3B) and all 5 CTests pass. Independent
  raw-score derivation and table/figure regeneration are byte-identical.

**Anomalies and confounders**

- Stable exact sorting differs from native exact membership in 78 tied cases,
  with 18 GT-hit gains and 22 losses; mean oracle .95315 versus native .95319.
  Score vectors are identical. No native substitution or old-metric rewrite.
- Stable-both group oracle recall .818 versus persistent group .974 is an
  important geometry-comparison confounder. The added oracle=1 sensitivity
  retains the local-gap direction but has only 82 stable queries and is
  explicitly exploratory.
- Harmful-query persistence is high partly because harmful prevalence is
  high; it does not establish an intrinsic hard-query class.
- Critical counts/displaced counts are post-scoring and structurally linked
  to recall; random pair rates have finite-sampling noise. Separate non-nested
  codebooks change partition and centroid geometry as well as byte rate.
- Existing filesystem clock-skew build warnings persisted. Compilation,
  linking and correctness checks completed. All prior raw data are preserved.

**One smallest next experiment**

Keep the exact candidate pools fixed; train three independent-seed standard
PQ64x8 codebooks under identical settings and score only these pools. Measure
harmful-query and residual-inversion persistence with prevalence context and
the complete-GT stratum. This tests codebook-specific versus repeatable local
sensitivity; it has not been run and introduces no new algorithm.

## Phase 3D — frozen-candidate quantizer initialization stability

**Protocol recorded before new results**

Git execution base: `71cb5be96e7cd062127282b48c715841d55d9ff5`, dirty
implementation sources with binary/source/config hashes. Pinned FAISS:
`20f14b31a6d54e243a3d1de6ae193fc4c3ec18ed`. Configuration:
`configs/indexes/phase3d_quantizer_seed_stability.conf`; pre-registration:
`runs/phase3d_quantizer_seed_stability_v1/preregistration.md`.
All 10,000 SIFT queries, identical saved FP32 L0 candidate pools, exact
scores and GT from Phase 3C, ef64/k10 provenance, PQ64x8. Fixed original
65,536-vector training subset/order (FAISS rand_perm seed30260907), five
initialization seeds 30260907/70300001/70300019/70300043/70300067. See report
for exact sampling-source reasoning; seed0 must reproduce the original
codebook/codes/scores before continuing. No graph search or cache write.

**Commands executed: preparation, model generation, correctness**

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target faiss_seed_stability -j2
/tmp/phase3b-plot-env/bin/python scripts/analyze_phase3d_quantizer_seed_stability.py prepare
/tmp/phase3b-plot-env/bin/python -m unittest discover -s tests -p 'test_phase3*_metrics.py' -v
ctest --test-dir build --output-on-failure
set -o pipefail
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=24 ./build/faiss_seed_stability seed-scores configs/indexes/phase3d_quantizer_seed_stability.conf runs/phase3d_quantizer_seed_stability_v1 0 2>&1 | tee runs/phase3d_quantizer_seed_stability_v1/seed_0_execution.log
set -euo pipefail
for model in 1 2 3 4; do
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=24 ./build/faiss_seed_stability seed-scores configs/indexes/phase3d_quantizer_seed_stability.conf runs/phase3d_quantizer_seed_stability_v1 "$model" 2>&1 | tee "runs/phase3d_quantizer_seed_stability_v1/seed_${model}_execution.log"
done
```

The same 23 Python tests (10 Phase 3D, 13 prior) pass after adding a
random small-set exhaustive reference test. All 5 CTests pass. Seed0
training, encoding and all 10,353,047 scores are identical to Phase 3C.
Existing filesystem clock-skew build warnings recur; the separate target
compiled and linked successfully, without replacing the old Phase 3C binary.

**Commands executed: primary analysis, checkpoint, diagnostic**

```bash
set -o pipefail
/tmp/phase3b-plot-env/bin/python scripts/analyze_phase3d_quantizer_seed_stability.py derive 2>&1 | tee runs/phase3d_quantizer_seed_stability_v1/derive_execution.log
MPLCONFIGDIR=/tmp/phase3b-mpl /tmp/phase3b-plot-env/bin/python scripts/analyze_phase3d_quantizer_seed_stability.py summarize 2>&1 | tee runs/phase3d_quantizer_seed_stability_v1/summary_execution.log
/tmp/phase3b-plot-env/bin/python scripts/analyze_phase3d_quantizer_seed_stability.py checkpoint
MPLCONFIGDIR=/tmp/phase3b-mpl /tmp/phase3b-plot-env/bin/python scripts/analyze_phase3d_quantizer_seed_stability.py ensemble 2>&1 | tee runs/phase3d_quantizer_seed_stability_v1/ensemble_execution.log
```

Independent regeneration used `mktemp -d /tmp/phase3d-reproduce.XXXXXX`,
which returned `/tmp/phase3d-reproduce.OSza6h`:

```bash
set -euo pipefail
/tmp/phase3b-plot-env/bin/python scripts/analyze_phase3d_quantizer_seed_stability.py derive --derived /tmp/phase3d-reproduce.OSza6h/analysis
MPLCONFIGDIR=/tmp/phase3b-mpl /tmp/phase3b-plot-env/bin/python scripts/analyze_phase3d_quantizer_seed_stability.py summarize --derived /tmp/phase3d-reproduce.OSza6h/analysis --tables /tmp/phase3d-reproduce.OSza6h/tables --figures /tmp/phase3d-reproduce.OSza6h/figures
MPLCONFIGDIR=/tmp/phase3b-mpl /tmp/phase3b-plot-env/bin/python scripts/analyze_phase3d_quantizer_seed_stability.py ensemble --derived /tmp/phase3d-reproduce.OSza6h/ensemble --tables /tmp/phase3d-reproduce.OSza6h/tables --figures /tmp/phase3d-reproduce.OSza6h/figures
/tmp/phase3b-plot-env/bin/python scripts/verify_phase3d_artifacts.py runs/phase3d_quantizer_seed_stability_v1 /tmp/phase3d-reproduce.OSza6h --output /tmp/phase3d-reproduce.OSza6h/preverification.json
/tmp/phase3b-plot-env/bin/python -m unittest discover -s tests -p 'test_phase3*_metrics.py' -v
git diff --check
/tmp/phase3b-plot-env/bin/python scripts/verify_phase3d_artifacts.py runs/phase3d_quantizer_seed_stability_v1 /tmp/phase3d-reproduce.OSza6h --output runs/phase3d_quantizer_seed_stability_v1/final_verification.json
```

**Observations, without causal interpretation**

- Across five models, recall mean .853052, sample SD .00024035, range
  .85266–.85331; mean ranking loss .100098; mean critical inversions 5.54252.
- Harmful categories: robust647 (6.47%), occasional1812 (18.12%), usual3841
  (38.41%), persistent3700 (37%). They contribute 0/6.32/35.15/58.53% of
  total signed loss. No negative net query-model loss was observed.
- Median harmful Jaccard .67116, median loss Spearman .41848: **H2 fails**
  its fixed .60 AND .50 thresholds. Mean loss Spearman .41781. Expected
  all-five harmful count from homogeneous independent marginals is1737.5,
  versus3700 observed; this is a descriptive, not significance, baseline.
- Inversion union174278 pairs: 61.96% one-off, 23.15% two-seed, 1.13%
  all-five. Median pair Jaccard .15879. Only963/3700 persistent queries
  contain an all-five inverted pair. Pair identity and query harm differ.
- Oracle=1 subset7136: robust161 and persistent3283; median relative gap
  .043743 versus .005288, median d12-d9 gap3891 versus887. Spearman of
  d12-d9 with harmful frequency is -.48781; relative boundary margin -.39360.
- Post-primary score average: recall .89577, loss .05738, inversions2.1834;
  42.68% less loss and60.61% fewer inversions than mean single-model values.
- All 45 derived/table/figure files independently reproduce byte-for-byte.
  Original source pools/GT/scores and graph remain unchanged; seed0 matches
  Phase 3C. Full five trained models and all raw scores/pairs are retained.

**Anomalies and possible confounders**

- Robust/persistent mean oracle .810/.987 differs substantially; report both
  all-query and pre-registered oracle=1 geometry. H2 also fails in oracle=1
  (median loss Spearman .36231). Candidate-size and d1 associations change
  after restriction; do not selectively report only favorable descriptors.
- 171 exact-boundary tie queries, 145–176 PQ-boundary ties/model. Excluding
  exact boundary ties leaves median loss Spearman .41728. Query1352 under
  models0/4 has .1 each of unassigned pair-allocation loss because of exact
  tied replacements. They remain in results; no candidate/control anomaly.
- Pair-loss allocation is equal-credit descriptive accounting, not a unique
  causal decomposition. Explicit GT-intruder credits sum -8.5; all net
  query-model losses remain nonnegative. No clipping or forced pair inversion.
- Five initializations share the same training subset, subspace layout,
  database/query realization and graph. Pair and query Jaccard have different
  marginal-support baselines. Top-severity overlaps depend on discrete-loss
  tie breaking; tie-inclusive and independent tie-resolution views are saved.
- Stable aggregate quality coexists with moderate per-query stability and
  substantial codebook-dependent pair changes. This is not strong Case A,
  nor evidence of aggregate instability (Case C); report the mixed result
  without forcing a pure exchangeable-noise Case B or an intrinsic hard class.

**Exactly one next experiment (not run)**

Repeat this five-initialization, fixed-candidate scoring protocol on one
new independent but internally fixed 65,536-vector training subset. Keep
everything else fixed; compare cross-subset harmful-frequency/severity
stability and oracle=1 local-gap associations. This separates vulnerability
that generalizes across training samples from sample-specific shared bias.
No new algorithm, graph change, selective reranking or trajectory logging.

## Phase 3E — balanced training sample x initialization stability

**Registered before new training/results**

Execution base Git `913d7eddbe5f6f6e7b83efa7f2244de1bc2bb246` with dirty
source/config hashes; pinned FAISS `20f14b31a6d54e243a3d1de6ae193fc4c3ec18ed`.
Config: `configs/indexes/phase3e_training_sample_stability.conf`;
pre-registration: `runs/phase3e_training_sample_stability_v1/preregistration.md`.
Same all-query SIFT1M/ef64/k10 exact L0 candidate pools, exact scores/GT,
graph and PQ64x8 architecture. Nine new models, three independent65,536-ID
samples A/B/C with seeds80400011/80400037/80400059, three independent initial
seeds per sample exactly as in config. Original five O models enter separate
secondary aggregate/stability tables only. No new graph traversal.

Pre-registered W_pop+B_pop=T_pop decomposition uses population denominators;
observed B_pop includes finite3-init mean noise. Also record W_sample,
B_sample and signed B_sample-W_sample/3, without random-effects inference.
No H1–H5 threshold is invented after results. Diagnostic compositions fixed
before results: A1/A2/A3 and A1/B1/C1, after primary checkpoint only.

**Preparation/model-generation commands**

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target faiss_sample_stability -j2
/tmp/phase3b-plot-env/bin/python scripts/analyze_phase3e_training_sample_stability.py prepare
ctest --test-dir build --output-on-failure
set -euo pipefail
for model in 0 1 2 3 4 5 6 7 8; do
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=24 ./build/faiss_sample_stability sample-scores configs/indexes/phase3e_training_sample_stability.conf runs/phase3e_training_sample_stability_v1 "$model" 2>&1 | tee "runs/phase3e_training_sample_stability_v1/model_${model}_execution.log"
done
/tmp/phase3b-plot-env/bin/python -m unittest discover -s tests -p 'test_phase3*_metrics.py' -v
```

All35 Python tests (12 new Phase 3E references, 23 previous) and all5 CTests
pass. Centered variance evaluation ensures exactly constant losses have
zero variance without clipping real negative signed contrasts. The existing
filesystem clock-skew warning recurs at build; the separate executable
compiled and linked, preserving historical Phase 3C/3D binaries.

**Primary, checkpoint and diagnostic commands**

```bash
set -o pipefail
/tmp/phase3b-plot-env/bin/python scripts/analyze_phase3e_training_sample_stability.py derive 2>&1 | tee runs/phase3e_training_sample_stability_v1/derive_execution.log
MPLCONFIGDIR=/tmp/phase3b-mpl /tmp/phase3b-plot-env/bin/python scripts/analyze_phase3e_training_sample_stability.py summarize 2>&1 | tee runs/phase3e_training_sample_stability_v1/summary_execution.log
/tmp/phase3b-plot-env/bin/python scripts/analyze_phase3e_training_sample_stability.py checkpoint
MPLCONFIGDIR=/tmp/phase3b-mpl /tmp/phase3b-plot-env/bin/python scripts/analyze_phase3e_training_sample_stability.py ensemble 2>&1 | tee runs/phase3e_training_sample_stability_v1/ensemble_execution.log
```

Independent re-derivation used `mktemp -d /tmp/phase3e-reproduce.XXXXXX`,
which returned `/tmp/phase3e-reproduce.XQaTuu`:

```bash
set -euo pipefail
/tmp/phase3b-plot-env/bin/python scripts/analyze_phase3e_training_sample_stability.py derive --derived /tmp/phase3e-reproduce.XQaTuu/analysis
MPLCONFIGDIR=/tmp/phase3b-mpl /tmp/phase3b-plot-env/bin/python scripts/analyze_phase3e_training_sample_stability.py summarize --derived /tmp/phase3e-reproduce.XQaTuu/analysis --tables /tmp/phase3e-reproduce.XQaTuu/tables --figures /tmp/phase3e-reproduce.XQaTuu/figures
MPLCONFIGDIR=/tmp/phase3b-mpl /tmp/phase3b-plot-env/bin/python scripts/analyze_phase3e_training_sample_stability.py ensemble --derived /tmp/phase3e-reproduce.XQaTuu/ensemble --tables /tmp/phase3e-reproduce.XQaTuu/tables --figures /tmp/phase3e-reproduce.XQaTuu/figures
/tmp/phase3b-plot-env/bin/python scripts/verify_phase3e_artifacts.py runs/phase3e_training_sample_stability_v1 /tmp/phase3e-reproduce.XQaTuu --output /tmp/phase3e-reproduce.XQaTuu/preverification.json
git diff --check
/tmp/phase3b-plot-env/bin/python scripts/verify_phase3e_artifacts.py runs/phase3e_training_sample_stability_v1 /tmp/phase3e-reproduce.XQaTuu --output runs/phase3e_training_sample_stability_v1/final_verification.json
```

**Observations**

- All9 independent PQ models score the same10,353,047 candidates for10,000
  queries. Sample hashes match within groups;9 codebook hashes differ.
  A/B, A/C, B/C ID intersections4206/4288/4356 are near the independent
  draw expectation4294.97; no enforced disjointness or altered preprocessing.
- Nine-model recall mean .8527178, SD .00059947, range .85154–.85364;
  ranking loss mean .1004322, critical inversions5.5486. Recall subset means
  A/B/C=.8529033/.8529100/.8523400. All model quality measures remain raw.
- Within/between harmful Jaccard medians .66836/.66060; loss Spearman
  .41282/.39313; inversion-count Spearman .44764/.41674. Direction supports
  an additional shared-sample component, with a modest query-level drop.
- Mean W_pop .00272459, observed B_pop .00101778, total .00374237:72.80%
  within and27.20% between subset means. The signed noise-floor contrast
  is .00016437 versus W_sample .00408689;27.20% is NOT a pure sample-effect
  estimate.4522 negative contrasts retained;430 zero-variance queries.
- Categories robust/model-specific/usual/universal:
  1103/1953/4503/2441 queries; signed-loss shares1.33/10.17/47.62/40.89%.
  Subset-persistent5676 (56.76%) carry77.59% of total loss. Within oracle=1,
  4851/7136 (67.98%) are subset-persistent and carry82.62% of that loss.
- Pair union243,988:50.63% unique to one model,58.03% confined to one
  subset,41.97% recurring across>=2 subsets,14.21% across all3. Only371
  pairs persist across all9 models; pair Jaccard .15871 within/.14264 between.
- Oracle=1 d12-d9 correlations: harmful frequency−.55815, mean loss−.66865,
  versus average single-model realized-loss correlation−.43060. Local-gap
  association survives conditioning; no predictor or causality claim.
- Post-primary fixed score diagnostics: A1/A2/A3 recall .88588, loss .06727,
  inversions2.8399,32.90% member-average loss recovery. A1/B1/C1 recall
  .89097, loss .06218, inversions2.5363,37.87% recovery. Their respective
  member-average recalls .8529033/.8530733 prevent hiding baseline differences.
- O1..O5 are only secondary: O-within/O-to-ABC Jaccard .67116/.66149 and
  loss Spearman .41848/.39418, consistent with the primary direction.
- All35 Python and5 C++ tests pass. Independent regeneration reproduces61
  files byte-for-byte, including8 SVG figures;174 artifact hashes audited.
  Historical sources, graph and all old runs remain unchanged.

**Anomalies/confounders and interpretation limits**

- There are171 exact-boundary-tied queries and149–187 PQ-boundary ties/model.
  Exact-tie exclusion retains within/between loss Spearman .41237/.39148.
  No negative net query/model loss occurred, but signed losses and all tied
  replacements are preserved. No new exact/candidate control anomaly.
- Population within/between shares depend on averaging only3 initializations
  per sample; they are not formal random-effects components. B_sample minus
  W_sample/3 is a signed descriptive contrast only, not a clipped variance.
  Between-query mean-loss variance likewise is not an identified geometry share.
- Robust means<=2/9 here, not0/5 as in3D;9/9 persistence is stricter than5/5.
  Oracle recall differs substantially across categories; the complete-GT
  stratum and all descriptors/targets are reported, not only favorable ones.
- Pairwise comparisons share models/queries. Sample draws may overlap;
  training subset includes order. Fixed SIFT realization/PQ subspace layout
  and just3 sample draws limit inference. The two diagnostics are fixed
  triplets sharing A1, not performance methods or all-composition estimates.
- Matplotlib3.9.4 warns that boxplot(labels=...) is deprecated; numeric
  outputs and reproducible figures complete. Existing clock-skew warnings
  do not prevent compilation/linking or tests.
- Case C is the conservative supported interpretation: repeatable local-
  geometry susceptibility plus substantial codebook realization noise and
  a smaller sample-linked effect. Case A has descriptive features but causal
  geometry dominance is unproven; Case B's training-sample dominance is not
  supported. No navigability, heavy-tail, novelty or algorithm claim.

**Exactly one next experiment (not run)**

Use one fixed non-learned random orthogonal-coordinate control on the same
candidate pools, ABC sample IDs and init seeds, with standard PQ64x8 scoring
only. Validate preservation of exact L2 geometry/GT numerically, retain the
original exact reference, and compare harmful-frequency persistence/local-gap
associations while reporting PQ approximation quality without tuning. This
separates Euclidean local-margin susceptibility from shared alignment with
PQ subspaces. No optimized rotation, new quantizer or algorithm is proposed.
