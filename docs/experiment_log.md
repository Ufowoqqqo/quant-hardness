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
