# Experiment log

## Phase 3G pre-registered residual-permutation negative control

Pre-registration: `runs/phase3g_residual_permutation_v1/preregistration.md`.
Configuration: `configs/indexes/phase3g_residual_permutation.conf`.
Five fixed R0-I1 through R4-I1 models,50 deterministic shuffles/query/model,
all10,000 queries; reuse exact FP32 pools and stored PQ ADC scores. No search
or training. Hypotheses and signed interpretation fixed before outcomes.

```bash
set -o pipefail
OPENBLAS_NUM_THREADS=1 /tmp/phase3b-plot-env/bin/python -m unittest discover -s tests -p test_phase3g_metrics.py -v 2>&1 | tee runs/phase3g_residual_permutation_v1/tests.log
OPENBLAS_NUM_THREADS=1 /tmp/phase3b-plot-env/bin/python scripts/run_phase3g_residual_permutation.py --config configs/indexes/phase3g_residual_permutation.conf --run runs/phase3g_residual_permutation_v1 2>&1 | tee runs/phase3g_residual_permutation_v1/execution.log
```

Reference tests:7 pass before production. Includes scalar nested-pair/full-sort
agreement, Phase3C metric regression, strict ties, signed loss, reconstruction,
deterministic bijections, multiset preservation and empirical midranks.
Results/anomalies/next experiment will be appended after primary analysis.

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

## Phase 3F — preregistration and invariance gate (2026-09-07)

Execution Git HEAD `676169d68f594e47d40340b20e8c77ac92c1a2f9`, with new
Phase 3F sources recorded by checksum. Preregistration:
`runs/phase3f_rotation_stability_v1/preregistration.md`; configuration:
`configs/indexes/phase3f_rotation_stability.conf`. No data-driven basis
selection or graph traversal. Same Phase 3E A sample IDs and initialization
seeds; identity plus4 independent QR bases; train15 new standard PQ models.

Commands executed:

```bash
/tmp/phase3b-plot-env/bin/python -m unittest discover -s tests -p 'test_phase3f_metrics.py' -v
set -o pipefail
OPENBLAS_NUM_THREADS=1 /tmp/phase3b-plot-env/bin/python scripts/phase3f_rotation_validation.py 2>&1 | tee runs/phase3f_rotation_stability_v1/invariance_execution.log
OPENBLAS_NUM_THREADS=1 /tmp/phase3b-plot-env/bin/python scripts/phase3f_rotation_validation.py 2>&1 | tee runs/phase3f_rotation_stability_v1/invariance_execution_retry1.log
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target faiss_rotation_stability -j2
/tmp/phase3b-plot-env/bin/python -m unittest discover -s tests -p 'test_phase3*_metrics.py' -v 2>&1 | tee runs/phase3f_rotation_stability_v1/python_tests.log
ctest --test-dir build --output-on-failure
```

First validation invocation failed on a Python string/Path mismatch before
preparation or rotation generation; fixed and retained its error log. A
pre-analysis variance unit test exposed tiny spurious positive variance in
constant-within-rotation designs; row-centering fixes arithmetic without
changing the variance definition. All47 Python tests now pass. No PQ results
had been generated during either fix. Build reports known filesystem clock
skew warnings but compiles and links the dedicated executable successfully.

Gate observations so far: R0-R3 preserve all non-tied candidate rankings.
Raw tie-related top10 membership changes are logged, not overwritten. R3 raw
tie order gives mean oracle .95316 versus canonical .95315; original exact
tie policy remains fixed. Numerical tolerances and H1-H5 are unchanged.
Training remains gated on all5 conditions passing.

All5 gates subsequently passed. Maximum orthogonality error1.67e-15;
all10,353,047 candidate distances checked per condition, no strict-reference
ranking reversal. Random-basis candidate max absolute error0.00686–0.00733,
max relative error2.52e-7–3.72e-7. Raw top10 membership tie changes
R1/R2/R3/R4=7/5/6/6; exact reference oracle remains.95315.100 exhaustive
queries per basis pass, including one R4 provided-GT membership tie change.

```bash
set -o pipefail
/tmp/phase3b-plot-env/bin/python scripts/run_phase3f_models.py 2>&1 | tee runs/phase3f_rotation_stability_v1/models_execution.log
```

Runner checks all rotated-input hashes against the completed gate and calls,
for model indices0..14, sequentially:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=24 ./build/faiss_rotation_stability rotation-scores configs/indexes/phase3f_rotation_stability.conf runs/phase3f_rotation_stability_v1 MODEL_INDEX
```

Exact expanded commands appear in `models_execution.log`; each model has its
own `Rr-Ii_execution.log`, resolved config, training IDs, full model/codes,
raw ADC scores and manifest. Model training/scoring results pending here.

All15 training/scoring calls completed successfully. R0-I1/I2/I3 independently
reproduce Phase3E A1/A2/A3 training-vector, codebook, code and complete score
checksums. All models use one common sample-ID checksum, one candidate-request
checksum and one graph fingerprint;15 distinct codebooks. No HNSW traversal
was invoked for rotated data. Every model scores10,000 queries and10,353,047
saved candidates, with scalar/batch4 ADC identity and1001 re-encoded ID checks.

```bash
set -o pipefail
OPENBLAS_NUM_THREADS=1 /tmp/phase3b-plot-env/bin/python scripts/analyze_phase3f_rotation_stability.py derive 2>&1 | tee runs/phase3f_rotation_stability_v1/derive_execution.log
```

## Phase 3F — primary analysis and post-primary diagnostic (2026-09-08)

Commands (same configuration and run throughout):

```bash
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/phase3b-mpl /tmp/phase3b-plot-env/bin/python scripts/analyze_phase3f_rotation_stability.py summarize 2>&1 | tee runs/phase3f_rotation_stability_v1/summary_execution.log
/tmp/phase3b-plot-env/bin/python scripts/analyze_phase3f_rotation_stability.py checkpoint
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/phase3b-mpl /tmp/phase3b-plot-env/bin/python scripts/analyze_phase3f_rotation_stability.py ensemble
mktemp -d /tmp/phase3f-reproduce.XXXXXX
# Returned /tmp/phase3f-reproduce.9znOHW
OPENBLAS_NUM_THREADS=1 /tmp/phase3b-plot-env/bin/python scripts/analyze_phase3f_rotation_stability.py derive --derived /tmp/phase3f-reproduce.9znOHW/analysis
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/phase3b-mpl /tmp/phase3b-plot-env/bin/python scripts/analyze_phase3f_rotation_stability.py summarize --derived /tmp/phase3f-reproduce.9znOHW/analysis --tables /tmp/phase3f-reproduce.9znOHW/tables --figures /tmp/phase3f-reproduce.9znOHW/figures
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/phase3b-mpl /tmp/phase3b-plot-env/bin/python scripts/analyze_phase3f_rotation_stability.py ensemble --derived /tmp/phase3f-reproduce.9znOHW/ensemble --tables /tmp/phase3f-reproduce.9znOHW/tables --figures /tmp/phase3f-reproduce.9znOHW/figures
/tmp/phase3b-plot-env/bin/python -m unittest discover -s tests -p 'test_phase3*_metrics.py' -v 2>&1 | tee runs/phase3f_rotation_stability_v1/python_tests_final.log
ctest --test-dir build --output-on-failure 2>&1 | tee runs/phase3f_rotation_stability_v1/cpp_tests_final.log
rsvg-convert results/figures/phase3f_geometry.svg -o /tmp/phase3f_geometry.png
rsvg-convert results/figures/phase3f_rotation_recall.svg -o /tmp/phase3f_rotation_recall.png
rsvg-convert results/figures/phase3f_ensemble.svg -o /tmp/phase3f_ensemble.png
```

**Observations (separate from interpretation)**

- Mean Recall R0..R4:.852903/.846283/.847147/.846940/.847267; within-basis
  SD .000288–.001141. Random bases modestly worsen MAE (1373.4–1378.8
  vs1312.6), random inversion rate (.03590–.03603 vs.03450) and critical
  inversions (6.0619–6.1945 vs5.5685). No rotation dropped or retuned.
- Median harmful Jaccard .68536 within/.66903 between; loss Spearman
  .41177/.36351; critical-count Spearman .44106/.37991. Identity-to-random
  loss correlation .35682, random-to-random .36858. Matched/unmatched init
  labels across bases yield .36430/.36318. All pair distributions preserved.
- 49.80% are majority-harmful across all5 rotations, accounting for68.91%
  of loss; within oracle=1,61.90% and75.76%, respectively. Independent
  rotation-marginal reference predicts2582.96 persistent queries vs4980.
- Mean population within-init/between-rotation-mean/total variance:
  .00286698/.00143299/.00429996; W+B=T for every query.66.67/33.33% observed
  shares are NOT causal variance components. Signed sample contrast .000357744,
  W_sample .00430047;4160 negative contrasts and186 zero-variance queries retained.
- Oracle=1 d12-d9 correlations: mean loss−.73912, harmful rotation frequency
  −.56766. Mean single-model loss correlation−.44074. Median d12-d9
  robust/occasional/usual/persistent:4056.5/2747/1693/915. Candidate-count
  mean-loss association .02784, relative adjacent gap−.49363, wider d15-d6
  −.65568. Weak/non-supportive descriptors retained alongside strong ones.
- 362,197 union inversion triples:44.55% one-model,49.55% one-rotation,
 50.45% across>=2 rotations,3.58% across all5. Only1 pair persists across15
  models. Pair Jaccard .14487 within/.11702 between, far below query overlap.
- Post-checkpoint five-basis I1 score average:Recall.90574,loss.04741,
  inversions1.5996,54.84% loss recovery from member mean. Initialization
  averages:Recall.88442–.88588,loss.06727–.06873,inversions2.8302–2.9063,
  recovery32.90–36.23%. Full query-level diagnostic data preserved.
- All49 Python and5 C++ tests pass; independent raw-data derivation reproduces
 45 files byte-for-byte, including8 SVGs. Selected figures rendered and checked.

**Anomalies/confounders and limits**

- Raw exact ties are explicitly audited, not forced to match:canonical
  exact oracle.95315 for every model, R3 raw floating tie oracle.95316.
 171 exact boundary ties and132–178 PQ boundary ties/model. Excluding exact
  boundary ties retains within/between loss correlation .40931/.36156.
- No negative net query/model losses occurred; all signed measurements
  were retained. No candidate-order, source, or graph identity anomaly.
- Higher harmful prevalence can mechanically increase Jaccard; independent
  marginal references and oracle=1 results are reported. Pairwise rows
  share queries/models; no independence or significance claim is made.
- Three versus five averaged values, matched seeds and substantial
  interaction prevent a pure rotation/initialization causal variance claim.
 15 versus9 models also confounds comparison of averaged geometry correlation
  with3E. The diagnostics compare5 vs3 models and unequal member quality.
- Four random bases, one SIFT realization and one training subset do not
  establish universal hard-query classes or PQ-specific novelty. Standard
  noisy top-k boundary fragility remains a competing explanation.

**Interpretation and exactly one next experiment**

CASE A is best supported for rotation-averaged susceptibility:high query
overlap, a large majority-persistent population and strong invariant local-gap
association survive changes of basis. This is not a causal variance-dominance
estimate. Realization noise remains, and basis changes modestly reduce query
stability and quality; they do not cause Case B's collapse.

Next (not run):within-query PQ-residual permutation negative control, on
the same saved exact scores/pools. Deterministic permutations preserve each
query/model's full error distribution, signed bias and MAE while disrupting
candidate-specific error alignment. Compare the same susceptibility and
critical-inversion metrics with actual PQ to distinguish generic noisy top-k
selection from PQ-specific structured errors. No algorithm or quantizer change.

Final machine capture and full independent audit commands:

```bash
lscpu | tee runs/phase3f_rotation_stability_v1/machine_cpu.txt
OPENBLAS_NUM_THREADS=1 /tmp/phase3b-plot-env/bin/python scripts/verify_phase3f_artifacts.py runs/phase3f_rotation_stability_v1 /tmp/phase3f-reproduce.9znOHW --output runs/phase3f_rotation_stability_v1/final_verification.json
```

Machine CPU:Intel Core i9-10920X,12 cores/24 hardware threads; the run also
records hostname/kernel/compiler, NumPy1.23.5, Matplotlib3.9.4 and the pinned
FAISS/executable/source hashes. The final audit re-generates all transformed
database/query arrays and QR matrices without deleting or replacing originals.

Final audit result:PASS.45 derived/table/figure files reproduce byte-for-byte;
318 artifact hashes recorded. All QR matrices and complete rotated FP32
database/query arrays independently regenerate identically. All15 models,
identity-model regression against3E and frozen source/candidate/graph inputs
pass validation. Raw data and every previous run remain preserved.

## Phase 3G completed — residual-permutation negative control

Config:`configs/indexes/phase3g_residual_permutation.conf`; pre-registration
`runs/phase3g_residual_permutation_v1/preregistration.md` written before null
outcomes. Execution Git commit5e6334a327e662b119c3b196530a2db327e9eb51;
dirty implementation/source hashes and CPU/kernel/NumPy details are in
`provenance.json`. No graph traversal or PQ training/scoring was performed.
Five existing R0-I1...R4-I1 score files; all10,000 fixed SIFT queries;
50 shuffles/query/model with PCG64 SeedSequence([90700011,r,q,p]).

Exact commands (shell uses `set -o pipefail` for logged pipelines):

```bash
OPENBLAS_NUM_THREADS=1 /tmp/phase3b-plot-env/bin/python -m unittest discover -s tests -p test_phase3g_metrics.py -v 2>&1 | tee runs/phase3g_residual_permutation_v1/tests.log
OPENBLAS_NUM_THREADS=1 /tmp/phase3b-plot-env/bin/python scripts/run_phase3g_residual_permutation.py --config configs/indexes/phase3g_residual_permutation.conf --run runs/phase3g_residual_permutation_v1 2>&1 | tee runs/phase3g_residual_permutation_v1/execution.log
OPENBLAS_NUM_THREADS=1 /tmp/phase3b-plot-env/bin/python -m unittest discover -s tests -p 'test_phase3*_metrics.py' -v 2>&1 | tee runs/phase3g_residual_permutation_v1/regression_tests.log
OPENBLAS_NUM_THREADS=1 /tmp/phase3b-plot-env/bin/python scripts/analyze_phase3g_residual_permutation.py --config configs/indexes/phase3g_residual_permutation.conf --run runs/phase3g_residual_permutation_v1 2>&1 | tee runs/phase3g_residual_permutation_v1/analysis.log
OPENBLAS_NUM_THREADS=1 /tmp/phase3b-plot-env/bin/python scripts/verify_phase3g_artifacts.py --config configs/indexes/phase3g_residual_permutation.conf --run runs/phase3g_residual_permutation_v1 2>&1 | tee runs/phase3g_residual_permutation_v1/verification.log
mktemp -d /tmp/phase3g-reproduce.XXXXXX
OPENBLAS_NUM_THREADS=1 /tmp/phase3b-plot-env/bin/python scripts/analyze_phase3g_residual_permutation.py --config configs/indexes/phase3g_residual_permutation.conf --run runs/phase3g_residual_permutation_v1 --tables /tmp/phase3g-reproduce.AAiOqX/tables --figures /tmp/phase3g-reproduce.AAiOqX/figures 2>&1 | tee runs/phase3g_residual_permutation_v1/reproduction.log
OPENBLAS_NUM_THREADS=1 /tmp/phase3b-plot-env/bin/python scripts/verify_phase3g_artifacts.py --config configs/indexes/phase3g_residual_permutation.conf --run runs/phase3g_residual_permutation_v1 2>&1 | tee runs/phase3g_residual_permutation_v1/verification_retry1.log
OPENBLAS_NUM_THREADS=1 /tmp/phase3b-plot-env/bin/python scripts/audit_phase3g_final.py --config configs/indexes/phase3g_residual_permutation.conf --run runs/phase3g_residual_permutation_v1 --reproduction /tmp/phase3g-reproduce.AAiOqX
```

The first verifier was stopped after discovering repeated NPZ decompression
inside its query loop. Retry reads each model's arrays once; no experimental
metric or raw output changed. Original log and primary table checkpoint are
preserved. Plotting fell back to a writable temporary Matplotlib cache.

Observations:five-model mean observed loss.104982 versus null.14153544;
excess−.03655344. Every model has negative mean excess(.0353–.0376 less loss
than its null). Null mean inversions9.65–10.29 versus observed5.56–6.25.
Under oracle=1(n=7136),five-model mean d12-d9 Spearman is−.6584 with observed
loss,−.7840 with null loss and+.1472 with excess. Observed top10 residual MAE
is957–1063 versus1332–1398 in farther candidates. No positive mean excess
damage is observed. Full report:`docs/phase3g_residual_permutation.md`.

Controls:all50,000 observed rankings reproduce Phase3F; unchanged frozen
inputs,all2.5 million saved top-10s audited,25,000 deterministic null samples
replayed,centered diagnostic gives zero ordered/set differences.56 metric
tests pass. Main derivation570.6s after initialization. All raw data retained.

Anomalies/confounders:744 null losses of−.1 occur only on pre-existing exact
boundary ties;1060 negative null candidate scores remain unclipped. Global
shuffling destroys rank-dependent bias and variance together with individual
assignment; these effects cannot be separately attributed. Null averaging50
shuffles differs from a single observed codebook. Exact descriptors/residuals
are offline oracle quantities,not demonstrated online uncertainty signals.

Exactly one next experiment (NOT run):within-query residual shuffling within
fixed exact-rank bands1–10,11–20,21–50,51–100,101+,same5 models/50 shuffles.
Compare against actual PQ and this global null to isolate coarse
rank-conditioned residual distributions from finer candidate dependence.
No new algorithm is proposed or implemented.

Final audit:PASS.33 table/figure files regenerate byte-for-byte in the
independent temporary output directory. Integer-count aggregate checks and
all744 negative-loss GT boundary-tie substitutions pass. Artifact checksums
are recorded in `runs/phase3g_residual_permutation_v1/final_audit.json`.

## Phase 3H pre-registered hierarchy (before outcomes)

Config:`configs/indexes/phase3h_rank_conditioned_nulls.conf`; preregistration:
`runs/phase3h_rank_conditioned_nulls_v1/preregistration.md`. Five existing
models,all10,000 fixed queries,N0-N4 plus N_distance,100 permutations each.
N0 first50 must reproduce Phase3G exactly. Five model-level workers,one
BLAS thread each. No training or traversal. Primary mechanisms first;
PQ-only observable bridge only after a saved primary checkpoint.

```bash
set -o pipefail
OPENBLAS_NUM_THREADS=1 /tmp/phase3b-plot-env/bin/python -m unittest discover -s tests -p test_phase3h_metrics.py -v 2>&1 | tee runs/phase3h_rank_conditioned_nulls_v1/tests.log
OPENBLAS_NUM_THREADS=1 /tmp/phase3b-plot-env/bin/python scripts/run_phase3h_rank_conditioned_nulls.py --config configs/indexes/phase3h_rank_conditioned_nulls.conf --run runs/phase3h_rank_conditioned_nulls_v1 2>&1 | tee runs/phase3h_rank_conditioned_nulls_v1/execution.log
```

Eight reference tests pass before production. Empty-tail handling,bin
multisets,N0 prefix identity,scalar metric reference,signed closure and
PQ-only feature/quintile semantics are covered. Equal-count distance bins
are rank quartiles,not an independent rank-versus-distance identification.

## Phase 3H completed — final exact-distance mechanism phase and PQ-only bridge

Execution commit92e5c0235fc6ec9eda6fe33bd78d497f42804984; resolved config,
source hashes,input/model/graph checksums and machine information are in
`runs/phase3h_rank_conditioned_nulls_v1/provenance.json`. The100-draw
six-condition,five-model run used972.8s after setup. No training or traversal.
N0 first50 exactly reproduces all Phase3G saved metrics/IDs.64 regression
tests pass. No bin/permutation/model/query settings changed after outcomes.

Further exact commands (logged pipelines use `set -o pipefail`):

```bash
OPENBLAS_NUM_THREADS=1 /tmp/phase3b-plot-env/bin/python -m unittest discover -s tests -p 'test_phase3*_metrics.py' -v 2>&1 | tee runs/phase3h_rank_conditioned_nulls_v1/regression_tests.log
OPENBLAS_NUM_THREADS=1 /tmp/phase3b-plot-env/bin/python scripts/analyze_phase3h_rank_conditioned_nulls.py primary --config configs/indexes/phase3h_rank_conditioned_nulls.conf --run runs/phase3h_rank_conditioned_nulls_v1 2>&1 | tee runs/phase3h_rank_conditioned_nulls_v1/primary_analysis.log
OPENBLAS_NUM_THREADS=1 /tmp/phase3b-plot-env/bin/python scripts/analyze_phase3h_rank_conditioned_nulls.py bridge --config configs/indexes/phase3h_rank_conditioned_nulls.conf --run runs/phase3h_rank_conditioned_nulls_v1 2>&1 | tee runs/phase3h_rank_conditioned_nulls_v1/bridge_analysis.log
OPENBLAS_NUM_THREADS=1 /tmp/phase3b-plot-env/bin/python scripts/verify_phase3h_artifacts.py --config configs/indexes/phase3h_rank_conditioned_nulls.conf --run runs/phase3h_rank_conditioned_nulls_v1 2>&1 | tee runs/phase3h_rank_conditioned_nulls_v1/verification.log
mktemp -d /tmp/phase3h-reproduce.XXXXXX
OPENBLAS_NUM_THREADS=1 /tmp/phase3b-plot-env/bin/python scripts/analyze_phase3h_rank_conditioned_nulls.py primary --config configs/indexes/phase3h_rank_conditioned_nulls.conf --run runs/phase3h_rank_conditioned_nulls_v1 --tables /tmp/phase3h-reproduce.5IFLI7/tables --figures /tmp/phase3h-reproduce.5IFLI7/figures 2>&1 | tee runs/phase3h_rank_conditioned_nulls_v1/reproduction_primary.log
OPENBLAS_NUM_THREADS=1 /tmp/phase3b-plot-env/bin/python scripts/analyze_phase3h_rank_conditioned_nulls.py bridge --config configs/indexes/phase3h_rank_conditioned_nulls.conf --run runs/phase3h_rank_conditioned_nulls_v1 --tables /tmp/phase3h-reproduce.5IFLI7/tables --figures /tmp/phase3h-reproduce.5IFLI7/figures --bridge-output /tmp/phase3h-reproduce.5IFLI7/bridge 2>&1 | tee runs/phase3h_rank_conditioned_nulls_v1/reproduction_bridge.log
OPENBLAS_NUM_THREADS=1 /tmp/phase3b-plot-env/bin/python scripts/audit_phase3h_final.py --config configs/indexes/phase3h_rank_conditioned_nulls.conf --run runs/phase3h_rank_conditioned_nulls_v1 --reproduction /tmp/phase3h-reproduce.5IFLI7
```

Observations,five-model means:observed loss.104982; N0.141570,N1.148221,
N2.114415,N3.106240,N4.106035,N_distance.118176. Descriptive loss-gap
closures−.1818,.7422,.9656,.9712,.6394 for N1...N_distance; not causal shares.
N1 is worse than N0 in four models. N3/N4 gains differ little. Oracle1
d12-d9 correlations with loss/excess are−.7133/+.0144 for N3 and
−.6913/+.0278 for N4. Rank-dependent residual magnitude/variance and signed
bias are present jointly; the experiment does not isolate variance alone.

Post-primary bridge selected g9_12_pq by the pre-registered stable-correlation
rule. Oracle1 Spearman−.4294 to−.4423; all-query−.3683 to−.3778. Across
model-specific quintiles,smallest gap mean loss.15189 versus largest gap
.05909; harmful rates.8705 versus.5168. Feature selection is exploratory,
not held-out validation; inputs are PQ-only but candidate pools remain the
fixed exact-traversal pools. No ready reconstruction-error scalar was present,
so that conditional optional feature was omitted.

Controls:all30million saved outcomes audited,300,000 complete sampled null
replays identical,all50,000 PQ-feature rows verified by scalar reference.
Frozen inputs and pre-bridge checkpoint unchanged.12,270 negative null
losses arise from GT substitutions at exact boundary ties;3,181 negative
null scores remain unclipped. Equal-count distance bins are ordinal rank
quartiles and do not independently identify numeric-distance versus rank
effects. Temporary Matplotlib cache fallback only; no failed scientific run.

Decision:closest Case A,with the N1 counterexample and joint bias/scale
caveat. End exact-distance-only mechanism work. Exactly ONE next experiment
(not run):held-out fixed-budget refinement-feasibility gate using frozen
g9_12_pq,on genuinely unused queries and actual PQ-traversal pools; compare
span-prioritized versus random selection at the same pre-fixed budget,
unrefined recall and an offline oracle bound. No further residual mechanism
experiment or new algorithm is implemented/recommended in this phase.

Final audit:PASS.27 table/figure files and the complete PQ-only observable
file reproduce byte-for-byte in the independent output directory. The
primary checkpoint remains unchanged after the bridge. Raw/script/report
checksums are preserved in `runs/phase3h_rank_conditioned_nulls_v1/final_audit.json`.
# Phase 4A — preregistered held-out refinement gatekeeper

Configuration: `configs/indexes/phase4a_gap_guided_refinement.conf`.
Preregistration: `runs/phase4a_gap_guided_refinement_v1/preregistration.md`.
Frozen observable raw g9_12_pq; L=16,32,64; fractions=.10,.25,.50;
100 random-selection replicates. No outcomes inspected when registered.

Commands (repository root):

```bash
curl -fL --connect-timeout 10 --max-time 120 'https://huggingface.co/datasets/qbo-odp/sift1m/resolve/bd8ccad6c2a0a0a3a7519f6d37c0e5a2d59fe55b/sift_learn.fvecs?download=true' -o runs/phase4a_gap_guided_refinement_v1/sift_learn.fvecs
curl -fL -C - --connect-timeout 10 --max-time 240 'https://huggingface.co/datasets/qbo-odp/sift1m/resolve/bd8ccad6c2a0a0a3a7519f6d37c0e5a2d59fe55b/sift_learn.fvecs?download=true' -o runs/phase4a_gap_guided_refinement_v1/sift_learn.fvecs
curl -fL -C - --retry 3 --retry-all-errors --connect-timeout 15 --max-time 300 'https://huggingface.co/datasets/qbo-odp/sift1m/resolve/bd8ccad6c2a0a0a3a7519f6d37c0e5a2d59fe55b/sift_learn.fvecs?download=true' -o runs/phase4a_gap_guided_refinement_v1/sift_learn.fvecs
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release > runs/phase4a_gap_guided_refinement_v1/configure.log 2>&1
cmake --build build --target faiss_phase4a -j 8 > runs/phase4a_gap_guided_refinement_v1/build.log 2>&1
cmake --build build --target faiss_phase4a -j 8 > runs/phase4a_gap_guided_refinement_v1/build_retry1.log 2>&1
OPENBLAS_NUM_THREADS=1 /tmp/phase3b-plot-env/bin/python -m unittest discover -s tests -p test_phase4a_metrics.py -v > runs/phase4a_gap_guided_refinement_v1/tests.log 2>&1
ctest --test-dir build --output-on-failure > runs/phase4a_gap_guided_refinement_v1/ctest.log 2>&1
OPENBLAS_NUM_THREADS=1 /tmp/phase3b-plot-env/bin/python scripts/prepare_phase4a.py > runs/phase4a_gap_guided_refinement_v1/prepare.log 2>&1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=24 ./build/faiss_phase4a configs/indexes/phase4a_gap_guided_refinement.conf > runs/phase4a_gap_guided_refinement_v1/execution.log 2>&1
```

Preparation observation: 10,017 learning records overlap old-query contents;
195 additional duplicate learning records removed; no base-content overlap;
89,788 eligible. Fixed split: 65,536 training + 10,000 never-before-used queries.
No scientific outcomes yet. Download resumed after timeout/TLS connection
failure; complete SHA256 matches pinned source. First build failed at FAISS
MaybeOwnedVector equality syntax; fixed before any measurement. All six new
tests and five C++ tests pass. Next in-protocol check: independent exhaustive
GT and candidate-score validation before policy comparisons.

### Phase 4A completed measurement, validation and frozen-policy analysis

The pre-registered experiment finished without model or graph changes. The
existing reference BLAS made full 10k × 1M ground truth the long stage; no
query reduction, distance backend substitution or timing benchmark was made.

```bash
OPENBLAS_NUM_THREADS=1 /tmp/phase3b-plot-env/bin/python scripts/phase4a_gt_reference.py > runs/phase4a_gap_guided_refinement_v1/gt_reference.log 2>&1
OPENBLAS_NUM_THREADS=1 /tmp/phase3b-plot-env/bin/python -m unittest discover -s tests -p 'test_phase*_metrics.py' -v > runs/phase4a_gap_guided_refinement_v1/regression_tests.log 2>&1
OPENBLAS_NUM_THREADS=1 /tmp/phase3b-plot-env/bin/python scripts/analyze_phase4a.py validate > runs/phase4a_gap_guided_refinement_v1/validation.log 2>&1
OPENBLAS_NUM_THREADS=1 /tmp/phase3b-plot-env/bin/python scripts/analyze_phase4a.py analyze > runs/phase4a_gap_guided_refinement_v1/analysis.log 2>&1
OPENBLAS_NUM_THREADS=1 /tmp/phase3b-plot-env/bin/python scripts/phase4a_tie_sensitivity.py > runs/phase4a_gap_guided_refinement_v1/tie_sensitivity.log 2>&1
OPENBLAS_NUM_THREADS=1 /tmp/phase3b-plot-env/bin/python -m unittest discover -s tests -p 'test_phase*_metrics.py' -v > runs/phase4a_gap_guided_refinement_v1/regression_tests_final.log 2>&1
mktemp -d /tmp/phase4a-reproduce.XXXXXX
# Returned /tmp/phase4a-reproduce.3YxCOu
OPENBLAS_NUM_THREADS=1 /tmp/phase3b-plot-env/bin/python scripts/analyze_phase4a.py analyze --tables /tmp/phase4a-reproduce.3YxCOu/tables --figures /tmp/phase4a-reproduce.3YxCOu/figures --derived /tmp/phase4a-reproduce.3YxCOu/analysis > runs/phase4a_gap_guided_refinement_v1/reproduction.log 2>&1
OPENBLAS_NUM_THREADS=1 /tmp/phase3b-plot-env/bin/python scripts/phase4a_tie_sensitivity.py --tables /tmp/phase4a-reproduce.3YxCOu/tables > runs/phase4a_gap_guided_refinement_v1/reproduction_tie.log 2>&1
OPENBLAS_NUM_THREADS=1 /tmp/phase3b-plot-env/bin/python scripts/audit_phase4a.py --reproduction /tmp/phase4a-reproduce.3YxCOu > runs/phase4a_gap_guided_refinement_v1/audit.log 2>&1
```

Observations (10,000 held-out queries): PQ native .850560; full PQ candidate
oracle .951390; exact native .952760; exact L0 oracle .952320. Candidate
recoverable gap .100830; mean signed discovery delta .000930. Raw gap/loss
Spearman -.357433. Quintile loss declines .146400 to .056200; harmful fraction
.8470 to .4925. GAP exceeds RANDOM empirical 97.5th percentile at 9/9 aggregate
and 36/36 shard budgets. All L pass both required 25%/50% comparisons.
At L16/25%: cost4, GAP recall .881250 vs random .873171. At L16/50%: cost8,
GAP .906200 vs random .895774. Depth-specific selector efficiency across all
points is 29.46–42.22%. Full tables preserve every point and all 4,500 random
allocation outcomes. Pre-registered useful criterion passes; the qualitative
strong-gatekeeper description is supported, not a significance claim.

Counterevidence/limits: top16-all recall .941020 at cost16 exceeds GAP
L32/50% .913540 at the same cost. Uniform shallow refinement is a necessary
practical baseline. Top32-all .951290 nearly equals full pool .951390; L64
adds little. One model/index and query shards do not establish independent
codebook or dataset replication. Budget selection is workload-level, not
yet a deployed streaming cutoff. No timing superiority is claimed.

Correctness: all 10,361,485 stored candidate FP32 distances equal direct
FP64 distances; 100 deterministic full-base GT comparisons pass; whole
base/graph-storage and PQ code ID alignment pass; graph fingerprint and
native instrumented results unchanged. Minimum candidate count236, no short
pools. 686 native/global-PQ ordered-list differences are score ties.
44 exact-control recall anomalies are all -.1 at GT rank10 ties (40 pairs
identical vectors, four distinct equidistant pairs), mean -.000440.
Six negative ranking-loss cases are retained and verified boundary ties.
Primary ID metrics remain unchanged. Separate post-primary tie-aware
sensitivity has zero exact-control discrepancies, native/oracle recalls
.850650/.951980 and still passes at all L. Matplotlib used a temporary
cache because its default home cache is not writable; SVG output succeeded.

Final audit PASS: 10,000 scalar query-reference checks, 135 nonrandom policy
points, all 4,500 random budgets, exact regeneration of query rows/selection
arrays and 22 byte-identical table/figure files. 71 Python and five C++ tests
pass. Prior raw runs are untouched. Full report:
`docs/phase4a_gap_guided_refinement.md`.

Exactly one next experiment: frozen L16/raw-gap 25th-percentile streaming
cutoff on the 14,252 remaining unused eligible learning vectors, same PQ and
graph, against equal-realized-count random refinement and a clearly labeled
higher-cost uniform top16 reference; measure recall, realized exact costs
and end-to-end latency/selector overhead without threshold retuning.

### Phase 4A fresh-distance execution control and final audit

```bash
OPENBLAS_NUM_THREADS=1 /tmp/phase3b-plot-env/bin/python scripts/verify_phase4a_refinement_execution.py > runs/phase4a_gap_guided_refinement_v1/physical_refinement.log 2>&1
OPENBLAS_NUM_THREADS=1 /tmp/phase3b-plot-env/bin/python scripts/audit_phase4a.py --reproduction /tmp/phase4a-reproduce.3YxCOu > runs/phase4a_gap_guided_refinement_v1/audit_final.log 2>&1
```

Fresh direct FP32 refinement reads no candidate-exact cache: all nine budgets
for GAP/ORACLE/RANDOM-replicate0 yield matching result IDs, unchanged unselected
native results, and exactly L distances per selected query. 27 controls,
2,856,000 counted FP32 distances, PASS. This validates executable refinement
semantics and cost counting in addition to offline replay; not a speed test.
Final audit refreshed to include this additional raw validation/source.

## Phase 4B — frozen streaming systems gatekeeper (2026-09-08)

Execution commit `31e7412687522be5366c75a0eeaba7fcab80fe22` plus recorded
source/binary hashes; FAISS `20f14b31a6d54e243a3d1de6ae193fc4c3ec18ed`.
Configuration: `configs/indexes/phase4b_streaming_feasibility.conf`.
Preregistration and threshold provenance were recorded before outcomes under
`runs/phase4b_streaming_feasibility_v1/`. Same graph/model as Phase4A, no
retraining or graph rebuild. Final14252 unused eligible learning vectors,
ef64/k10/PQ64x8/L16, inclusive frozen threshold710.40625, not a test percentile.
Seven repeats,512 separate old-query warmups,CPU2/single thread,30 random masks,
batch32 secondary. Policy-order/random/GT/batch-order seeds respectively
94100011/94100037/94100059/94100083. All vector IDs/checksums preserved.

Commands executed (all relative to repository root; build revisions completed
before timing, with no outcome-driven performance changes):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release > runs/phase4b_streaming_feasibility_v1/configure.log 2>&1
cmake --build build --target faiss_phase4b -j 8 > runs/phase4b_streaming_feasibility_v1/build.log 2>&1
OPENBLAS_NUM_THREADS=1 /tmp/phase3b-plot-env/bin/python scripts/prepare_phase4b.py > runs/phase4b_streaming_feasibility_v1/prepare.log 2>&1
cmake --build build --target faiss_phase4b -j 8 > runs/phase4b_streaming_feasibility_v1/build_final.log 2>&1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 taskset -c 2 ./build/faiss_phase4b reference configs/indexes/phase4b_streaming_feasibility.conf > runs/phase4b_streaming_feasibility_v1/reference.log 2>&1
OPENBLAS_NUM_THREADS=1 /tmp/phase3b-plot-env/bin/python -m unittest discover -s tests -p test_phase4b_metrics.py -v > runs/phase4b_streaming_feasibility_v1/tests.log 2>&1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=24 ./build/faiss_phase4b gt configs/indexes/phase4b_streaming_feasibility.conf > runs/phase4b_streaming_feasibility_v1/gt.log 2>&1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 taskset -c 2 ./build/faiss_phase4b export configs/indexes/phase4b_streaming_feasibility.conf > runs/phase4b_streaming_feasibility_v1/export.log 2>&1
cmake --build build --target faiss_phase4b -j 8 > runs/phase4b_streaming_feasibility_v1/build_stream_timer.log 2>&1
OPENBLAS_NUM_THREADS=1 /tmp/phase3b-plot-env/bin/python scripts/validate_phase4b.py > runs/phase4b_streaming_feasibility_v1/validation.log 2>&1
ctest --test-dir build --output-on-failure > runs/phase4b_streaming_feasibility_v1/ctest.log 2>&1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /tmp/phase3b-plot-env/bin/python scripts/run_phase4b_benchmarks.py > runs/phase4b_streaming_feasibility_v1/runner.log 2>&1
# Runner launches these sequentially, with environment/process snapshots:
# taskset -c 2 ./build/faiss_phase4b benchmark configs/indexes/phase4b_streaming_feasibility.conf
# taskset -c 2 ./build/faiss_phase4b batch configs/indexes/phase4b_streaming_feasibility.conf
OPENBLAS_NUM_THREADS=1 /tmp/phase3b-plot-env/bin/python scripts/analyze_phase4b.py > runs/phase4b_streaming_feasibility_v1/analysis.log 2>&1
OPENBLAS_NUM_THREADS=1 /tmp/phase3b-plot-env/bin/python -m unittest discover -s tests -p 'test_phase*_metrics.py' -v > runs/phase4b_streaming_feasibility_v1/regression_tests.log 2>&1
mktemp -d /tmp/phase4b-reproduce.XXXXXX
# Returned /tmp/phase4b-reproduce.nGOpEJ
OPENBLAS_NUM_THREADS=1 /tmp/phase3b-plot-env/bin/python scripts/analyze_phase4b.py --tables /tmp/phase4b-reproduce.nGOpEJ/tables --figures /tmp/phase4b-reproduce.nGOpEJ/figures --derived /tmp/phase4b-reproduce.nGOpEJ/analysis > runs/phase4b_streaming_feasibility_v1/reproduction.log 2>&1
OPENBLAS_NUM_THREADS=1 /tmp/phase3b-plot-env/bin/python scripts/audit_phase4b.py --reproduction /tmp/phase4b-reproduce.nGOpEJ > runs/phase4b_streaming_feasibility_v1/audit.log 2>&1
```

Machine: i9-10920X,12 physical/24 logical cores,32,273,192 KiB RAM;
GCC11.5.0; app `-O3 -DNDEBUG -std=c++20 -Wall -Wextra -Wpedantic -fopenmp`,
FAISS generic CPU `-O3 -DNDEBUG -std=gnu++20 -fPIC -fopenmp`, FINTEGER=int.
One OpenMP/OpenBLAS thread for timing. Performance governor/turbo enabled,
unchanged. Frequency snapshots are idle readings, not active frequency locks.
Read-only host process inspection (`ps -eo pid,comm,pcpu --sort=-pcpu`)
found no competing heavy job. No unrelated processes or host settings changed.
Shared-machine limitations, raw snapshots and complete environment are retained.

Observations: selected3556/14252 (24.950884%), exact56896 (3.992141/query),
threshold drift−0.049116 percentage points. Native/GAP/RANDOM-mask0/ALL16
recall .850309/.880501/.872523/.939973; offline full candidate oracle .950365.
RANDOM30 mean .872784, central95% allocation interval [.872184,.873504].
GAP gain over randommean .007717; gap/loss Spearman−.356737. Selected outcomes:
2823 improve,733 neutral,0 harmful. Refined/unrefined mean loss .139370/.086986.

Primary mean microseconds native/GAP/random/all16:
161.650/182.427/171.107/184.651; p95 189.410/215.582/217.673/218.145.
GAP exact-saving75.0491%, latency-saving1.2042%, ALL16-gain-retained33.6724%.
Frozen20%/60% targets both fail. Formally GAP is not dominated by these discrete
points, but its latency saving is small. Clocks-ON components show ALL16
exact-stage2.166 microseconds; GAP extraction/branch3.056, with retention also
inside search. This is not a pure subtraction/branch overhead estimate.
Batch32 scalar/indexed whole-path amortized182.37694/182.37550 microseconds:
no meaningful acceleration. Batch completion (~6ms) is distinct from per-query
streaming service latency.

Anomalies/confounders: 271 GT boundary-tie queries; separate tie-aware metrics
preserve the conclusion and show no selected harmful refinement. Phase4A's
inclusive threshold tie would select2501 rather than its quota2500, explicitly
recorded before evaluation. Offline GT uses FAISS direct FP32 exhaustive
kernel rather than Phase4A's BLAS route;100 FP64 full-base validations pass.
No new FP32 traversal was run, so no newly measured discoverydelta is claimed.
Single resident-index machine, fixed query order, turbo/cache variation and
integration-specific retention overhead limit deployment generality. Small
negative ON−OFF timing differences reflect measurement drift, not negative
timestamp overhead. Build NFS clock-skew and matplotlib cache-fallback warnings
did not prevent successful output; no measured repetition was discarded.

76 Python and5 C++ tests pass. All14252 public-native IDs, top16 full-sort
references and policy results match; all14,744,891 candidate distances match
independent FP64. Original graph/model/input hashes remain fixed. Timed
correctness checks occur after each complete stream. Prior runs preserved.
Full report: `docs/phase4b_streaming_feasibility.md`; per-query raw/timing data,
22 tables and7 figures can be regenerated without new graph searches/training.

Interpretation: held-out query risk generalizes but useful systems value is
not demonstrated; reducing cheap exact distance work barely changes total
latency. Do not proceed to learned query-risk modeling on this evidence.

Exactly one next experiment: fixed-output ALL-L16 full-materialization versus
bounded top16 candidate-retention ablation, with identical graph/model/L/ties
and verified result IDs before timing. Measure end-to-end latency and scratch
memory to isolate candidate accessibility as the next systems bottleneck.

Final reproduction audit PASS:798112 timed query traces,57008 policy/query
recall checks,30 exactly regenerated random masks, identical derived query
rows and29 byte-identical tables/figures. Artifact hashes are preserved in
`runs/phase4b_streaming_feasibility_v1/final_audit.json`.

## Phase 4C — bounded top16 retention (2026-09-08)

Frozen design recorded before timings at
`runs/phase4c_bounded_candidate_retention_v1/preregistration.md`;
configuration `configs/indexes/phase4c_bounded_candidate_retention.conf`.
Git execution revision is the Phase4B commit `a9591c9` (full hash in provenance),
plus immutable source/binary hashes. Same FAISS revision
`20f14b31a6d54e243a3d1de6ae193fc4c3ec18ed`, graph/PQ model/14252 queries and
512 old warmups. No data transformation, training, GT recomputation or altered
search parameters. ef64/k10/L16, frozen inclusive gap710.40625. Seven repeats,
primary-order seed94200011, secondary-order seed94200037. Reuse of Phase4B
queries is an implementation comparison, not a new generalization test.

Commands executed:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release > runs/phase4c_bounded_candidate_retention_v1/configure.log 2>&1
# This first build was launched before asynchronous configure finished:
cmake --build build --target faiss_phase4c bounded_retention_correctness -j 8 > runs/phase4c_bounded_candidate_retention_v1/build.log 2>&1
# No target existed yet; after successful configure, rerun:
cmake --build build --target faiss_phase4c bounded_retention_correctness -j 8 > runs/phase4c_bounded_candidate_retention_v1/build_after_configure.log 2>&1
./build/bounded_retention_correctness > runs/phase4c_bounded_candidate_retention_v1/bounded_test.log 2>&1
OPENBLAS_NUM_THREADS=1 /tmp/phase3b-plot-env/bin/python scripts/prepare_phase4c.py > runs/phase4c_bounded_candidate_retention_v1/prepare.log 2>&1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 taskset -c 2 ./build/faiss_phase4c validate configs/indexes/phase4c_bounded_candidate_retention.conf > runs/phase4c_bounded_candidate_retention_v1/validation.log 2>&1
OPENBLAS_NUM_THREADS=1 /tmp/phase3b-plot-env/bin/python -m unittest discover -s tests -p 'test_phase*_metrics.py' -v > runs/phase4c_bounded_candidate_retention_v1/python_tests.log 2>&1
ctest --test-dir build --output-on-failure > runs/phase4c_bounded_candidate_retention_v1/ctest.log 2>&1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /tmp/phase3b-plot-env/bin/python scripts/run_phase4c_benchmarks.py > runs/phase4c_bounded_candidate_retention_v1/runner.log 2>&1
# Runner executes sequentially (never alongside validation/analysis/build):
# taskset -c 2 ./build/faiss_phase4c primary configs/indexes/phase4c_bounded_candidate_retention.conf
# taskset -c 2 ./build/faiss_phase4c diagnostic configs/indexes/phase4c_bounded_candidate_retention.conf
# taskset -c 2 ./build/faiss_phase4c detail configs/indexes/phase4c_bounded_candidate_retention.conf
```

Before timing, host read-only `ps -eo pid,comm,pcpu --sort=-pcpu | head -12`
showed no heavy competing job. No host settings or unrelated processes changed.
Same i9-10920X/CPU2/one OMP+OpenBLAS thread as4B; GCC11.5.0, Release generic
CPU FAISS, `-O3 -DNDEBUG -std=c++20 -Wall -Wextra -Wpedantic -fopenmp` app flags.
Exact compiler/FAISS flags, memory, CPU, governor/turbo snapshots and process
state are archived in provenance/environment JSON. Shared machine, not exclusive
isolation. Build clock-skew warnings from NFS retained; both targets built.

Pre-benchmark equivalence PASS:256536 mode/clock/query checks across14252
queries; legacy full pool order/scores, native bits, ordered top16 bits,
ALL16 exact distances/outputs and frozen GAP results identical.210 rank16
boundary-tie queries require no equivalence relaxation. Fixed heap208 bytes;
full scratch4024576 bytes, no vector-capacity growth.200 randomized/tied/
duplicate stream references pass;81 Python and6 C++ tests pass. Original4B
implementation and raw runs are unchanged.

Post-benchmark analysis and independent regeneration commands:

```bash
OPENBLAS_NUM_THREADS=1 /tmp/phase3b-plot-env/bin/python scripts/analyze_phase4c.py > runs/phase4c_bounded_candidate_retention_v1/analysis.log 2>&1
mktemp -d /tmp/phase4c-reproduce.XXXXXX
# Returned /tmp/phase4c-reproduce.JbBbYg
OPENBLAS_NUM_THREADS=1 /tmp/phase3b-plot-env/bin/python scripts/analyze_phase4c.py --tables /tmp/phase4c-reproduce.JbBbYg/tables --figures /tmp/phase4c-reproduce.JbBbYg/figures --derived /tmp/phase4c-reproduce.JbBbYg/analysis > runs/phase4c_bounded_candidate_retention_v1/reproduction.log 2>&1
OPENBLAS_NUM_THREADS=1 /tmp/phase3b-plot-env/bin/python scripts/audit_phase4c.py --reproduction /tmp/phase4c-reproduce.JbBbYg > runs/phase4c_bounded_candidate_retention_v1/audit.log 2>&1
```

Observations (primary clocksOFF): NATIVE/FULL/BOUNDED/FULL_ALL16/BOUNDED_ALL16
mean microseconds162.831814/184.732080/169.108819/186.707853/172.358667.
Corresponding p95:190.8707/218.78995/198.53685/220.047/203.59665;
p99:205.50137/235.28485/213.11711/236.49237/219.40833. Native/access-only
recall .85030873; both ALL16 recall .93997334, IDs exactly identical.
Full-overhead21.900266us, bounded-overhead6.277005us: reduction71.338225%,
passes <=50% residual-overhead criterion. End-to-end ALL16 saving7.685368%
(speedup1.083252x), below the alternative10% threshold. The preregistered OR
optimization gate passes. Per-repeat savings and complete latency distributions
are retained, not only averages.

Stage-clock observation: full extraction3.152us, bounded finalize0.245us;
exact16 stage2.122/2.093us for full/bounded. Intrusive capture-clock readings
23.593/8.977us raise total to200.048/181.130us, versus stage-only184.965/168.944.
Do not treat those timer-perturbed readings as exclusive uninstrumented work.
No exact cache/bandwidth attribution. OFF/stageON confirms the optimization
without per-callback clocks; maximum primary ON−OFF magnitude0.504%.

Secondary contemporaneous NATIVE/GAP_BOUNDED/BOUNDED_ALL16 mean
162.828867/169.907621/172.555502us. GAP remains recall .88050098 and selects
3556 (24.950884%) with exactly56896 distances. Its saving vsALL16 is1.534510%,
and ALL16-gain retention33.672431%; both selective-value gates fail. ALL16
costs only2.648us more than GAP for .05947235 more recall. All three discrete
points are formally nondominated; practical value is a separate judgment.

Memory observations: pool mean1034.584/p951337, range216–1920. Full live
ID/score/order bytes mean12415 plus4MB tags (reserved total4024576/worker).
Bounded heap208 bytes plus192-byte sorted copy and common outputs, no extra
tags/full arrays. Both collectors allocate0 dynamically/query after full
reserve on this stream; shared FAISS/DC allocations remain. This is source/
capacity accounting, not RSS/malloc profiling. Identical16-row FP32 payload
8192 bytes/query for both ALL16 paths; FP32 side database512MB still required.

Confounders: warm resident index, fixed query order, one shared nonexclusive
machine/model; intentional old-query reuse is not a new generalization test.
Active CPU frequency is not inferred from idle frequency snapshots. Runner ps
snapshots are namespace-local; separate host-visible precheck is preserved in
host_check_note.md. Callback clocks alter timing substantially; their subtracted
search remainder is not a pure-core counterfactual. GT metric/tie conventions
are unchanged because outputs/inputs are unchanged. No performance-driven
implementation revisions or discarded repetitions. Full report:
`docs/phase4c_bounded_candidate_retention.md`.

Interpretation: Case A is supported for candidate-access overhead, with modest
whole-query benefit; not Case B. Uniform bounded ALL16 is within approximately6%
of native latency for8.97 recall percentage points more. Do not invest further
in query-level risk prediction on this evidence. Native remains the latency
endpoint; bounded ALL16 the strong high-recall reference.

Exactly one next experiment: NATIVE versus BOUNDED_ALL16 at1/2/4/8 pinned,
independent single-thread query workers, same frozen inputs/results. Measure
QPS and p95/p99 to test whether uniform refinement's small cost survives
concurrent shared-memory/cache pressure before deployment conclusions.

Final audit PASS:1,624,728 timed output/count checks,14252 independent
stable-full-sort top16 references, unchanged immutable input/source/binary
hashes, identical derived query rows,21 byte-identical tables/figures.
Artifact manifest: `runs/phase4c_bounded_candidate_retention_v1/final_audit.json`.

## Phase 4D — concurrent bounded ALL-L16 (2026-09-08)

Execution Git `9a0c375c9151ec45bf7866c8d5b14e23198b48f6`, plus immutable
new-harness/build hashes. FAISS `20f14b31a6d54e243a3d1de6ae193fc4c3ec18ed`.
Preregistered in `runs/phase4d_multithread_scalability_v1/preregistration.md`
before concurrent outcomes. Config `configs/indexes/phase4d_multithread_scalability.conf`.
Same engine source, graph/PQ model/SIFT query IDs/GT, ef64/k10/L16; no GAP,
training, data changes or query-risk work. Threads1/2/4/8 on physical CPUs2–9,
coordinator0, internal OMP/BLAS1. Four complete14252-query passes/cell, five
repetitions, shuffled policy/worker order seed94300011. Stress seed94300037;
512 stress queries include210 previously known PQ cutoff ties;4096 calls/worker.

Exact commands executed:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release > runs/phase4d_multithread_scalability_v1/configure.log 2>&1
cmake --build build --target faiss_phase4d -j 8 > runs/phase4d_multithread_scalability_v1/build.log 2>&1
OPENBLAS_NUM_THREADS=1 /tmp/phase3b-plot-env/bin/python scripts/prepare_phase4d.py > runs/phase4d_multithread_scalability_v1/prepare.log 2>&1
ctest --test-dir build --output-on-failure > runs/phase4d_multithread_scalability_v1/ctest.log 2>&1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_DYNAMIC=FALSE OMP_MAX_ACTIVE_LEVELS=1 taskset -c 0,2-9 ./build/faiss_phase4d stress configs/indexes/phase4d_multithread_scalability.conf > runs/phase4d_multithread_scalability_v1/stress.log 2>&1
OPENBLAS_NUM_THREADS=1 /tmp/phase3b-plot-env/bin/python -m unittest discover -s tests -p 'test_phase*_metrics.py' -v > runs/phase4d_multithread_scalability_v1/python_tests.log 2>&1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_DYNAMIC=FALSE OMP_MAX_ACTIVE_LEVELS=1 taskset -c 0 /tmp/phase3b-plot-env/bin/python scripts/run_phase4d_benchmarks.py > runs/phase4d_multithread_scalability_v1/runner.log 2>&1
# Runner primary:
# taskset -c 0,2-9 ./build/faiss_phase4d benchmark configs/indexes/phase4d_multithread_scalability.conf
# After primary only, perf availability probe:
# perf stat --all-user -e task-clock,cycles,instructions,cache-references,cache-misses -- true
# Gated perf commands (actual FIFO paths recorded in benchmark_environment.json):
# perf stat --all-user -D -1 --control fifo:CTL,ACK -x , -e task-clock,cycles,instructions,cache-references,cache-misses -o RAW_COUNTER_FILE -- taskset -c 0,2-9 ./build/faiss_phase4d profile configs/indexes/phase4d_multithread_scalability.conf T POLICY CTL ACK
```

Source audit: PQ ADC tables are per-call; visited tables are thread_local;
each worker owns CandidateAccess; low-level HNSW APIs avoid shared global
high-level statistics accumulation. Ready gate verifies T+1 live threads,
distinct TLS addresses and OpenMP max_threads1. Warmup/start barrier excluded
from service timing; per-query actual API latency retained, QPS uses total
completion window. Post-stream comparisons/file writes excluded. Every repeat
retained; worker-end offsets quantify finite-stream drain/imbalance.

Machine: Intel i9-10920X,12 physical/24 logical cores, one NUMA node. Workers
2–9 are distinct cores, not siblings14–21. L1d/L1i32KiB per core,L2 1MiB per
core, shared L3 19712KiB; RAM32273192KiB. GCC11.5.0, same Release app flags
`-O3 -DNDEBUG -std=c++20 -Wall -Wextra -Wpedantic -fopenmp`, generic CPU FAISS.
Governor performance, turbo enabled and unchanged. Source/binary/config/model/
input/cache/topology checksums and flags are in provenance; per-cell load,
CPU/memory/frequency snapshots are raw *_before.txt/*_after.txt. Idle frequency
snapshots are not locked active frequencies. Host precheck found no competing
heavy job; namespace-local process snapshots are not continuous host monitoring.
No agent-owned heavy jobs overlap timing. No exclusive-machine claim.

Pre-benchmark correctness PASS:61440 mixed concurrent calls over1/2/4/8
workers, native and bounded-refinement output IDs/top16 bits equal frozen
references. No observed state leakage; model and graph memory hashes unchanged.
Timestamp pair-loop upper-bound0.03590133us, not subtracted. Six C++ and86 Python
tests pass. NFS build clock-skew warnings retained; executable built successfully.

Analysis/reproduction commands after all primary and optional profiles:

```bash
OPENBLAS_NUM_THREADS=1 /tmp/phase3b-plot-env/bin/python scripts/analyze_phase4d.py > runs/phase4d_multithread_scalability_v1/analysis.log 2>&1
mktemp -d /tmp/phase4d-reproduce.XXXXXX
# Returned /tmp/phase4d-reproduce.p7bkzL
OPENBLAS_NUM_THREADS=1 /tmp/phase3b-plot-env/bin/python scripts/analyze_phase4d.py --tables /tmp/phase4d-reproduce.p7bkzL/tables --figures /tmp/phase4d-reproduce.p7bkzL/figures > runs/phase4d_multithread_scalability_v1/reproduction.log 2>&1
OPENBLAS_NUM_THREADS=1 /tmp/phase3b-plot-env/bin/python scripts/audit_phase4d.py --reproduction /tmp/phase4d-reproduce.p7bkzL > runs/phase4d_multithread_scalability_v1/audit.log 2>&1
```

Observations, five-repeat means for1/2/4/8 workers:

- NATIVE QPS:6118.50/11892.92/23076.26/45165.50.
- ALL16 QPS:5687.78/10988.41/21218.67/41682.04.
- NATIVE mean latency us:163.312/167.839/171.779/175.208.
- ALL16 mean latency us:175.698/181.497/186.388/189.198.
- NATIVE p95 us:191.491/196.074/200.470/203.366;
  p99:205.558/209.776/214.253/216.942.
- ALL16 p95 us:207.768/214.710/220.449/222.563;
  p99:223.714/230.820/236.823/240.508.
- Throughput penalty7.040/7.605/8.050/7.713%; mean overhead
  7.584/8.138/8.505/7.984%; p99 ratios1.0883/1.1003/1.1053/1.1086.

All<=15% throughput criteria pass.8-worker speedup7.382 native /7.328 ALL16;
efficiency92.272/91.604%. Recall remains .85030873/.93997334 (all ID/counts
identical),8.966 percentage-point gain. ALL16 retains92.287% of native QPS
at8 workers. Every primary cell's mean/median/p90/p95/p99/QPS/recall and the
mean/SD/min/max across repeats are in phase4d_repetitions/aggregate tables.
40 primary cells,2,280,320 individual measurements,285040 per policy/T.

Tail/CPU observations:8-worker paired p99 overhead ranges9.66–12.03%; maximum
raw latency563.510us, not removed. Assigned-core utilization98.79/98.44% at8;
last-worker drain fraction1.56/2.45%. Largest launch lag14.91us/cell. No extra
live query threads, TLS aliasing, core migration or changed index hashes.

Optional perf profiles all succeeded after primary, gates acknowledged,100%
event-running time. Instructions/query1.137M native/1.185M ALL16 at both1/8;
cache misses/query3174/3442 at1 and3429/3688 at8. Additional instructions
approximately4.15%; misses approximately8.45% then7.56% above native. These
are generic user-space cache events, not DRAM bandwidth or isolated FP32
refinement traffic. Task-clock counts agree closely with worker CPU time;
perf's default CPUs-utilized denominator includes disabled process phases and
must not replace the hot-loop utilization estimate. Nominal additional FP32
payload8192 bytes/query,0.3415GB/s at measured8-worker ALL16 QPS; not actual
bandwidth. No unsupported hardware-causation claim.

Confounders: fresh1-worker overhead7.58% differs from4C5.85% despite unchanged
engine/inputs/results. Different longer-stream/worker allocator/cache/frequency
context may matter, but no cause established; startup itself was excluded.
All scaling uses the new contemporaneous baseline. Closed-loop service times
exclude arrival queues/network delays; repeated queries and one warm resident
single-NUMA machine do not establish production SLAs or external validity.
Finite shard drain is included in QPS, not corrected away. Counter samples
are diagnostic single runs, not primary repeats. No tuning, selective risk
policy, retraining or run removal.

Inherited training metadata: same65536 learning vectors, split94000011,
PQ seed94000037,25 iterations,24 historical training threads; graph construction
seed20260907/24 threads. None is an online internal-thread setting. Inputs and
prior runs remain unchanged. Full report:
`docs/phase4d_multithread_scalability.md`.

Interpretation: Case A supported on this workload. No materially widening
ALL16-specific efficiency loss or disproportionate tail breakdown through8
physical cores. Bounded shallow exact refinement is worth external validation,
not a universal systems/novelty claim. Do not return to query-level risk models.

Exactly one next experiment: one preselected high-dimensional real embedding
dataset external-validity gate, NATIVE versus unchanged bounded ALL-L16 at1/8
workers, with independently fixed reasonable PQ/HNSW settings and offline
candidate-oracle recall control. Jointly assess recall,QPS,p95/p99, without
choosing data/policy settings based on favorable test outcomes.

Final independent audit PASS:61440 stress,2,280,320 primary and228032 profile
returned-ID/count checks, regenerated deterministic schedule, all immutable
input/source/binary hashes unchanged, shared model/graph hashes unchanged,
28 byte-identical regenerated tables/figures. Artifact manifest:
`runs/phase4d_multithread_scalability_v1/final_audit.json`.

## Phase 5A preflight only — 2026-09-08

Parent Git `d56f1b2c8edfba8e7dc9a53eab46096b2b4d9198`, pinned FAISS
`20f14b31a6d54e243a3d1de6ae193fc4c3ec18ed`. No retrieval outcomes inspected.
Config: `configs/indexes/phase5a_highdim_external_validity.conf`.
Preregistration: `runs/phase5a_highdim_external_validity_v1/preregistration.md`.
Dataset/PQ/GT preparation and all requested experiments remain NOT RUN.

Commands (network calls required sandbox escalation):

```bash
curl -I -L --max-time 30 https://ann-benchmarks.com/dbpedia-openai-1000k-angular.hdf5
curl -I -L --max-time 30 https://storage.googleapis.com/ann-filtered-benchmark/datasets/dbpedia_openai_1M.tgz
curl -L --fail --max-time 30 https://huggingface.co/api/datasets/KShivendu/dbpedia-entities-openai-1M
curl -L --fail --max-time 30 https://raw.githubusercontent.com/qdrant/vector-db-benchmark/master/datasets/datasets.json
mkdir -p runs/phase5a_highdim_external_validity_v1/source
curl --fail --location --retry 3 --continue-at - --output runs/phase5a_highdim_external_validity_v1/source/dbpedia_openai_1M.tgz https://storage.googleapis.com/ann-filtered-benchmark/datasets/dbpedia_openai_1M.tgz
curl -L --fail --max-time 25 --range 0-1048575 -o /tmp/phase5a_archive_header.gz https://storage.googleapis.com/ann-filtered-benchmark/datasets/dbpedia_openai_1M.tgz
curl -I -L --max-time 20 https://storage.googleapis.com/ann-datasets/ann-benchmarks/dbpedia-openai-1000k-angular.hdf5
curl --http1.1 --fail -L --max-time 40 --range 0-8388607 -o /tmp/phase5a_hdf5_header https://storage.googleapis.com/ann-datasets/ann-benchmarks/dbpedia-openai-1000k-angular.hdf5
curl --fail -L --retry 2 --retry-all-errors --max-time 30 --range 0-1048575 -o /tmp/phase5a_hdf5_header https://storage.googleapis.com/ann-datasets/ann-benchmarks/dbpedia-openai-1000k-angular.hdf5
curl --fail -L --max-time 25 --range 0-1048575 -o /tmp/phase5a_hdf5_alternate_header https://ann-datasets.storage.googleapis.com/ann-benchmarks/dbpedia-openai-1000k-angular.hdf5
curl --fail -L --max-time 30 --output runs/phase5a_highdim_external_validity_v1/source/dbpedia-openai-1000k-angular.hdf5.partial https://storage.googleapis.com/ann-datasets/ann-benchmarks/dbpedia-openai-1000k-angular.hdf5
cp /tmp/phase5a_hdf5_header runs/phase5a_highdim_external_validity_v1/source/hdf5_prefix_1MiB.metadata_only
cp /tmp/phase5a_archive_header.gz runs/phase5a_highdim_external_validity_v1/source/archive_prefix_1MiB.incomplete.gz
/usr/local/bin/python scripts/phase5a_inspect_header.py runs/phase5a_highdim_external_validity_v1/source/hdf5_prefix_1MiB.metadata_only --artifact-size 6160008192 --output runs/phase5a_highdim_external_validity_v1/source/header_metadata.json
/usr/local/bin/python -m unittest discover -s tests -p test_phase5a_header.py
git diff --check
```

Observations: original HDF5 link404; archive full transfercurl92 after15113B;
mirror HEAD200; 1MiB range succeeded(~85KB/s); forcedHTTP1.1 failedcurl35;
alternate hostname range also succeeded(~104KiB/s). A30-second full HDF5
transfer probe was used to measure whether the range result was misleading;
no partial file qualifies as a valid input. Prefix metadata identifies
990000×1536 train and10000×1536 test, FP32/angular; contents/GT unchecked.
Metadata inspection uses a disposable sparse file solely to satisfy HDF5's
EOF check, never for reading vector data; missing object metadata is explicitly
marked missing. One metadata/no-overwrite unit test passed. This is not an
ANN/GT/retention correctness test.

Full HDF5 probe outcome:2,867,200 bytes/30s, curl28 at the deliberately set
timeout. Estimated full duration~18h at that rate; the mirror is reachable,
not proven unavailable. No background transfer is left active. Complete
artifact acquisition remains pending; no dataset integrity claim is made.

Environment: Intel i9-10920X,12physical/24logical CPUs, approximately30GiB
RAM/27GiB available, workspace929GB free, /tmp101GB free at initial preflight.
Old plotting environment lacks h5py; attempting
`/tmp/phase3b-plot-env/bin/python -m pip install h5py==3.11.0` failedTLS EOF.
Existing `/usr/local/bin/python` has h5py2.10.0 and was sufficient; no dependency
or search implementation was changed. Two initial metadata probe attempts
failed (anonymous file pathname, then out-of-prefix object); fixed without
writing a false success artifact. All incomplete downloads preserved.

Confounders/prerequisites: full integrity unverified; reserving training IDs
changes the canonical base and invalidates provided GT for primary scoring;
high-dimensional PQ table cost must be included; seed alone does not guarantee
multithread graph reconstruction. No scientific interpretation yet. Next
operational step is a complete verified artifact, then the frozen primary
experiment—not parameter tuning or a low-dimensional substitute.

## Phase 5A pre-run dataset amendment — 2026-09-09

User approved availability-based replacement of ada-002 with local
`Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M`, cached source
revision `4a9731217921bc476a0f03544f11f22ae4903fa5`, BEFORE any ANN results.
Parent Git `d56f1b2c8edfba8e7dc9a53eab46096b2b4d9198`.
Binding design: `docs/phase5a_dataset_amendment.md`; frozen config:
`configs/indexes/phase5a_highdim_external_validity.conf`.

```bash
/rwproject/kdd-db/kluaq/miniconda3/envs/build_env/bin/python -m unittest discover -s tests -p test_phase5a_dataset.py
/rwproject/kdd-db/kluaq/miniconda3/envs/build_env/bin/python scripts/freeze_phase5a_amendment.py
/rwproject/kdd-db/kluaq/miniconda3/envs/build_env/bin/python scripts/audit_phase5a_amendment.py
git diff --check -- docs/experiment_log.md
```

Environment recorded in manifest: NumPy1.26.4, PyArrow17.0.0, existing build_env
Python; no dependency installation. Freezer read all26 local Arrow shards,
calculated individual SHA-256, validated1000000 rows,1536 coordinates,
no null/nonfinite/zero vectors, max normalized FP32 norm error2.995396664e-08.
Three unit tests passed; regenerating complete role ID lists matched saved
bytes and hashes. These are data/preprocessing tests, not retrieval tests.

Split PCG64 seed20260909: first990000 permuted rows base, remaining10000 query;
training65536 sampled from base positions only, independent PCG64 seed95000011.
Base insertion/query/training order is explicitly saved. First cast to FP32,
then FP64 norm/division of those values, final FP32 storage. PQ768×8 and all
other ANN/system hypotheses remain unchanged. No ground truth, training,
graph construction or ANN experiment was run. This commit is the required
pre-retrieval gate, not evidence about scientific hypotheses.

Manifest: `runs/phase5a_highdim_external_validity_v1/amendment/dataset_manifest.json`,
SHA-256 `98aa2949f8faf56aea0e3c35da9f73c549a32a1f5b4ba4280e343dff25b36ebd`.
Source shard paths/hashes/row ranges, source metadata hash, code/config hashes,
all role-ID hashes and environment are retained there. No old files deleted.
Confounders: dataset version is different; no official evaluation split;
row exclusion does not guarantee semantic independence/content uniqueness;
local hashes are not publisher checksum authentication. Characterize as
held-out entity-to-entity NN, not natural-language-query retrieval.
Next operational step after committing: use the frozen inputs/preprocessing
for the original Phase5A external-validity experiment, without retuning.

## Phase 5A primary execution — implementation and reconstruction

Started from committed/pushed amendment
`fbdf4784a505dfbf201be9e78f788e67d31fd4b2`. No dataset/PQ/normalization/split
changes. Latest user request increases independent GT validation to100
queries; the immutable config still records the earlier32 and execution
metadata explicitly records100. Details: `runs/phase5a_highdim_external_validity_v1/execution_notes.md`.

Commands so far:

```bash
/rwproject/kdd-db/kluaq/miniconda3/envs/build_env/bin/python scripts/prepare_phase5a.py
# Interrupted random NFS writes after78924 rows; preserved, not used:
mv runs/phase5a_highdim_external_validity_v1/prepared runs/phase5a_highdim_external_validity_v1/prepared_interrupted_nfs_writes
# Same frozen vector arithmetic/IDs, local NVMe writes + sequential archive copy:
/rwproject/kdd-db/kluaq/miniconda3/envs/build_env/bin/python scripts/prepare_phase5a.py
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target faiss_phase5a faiss_phase5a_bench phase5a_retention_correctness -j 8
/rwproject/kdd-db/kluaq/miniconda3/envs/build_env/bin/python -m py_compile scripts/prepare_phase5a.py scripts/validate_phase5a_gt.py scripts/audit_phase5a_recall.py scripts/analyze_phase5a.py scripts/run_phase5a.py
/rwproject/kdd-db/kluaq/miniconda3/envs/build_env/bin/python -m unittest discover -s tests -p test_phase5a_dataset.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ctest --test-dir build --output-on-failure
cp build/Testing/Temporary/LastTest.log runs/phase5a_highdim_external_validity_v1/implementation_tests.log
```

Implementation validation:3 Python split/preprocessing tests passed; all7
C++ tests passed, including new d1536/PQ768 fixture at ef32/64/128 comparing
bounded retention to full sort and exact refinement references, native IDs
and graph fingerprint. Synthetic fixture uses deterministic centroids, not
another trained primary PQ model. Test log is preserved. No primary recall
or performance outcomes yet at this entry. Only CandidateAccess ef guard
generalized; FAISS HNSW and L16 comparator unchanged.

Build anomaly: NFS worktree scan stalled CMake; status now has30s timeout and
conservatively marks dirty on failure. Initial configure interrupted; completed
configure/build then succeeded with existing clock-skew warnings. No compiler
optimization flags changed. Test outcomes validate compiled code, not a claim
that the NFS clock is synchronized. Query timing uses steady_clock, not mtime.
Next: finish verified arrays, frozen GT/index/PQ stages, then correctness-gated
primary measurement. All exploratory extensions remain forbidden.

Pre-ANN execution recovery: after all1000000 normalized rows and sequential
archive copies, the metadata call `git -C third_party/faiss rev-parse HEAD`
stalled. A20s subprocess timeout isolated it; direct Git-dir lookup returned
the pinned20f14b31a6d54e243a3d1de6ae193fc4c3ec18ed. The process was interrupted
without deleting arrays. Recovery command (same arithmetic/IDs):

```bash
git --git-dir=third_party/faiss/.git rev-parse HEAD
/rwproject/kdd-db/kluaq/miniconda3/envs/build_env/bin/python -u scripts/prepare_phase5a.py --finalize-existing /tmp/phase5a-normalization-aifuhbw5
```

Finalizer must match full scratch/archive hashes and recheck all output
vectors and all training/base alignment before emitting PASS. No GT or ANN
result was inspected during either preparation recovery.

Preparation recovery completed PASS: all normalized array archive/scratch
hashes match; all output vectors finite/unit-norm; all65536 training rows
match saved base positions exactly. Base SHA-256
`313519a90d5baf29024167aff7bef53736303bb4ac02436c6709293d2de0b86c`;
query `968414d2566fafbaefdbe2705e230ab06c09b7c020565f95b17648ccbd7bdb8e`;
training `176b72ea78a75926add9f8aa915a3de70963f9e3a6f13d9e3f4e0627274a8235`.
Max normalized norm error2.995396664040584e-08. Source and prepared provenance
are retained. Primary-only offline execution started after these gates:

```bash
/rwproject/kdd-db/kluaq/miniconda3/envs/build_env/bin/python -u scripts/run_phase5a.py --stage offline
```

Expanded exact commands/environment: `execution_logs/offline_commands.json`
under the Phase5A run. It runs GT, independent GT references/validation, one
graph, one PQ, sanity, recall, correctness audit and recall analysis in that
order; any failure stops. Logs are exclusive-create, no overwrite. Source/
binary/flags/environment hashes: `execution_provenance.json`.

GT completed:10000 queries, primary exhaustive FP32 kernel runtime551.1513s;
stored top11 (first10 are Recall@10 labels; last is tie diagnostic), ties
ordered by ascending base ID. GT IDs SHA-256
`27d79de76e344ab0237dde665d38a0b400e5847e5ef4893823dbfc9f4b5fd495`.
Independent scalar FP64 L2 and NumPy FP64 cosine scans of100 deterministic
queries passed with no top10 membership disagreement. Max primary FP32 vs
FP64 distance error4.811763674e-07 (<2e-5); zero exact FP32 rank10/11 ties
among all10000 queries. Strict ID recall unchanged. Full details and hashes
in `gt_validation.json`. The one frozen FP32 graph construction then started.

The single FP32 graph completed: M16, efConstruction80, seed20260907,
24 construction threads, graph.add time1243.666647s; serialized size
6225374202 bytes. Fingerprint
`cd95032163d55937f2d4e90b1ab6e33d5509c938987d10a3c2cacc16d01b8f5d`;
file SHA-256 `ddc7a331a1a664bf41ae67408b6e7e35321fbb47c62b64ace18c2c8d1eb36a2b`.
The runner's2752.7s graph-stage elapsed time includes serialization/hash/I/O
and must not be reported as graph.add time. No graph was rebuilt.

One PQ768x8 model completed: training799.073615s, encoding70.626760s;
seed95000037,25 iterations,1 redo,65536 frozen training rows,dsub2,768B/code.
Codebook SHA-256 `78c2dbd08ae1deabd177022609382ff56e1f3928c718419eb2be5d211ae035e0`;
codes SHA-256 `a50637f62259fb08ed3425868194ef9d42aaeb3dfba73a8b1b5fe7f1fb85f51b`.
Pre-ANN sanity gate passed10000 deterministic distance/reconstruction samples
and10000 order pairs, including independent decoded-vector ADC comparison
and sampled exact code/ID alignment. No alternative PQ was trained.

Primary recall completed340.3s runner elapsed; independent saved-output audit
PASS for all30000 query/ef rows, no exact-control anomalies, no top16 boundary
ties, identical graph fingerprint and native/full/bounded reference IDs.
ef32/64/128 exact recall: .89641/.94202/.96602; PQ: .87758/.91685/.93799;
candidate oracle: .89657/.94087/.96588; ALL16: .89655/.94084/.96585.
Signed discovery losses: -.00016/.00115/.00014. The slightly negative ef32
aggregate is preserved; the two traversals may discover different sets.
Sanity sampled relative distance MAE .00861728, reconstruction L2 norm
mean .08872676, random strict pairwise inversion rate .0254. No parameter
selection or exploratory diagnostics were performed after these outcomes.

After offline completion and source/binary hash verification, primary systems
execution launched:

```bash
/rwproject/kdd-db/kluaq/miniconda3/envs/build_env/bin/python -u scripts/run_phase5a.py --stage systems
```

It runs single-thread six-point benchmark and analysis BEFORE the limited
ef64 1/8-worker check. Five randomized-order repetitions per cell, four query
passes per repetition, separate512 training-row warmups/worker; every timed
return ID checked against the offline reference. CPU0 coordinator, CPU2..9
distinct physical workers, internal OpenMP/BLAS1. Expanded exact commands in
`execution_logs/systems_commands.json`; full environment in execution provenance.
Process listing within the command sandbox does not expose the entire host;
absence of competing host workloads cannot be guaranteed. Before/after host
load and CPU/memory counters are preserved per cell for interpretation.

Single-thread primary complete:30 cells,5 repetitions of six points,1200000
timed queries, every timed output matching its reference. Stage2242.2s includes
untimed loading/checks/I/O. QPS native/ALL16 at ef32:959.64/928.88;
ef64:740.08/726.92; ef128:520.09/516.42. Throughput penalties3.206%/1.779%/.706%.
Observed ALL16 ef64 versus native ef128: Recall difference+.00285 and QPS
ratio1.39767, no interpolation. Primary single analysis completed before the
limited ef64 1/8-worker benchmark began. No parameters changed.

Limited concurrency completed830.9s stage elapsed,20 cells,800000 timed queries,
all outputs identical. ef64 native/ALL16 QPS at1 worker735.43/719.97 and at8
workers5591.49/5478.56. Throughput penalties2.103%/2.020%;8-worker efficiencies
95.04%/95.12%;8-worker p99 overhead2.366%. Separate concurrency-block1-worker
timings drift more than the primary block; repetition3 is slower for both,
and is retained. Before/after host load2.0–10.5, minimum available memory21.44GiB;
no exclusive-host or fixed-frequency claim. Governor performance, turbo enabled.

Post-run correctness/reproduction commands (no new ANN experiment):

```bash
/rwproject/kdd-db/kluaq/miniconda3/envs/build_env/bin/python scripts/verify_phase5a_outputs.py
/rwproject/kdd-db/kluaq/miniconda3/envs/build_env/bin/python scripts/summarize_phase5a_environment.py
mktemp -d /tmp/phase5a-regeneration-XXXXXX
MPLCONFIGDIR=/tmp/phase5a-mpl-cache /tmp/phase3b-plot-env/bin/python scripts/analyze_phase5a.py --stage final --tables /tmp/phase5a-regeneration-eoWhre/tables --figures /tmp/phase5a-regeneration-eoWhre/figures
/rwproject/kdd-db/kluaq/miniconda3/envs/build_env/bin/python scripts/summarize_phase5a_environment.py --tables /tmp/phase5a-regeneration-eoWhre/tables
/rwproject/kdd-db/kluaq/miniconda3/envs/build_env/bin/python scripts/verify_phase5a_regeneration.py /tmp/phase5a-regeneration-eoWhre
```

Independent audit PASS:24 frozen files unchanged, graph/PQ file hashes match,
all2000000 timed IDs match their reference.29 regenerated tables/figures are
byte-identical. Audit scripts added after timing are separately hashed in
`analysis_regeneration.json`; they did not alter timed implementation.

Observations: high-recall ef64/128 candidate-oracle recovery95.43%/99.50%;
L16 ranking-gap recovery99.875%/99.892%. ef32 recovery100.85% is preserved,
not clipped. Discovery sign cancellation exists; ef64 positive2.04%, negative
1.61%, including rare large signed cases. No exact-control anomalies or PQ
top16 boundary ties. Strict recall definitions and all adverse cases retained.

Interpretation (separate from observations): H1/H2/H3 pass in this frozen
configuration; Case A supported with single-model/workload/hardware limits.
The possible H4 larger relative overhead does not occur, despite larger exact
payload and central absolute latency increment. No causal bandwidth claim.
Full report: `docs/phase5a_highdim_external_validity.md`.

Smallest next discriminating experiment, recommendation ONLY: freeze a
PQ384x8/dsub4/16x-compression replication on the identical DBPedia split and
graph, same training IDs/seed, ef32/64/128 and L16, compared with current
PQ768x8. Test dependence on fine dsub2 precision, not a broad parameter sweep.
This next experiment has NOT been run. Phase5A primary execution is complete
and stopped; no unauthorized exploratory diagnostics were launched.

## Phase5B — preregistered compression robustness

User subsequently authorized exactly one PQ384x8 experiment on frozen Phase5A
artifacts. Parent commit541a0eb8c812bf9926f0bc5de3699d908dfa20a4. Preregistration
`runs/phase5b_compression_robustness_v1/preregistration.md` and configuration
`configs/indexes/phase5b_compression_robustness.conf` were written before model
training/retrieval. H1 oracle recovery>=.90; H2 ranking-side predominance;
H3 L16 recovery>=.80 at one practical point, secondary>=.60 at most points;
H4 systems direction left open. No graph rebuild, new split, normalization,
training sample, alternative PQ or L32/L64 refinement.

Commands:

```bash
cmake --build build --target faiss_phase5a faiss_phase5a_bench phase5a_retention_correctness -j 4
/rwproject/kdd-db/kluaq/miniconda3/envs/build_env/bin/python -u scripts/prepare_phase5b.py
/rwproject/kdd-db/kluaq/miniconda3/envs/build_env/bin/python -u scripts/run_phase5b.py --stage model
```

Input verification PASS: all archived vectors, role IDs reconstructed from
frozen seeds, exact GT/reference files, serialized graph, original PQ and
historical result tables match committed Phase5A metadata. No graph rebuilt.
Symlinks reference original arrays and graph, not newly transformed copies.
Model stage runs correctness tests before one PQ384 training and sanity.
Shared driver accepts768/384; online instrumentation/search code is unchanged
and checked against its Phase5A hash. The bounded fixture now tests both
1536-dimensional PQ layouts against full-sort/reference exact reranking.
Build retains GCC11.5.0 Release generic flags; NFS clock-skew warnings persist
but modified targets compile/link successfully. Full model/source/environment
provenance is saved before training. Historical5A QPS is not contemporaneously
randomized with5B; cross-phase timing comparisons retain this confound.

Phase5B model stage passed7/7 CTests and trained exactly one PQ384x8:
training530.700027s,encoding36.558373s,384B/code,dsub4,seed95000037,
25 iterations,1 redo. Codebook SHA-256
`25b67a5fb81c84f25f2d61dcc16f29847dedb86b27e063653390da03044c5061`;
base-code SHA-256 `883b61f52605d9ac3e2366aa77638ca12c9bea5673916c751914407ae0094aa2`.
Sanity implementation gate passed10000 paired/order samples. Additional
offline rank/tie tests passed2/2 using:

```bash
MPLCONFIGDIR=/tmp/phase5b-mpl-cache /tmp/phase3b-plot-env/bin/python tests/test_phase5b_analysis.py
/rwproject/kdd-db/kluaq/miniconda3/envs/build_env/bin/python -u scripts/run_phase5b.py --stage recall
```

Recall stage starts only after model/sanity completion. The analysis/audit
source hashes are frozen before this retrieval stage. Native exact results
must equal Phase5A on all queries; no results are used to change configuration.

Conversation/usage interruption recovery: the old unified-process handle no
longer existed when the user returned, but all three recall completion files,
the PASS audit and generated recall tables were present. No training or recall
restart was performed. All30000 exact-control discrepancies are zero and
full/bounded/native identity passes. System benchmarks had not started.
Continuation command:

```bash
/rwproject/kdd-db/kluaq/miniconda3/envs/build_env/bin/python -u scripts/run_phase5b.py --stage systems
```

The runner verifies frozen source/config/binary hashes before starting. PQ384
recall native/oracle/ALL16 at ef32=.83320/.89709/.89341,
ef64=.86500/.94159/.93606, ef128=.88098/.96537/.95864. Signed discovery
losses−.00068/.00043/.00065. L16 recovery94.24%/92.78%/92.03%.
These observations did not change any model, threshold or benchmark policy.

Single-thread benchmark completed30 cells,1200000 timed queries,1091.11s
runner stage elapsed (includes untimed I/O/checks). PQ384 native/ALL16 QPS:
ef32 1656.37/1568.21;ef64 1297.42/1233.50;ef128 915.14/892.29.
ALL16 throughput penalties5.323%/4.927%/2.497%. All12 measured5A/5B points
were compared without interpolation: frontier=PQ384 nativeef32,PQ384 ALL16
ef32/64/128,and PQ768 ALL16ef128. Phase5A remains historical timing reference.
Single analysis completed before the limited ef64 1/8-worker benchmark began.

Limited concurrency complete20 cells/800000 timed queries,410.42s runner stage.
ef64 native/ALL16 QPS=1297.21/1242.11 at1 worker,9730.02/9285.84 at8 workers.
Throughput penalties4.247%/4.565%,8-worker efficiencies93.76%/93.45%,p99
overhead6.488%. All timed outputs pass. Final analysis then performed only
the authorized approximate-rank diagnostic from recorded candidate scores:
GT-relevant oracle members within top16=99.5898%/99.4127%/99.3029% at
ef32/64/128. No L32/L64 distances or search were run.

Post-run verification/reproduction commands:

```bash
/rwproject/kdd-db/kluaq/miniconda3/envs/build_env/bin/python scripts/verify_phase5a_outputs.py --root runs/phase5b_compression_robustness_v1
/rwproject/kdd-db/kluaq/miniconda3/envs/build_env/bin/python scripts/summarize_phase5a_environment.py --root runs/phase5b_compression_robustness_v1 --prefix phase5b
mktemp -d /tmp/phase5b-regeneration-XXXXXX
MPLCONFIGDIR=/tmp/phase5b-mpl-cache /tmp/phase3b-plot-env/bin/python scripts/analyze_phase5b.py --stage final --tables /tmp/phase5b-regeneration-KkuxnE/tables --figures /tmp/phase5b-regeneration-KkuxnE/figures
/rwproject/kdd-db/kluaq/miniconda3/envs/build_env/bin/python scripts/summarize_phase5a_environment.py --root runs/phase5b_compression_robustness_v1 --prefix phase5b --tables /tmp/phase5b-regeneration-KkuxnE/tables
/rwproject/kdd-db/kluaq/miniconda3/envs/build_env/bin/python scripts/verify_phase5a_regeneration.py /tmp/phase5b-regeneration-KkuxnE --root runs/phase5b_compression_robustness_v1 --prefix phase5b
/rwproject/kdd-db/kluaq/miniconda3/envs/build_env/bin/python scripts/verify_phase5b_controls.py
```

Independent benchmark audit PASS:14 frozen implementation/config/binary files,
all50 cells/2000000 timed query IDs match references.34 regenerated tables/
figures are byte-identical. Shared audit helpers were parameterized for root/
output prefix only; original5A outputs are unchanged. Raw Phase5B model is
kept locally and excluded from ordinary Git staging because of its size;
codebook/code/model hashes are recorded, no file deleted.

Host snapshots: load1.33–3.65,min available memory23012284KiB,max host-wide
iowait.187%,no free-swap change. These counters include warmup, not per-kernel
profiling. Separate-phase timing, turbo/frequency and one model/index are
confounders; all repetitions retained. Adverse discovery variation is also
retained: ef64 positive/negative query fractions grow2.04%/1.61% (PQ768) to
4.56%/3.91% (PQ384). Small mean discovery loss includes cancellation.

Interpretation: Case A supported at frozen aggregate operating points; H1/H2/H3
pass, but not a universal per-query or further-compression claim. Only suggested
next experiment: a preregistered contemporaneous baseline comparison against
pinned SymphonyQG on the same data/GT with explicit ISA/compiler/full-memory
accounting, using each system's native graph and observed Recall-QPS points.
Paper/author repository were inspected read-only for this recommendation;
no SymphonyQG implementation or additional experiment was run. Phase5B stops
after the authorized primary results and audits.

Final Phase5B post-run controls PASS:51 original input/reference hashes
unchanged, exact native/oracle IDs identical to5A, oracle=coverage for every
query,300000 candidate-rank rows independently validated by stable argsort.
For every query, oracle−ALL16 recall equals excluded relevant-item count/10;
outside-top16 totals368/553/673 at ef32/64/128. The extra full-file hash audit
took a long wall-clock interval; the completed result was recovered by polling
the existing session, without rerunning training/recall/benchmarks. No index
was rebuilt, file deleted, or parameter changed. Report and artifacts complete.
