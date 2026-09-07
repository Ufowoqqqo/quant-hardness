# Phase 3A: SIFT1M real-data gatekeeper

Status: completed. The criteria and grids below were committed to the file
before any SIFT1M decomposition result was generated.

## Scientific question

Does standard SIFT1M show reproducible quantization-induced level-0 candidate-
discovery degradation that was absent from the IID, query-shift, and clustered
synthetic controls?

## Fixed protocol

- Dataset: standard SIFT1M `sift_base.fvecs`, `sift_query.fvecs`, and
  `sift_groundtruth.ivecs`, obtained from Hugging Face repository
  `qbo-odp/sift1m` at immutable revision
  `bd8ccad6c2a0a0a3a7519f6d37c0e5a2d59fe55b`, with 1,000,000 base vectors,
  10,000 queries, dimension 128, and 100 provided neighbors. Each file checksum
  is fixed before search.
- One HNSW graph is built from FP32 vectors with `M=16`,
  `efConstruction=80`, graph seed `20260907`, and exact squared-L2 distance.
- The exact calibration grid is
  `{16,32,48,64,96,128,192,256,384,512}`. For each target in
  `{0.90,0.95,0.98}`, the closest-recall unused point is selected, with lower
  `efSearch` breaking ties. This rule does not inspect PQ decomposition.
- Standard PQ candidates are `M in {16,32,64}`, 8 bits per subquantizer,
  seed `30260907`. At the central exact operating point, eligible candidates
  must have PQ recall at least 0.70 and exact-minus-PQ mean recall in
  `[0.02,0.25]`. Among them, the configuration closest to a 0.10 gap is chosen;
  smaller `M` breaks ties. If none is eligible, the run stops before
  decomposition rather than changing the rule.
- Search is single-threaded and preserves query order. The built graph and
  trained PQ indexes are serialized once and reused in later stages.
- Discovery oracles use corrected `V_L0` semantics. Upper-level-only distance
  evaluations are excluded, while their effect on the level-0 entry remains.

## Ground-truth and approximation controls

A deterministic 100-query sample of provided ground truth is checked against
exhaustive FP32 squared-L2 top-10. Every PQ configuration retains per-query
native calibration rows plus independently sampled distance-error and pairwise-
order rows. PQ code slots are checked against direct encoding of deterministic
database IDs.

## Pre-registered gatekeeper criteria

These thresholds were fixed before inspecting decomposition results.

**No reproducible real-data discovery signal:** at every selected point with
exact recall at least 0.90,

```
abs(mean delta_discovery) <= 0.01
rerank recovery >= 0.90
```

**Emerging real-data discovery signal:** at least one reasonable point with
exact recall at least 0.90 has

```
mean delta_discovery >= 0.03
rerank recovery <= 0.80
```

**Strong real-data discovery signal:** the emerging criterion persists at two
nearby reasonable operating points and is not confined to an extreme PQ rate.

The thresholds are exploratory decision criteria, not significance claims.
Signed negative discovery and ranking deltas are retained.

SIFT descriptors can have exact distance ties. Provided-GT candidate coverage
and ID-based oracle recall are therefore retained as distinct metrics. A
mismatch is accepted only when instrumentation verifies that every covered GT
ID excluded by deterministic oracle reranking is tied exactly at the oracle
top-10 boundary; every such row is flagged. This exception does not apply to
the existing synthetic experiments, whose strict equality guard remains the
default.

For SIFT1M only, an independent exact rerank first establishes the top-10 FP32
distance multiset. A hard control then verifies that every exact-native ID is
in `V_L0` and that its distance multiset is identical. Exact-native IDs are
therefore used as the exact-oracle realization when boundary IDs tie, making
`delta_exact_control` well-defined without changing distances or Recall@10.
PQ-oracle ties use stable first-evaluation order, and all coverage/oracle
mismatches remain explicitly flagged.

## Planned outputs

The preparation, calibration, decomposition, and analysis stages write under
`runs/phase3a_sift1m_v1/`, `results/tables/phase3a_sift1m_*`, and
`results/figures/phase3a_sift1m_*`. Results, anomalies, confounders, and the
mandatory go/pivot decision will be appended after validation.

## Executed dataset and system

- Source revision: `qbo-odp/sift1m` at
  `bd8ccad6c2a0a0a3a7519f6d37c0e5a2d59fe55b`.
- Base: 1,000,000 x 128, SHA-256
  `21f66e2975057b5728ba56de1c825bac4f4d89d596609ae985741c6242631816`.
- Queries: 10,000 x 128, SHA-256
  `f7fc9be140accdfd64116c2fa2365ecdb69b8f084970c6b0532db5ff79ac8fdc`.
- Ground truth: 10,000 x 100, SHA-256
  `2b71de0a8d5a83e6a84eec3e23fb8b611d8801dd9b3a6cd62f070ab65ea65f4f`.
- A deterministic 100-query sample had top-10 set recall 1.0 against
  exhaustive FP32 L2 for every sampled query.
- HNSW fingerprint:
  `1b3c18f6e2dd3ccec56e74eed30458f60a65f01234bf51478af25d84b5c02d16`.
  One graph was built and serialized; all calibration and decomposition runs
  loaded that frozen graph.
- FAISS commit: `20f14b31a6d54e243a3d1de6ae193fc4c3ec18ed`.
- Repository HEAD recorded by the binary:
  `4e5137dd71e64df669eeb4c18b9f4af3aee7a556`, with a dirty worktree containing
  this Phase 3A implementation.
- Machine: `rwcpu8.cse.ust.hk`, Linux 5.14.0-687.24.1.el9_8.x86_64, GCC 11.5.0,
  24 hardware threads. Graph construction used 24 threads; every search used
  one thread and ascending query order.

The xvec loader checks every repeated row dimension, complete-record file
size, query/GT row agreement, and GT ID range. Preparation took 33.70 seconds
for HNSW construction. PQ16, PQ32, and PQ64 training plus encoding took 37.32,
49.10, and 74.31 seconds, respectively.

## Calibration observations

Exact HNSW recall increased monotonically over the registered grid:

| efSearch | 16 | 32 | 48 | 64 | 96 | 128 | 192 | 256 | 384 | 512 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Recall@10 | .78164 | .88629 | .92974 | .95319 | .97442 | .98379 | .99192 | .99491 | .99748 | .99836 |

The pre-registered closest-target rule selected efSearch 32, 64, and 128.
The low point is 0.01371 below 0.90; it is reported but excluded from criteria
that explicitly require exact recall at least 0.90. No post-result substitution
with efSearch 48 was made.

At the central point, the PQ-rate calibration was:

| PQ | bytes/vector | PQ Recall@10 | exact minus PQ | mean abs. relative error | pair inversion rate |
|---|---:|---:|---:|---:|---:|
| 16x8 | 16 | .54268 | .41051 | .04895 | .0574 |
| 32x8 | 32 | .70025 | .25294 | .02724 | .0386 |
| 64x8 | 64 | .85245 | .10074 | .00953 | .0142 |

PQ64x8 was selected by the registered target-gap rule. PQ16 was destructive;
PQ32 missed the allowed maximum gap by 0.00294. PQ64 had 8x compression versus
FP32, a 0.95 percentile absolute squared-distance error of 7108.92, and a
sampled strict inversion rate of 0.0142. Selection did not inspect discovery
metrics.

## L0 decomposition observations

| efSearch | exact native | PQ native | PQ L0 oracle | mean total delta | mean discovery delta | mean ranking delta | rerank recovery | L0 Jaccard |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | .88629 | .80964 | .88402 | .07665 | .00227 | .07438 | .97038 | .78223 |
| 64 | .95319 | .85245 | .95200 | .10074 | .00119 | .09955 | .98819 | .82464 |
| 128 | .98379 | .86867 | .98287 | .11512 | .00092 | .11420 | .99201 | .85128 |

Observation: at both criterion-eligible points, exact reranking of the PQ
`V_L0` recovered more than 98.8% of the exact-minus-PQ native gap. Mean
discovery delta was below 0.0012, while mean ranking delta accounted for nearly
all total degradation. Increasing efSearch increased PQ candidate coverage
and evaluated-set overlap; total delta nevertheless increased because exact
and PQ-oracle recall approached one while PQ-native recall rose much less.

Per-query discovery heterogeneity was:

| efSearch | median | p90 | p95 | p99 | fraction >0 | fraction >=.1 | fraction >=.2 | fraction =0 | fraction <0 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 0 | .1 | .1 | .3 | .1182 | .1182 | .0364 | .7761 | .1057 |
| 64 | 0 | 0 | .1 | .2 | .0639 | .0639 | .0125 | .8808 | .0553 |
| 128 | 0 | 0 | 0 | .1 | .0295 | .0295 | .0035 | .9494 | .0211 |

Among positive-loss queries, the fractions needed for 50%/80%/90% of positive
discovery loss were 0.282/0.702/0.851 at ef32, 0.377/0.751/0.876 at ef64, and
0.441/0.776/0.888 at ef128. This concentration weakens as efSearch grows and
does not support a strong or stable heavy-tail description. Negative discovery
deltas were retained.

Spearman associations between discovery delta and the six recorded geometric
descriptors were small. Across eligible points, the largest absolute value was
0.0214 (exact 10-NN distance at ef128). These are exploratory correlations,
not causal estimates.

## Correctness controls and anomalies

- `delta_exact_control` was exactly zero for all 30,000 completed rows.
- Native PQ IDs at the central point matched the uninstrumented calibration
  IDs query-for-query. Instrumented search also performs an internal native
  replay equality check for exact and PQ runs.
- The graph fingerprint was identical in every raw row and after each search.
- PQ code slots for IDs 0, 500000, and 999999 exactly matched direct encoding
  under every tested PQ configuration.
- Exact/PQ candidate-set IDs were retained for the first 100 queries at every
  operating point; scalar metrics and all five result-ID arrays were retained
  for every query.
- Integer SIFT vectors produced exact boundary ties. Candidate coverage and
  ID-based oracle recall differed on 29/28 exact/PQ rows at ef32, 46/49 at
  ef64, and 55/64 at ef128. Every mismatch passed the exact-boundary-tie
  validator. The exact-native result for every query was independently shown
  to have the same FP32 distance multiset as exact reranking of `V_L0`.
- Two 93-row partial outputs from the failed pre-correction guards are retained
  under `runs/phase3a_sift1m_v1/failed_attempts/`; they are excluded from all
  aggregate results. Three completed decomposition files using the superseded
  field name `rerank_recovery_query` are retained under
  `runs/phase3a_sift1m_v1/superseded_schema/`; their replacement files use
  `rerank_recovery` and are the inputs to the reported analysis.
  `metadata_amendment.json` records stage history and exact SHA-256 hashes for
  the completed decomposition binary and relevant sources.
- The build host reported file timestamps up to about four minutes in the
  future. Clean rebuilds and all five tests passed despite this filesystem
  clock-skew warning.

Possible confounders are that this is one fixed real dataset and one graph/PQ
seed, uniform random query-database pairs underweight near-neighbor PQ error,
and PQ64 is the only rate taken through full decomposition. The nearby exact
operating points provide mechanism consistency but are not independent index
replicates.

## Gatekeeper decision

The pre-registered **No reproducible real-data discovery signal** criterion
passes at efSearch 64 and 128: both have exact recall at least 0.90,
`abs(mean delta_discovery) <= 0.01`, and rerank recovery at least 0.90. Neither
the emerging nor strong criterion passes. efSearch 32 is reported as a useful
lower-recall robustness point but is not used for this gate.

Thus the candidate-discovery/navigability hypothesis has received negative
evidence from IID synthetic data, query shift, clustered synthetic geometry,
and SIFT1M real data. Full trajectory instrumentation is not justified by the
current evidence.

The decision is **B: pivot away from candidate-discovery failure and toward
the empirically supported ranking/selection mechanism**. The single smallest
next experiment is a reference-only SIFT1M candidate-ranking analysis that
holds each recorded `V_L0_pq` fixed and relates exact-versus-PQ top-10 selection
errors to exact local distance margins and PQ distance errors. It adds no new
algorithm and directly tests the remaining mechanism.
