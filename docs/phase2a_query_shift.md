# Phase 2A: controlled query-distribution shift

## Objective and pre-registration

This experiment asks whether changing only the query distribution causes
candidate-discovery degradation under PQ-guided HNSW traversal. It does not
assume that such degradation exists.

Before execution, the configuration fixed four independent seed tuples, the
shift grids, and these exploratory criteria:

- **A, no evidence of discovery breakdown:** for every non-pathological shift
  with mean exact Recall@10 at least 0.90,
  `abs(mean delta_discovery) <= 0.01` and rerank recovery at least 0.90.
- **B, emerging discovery degradation:** at least one shift with mean exact
  Recall@10 at least 0.90 has mean discovery delta at least 0.03 and recovery
  at most 0.80 in at least three of four replicates.
- **C, strong discovery breakdown:** discovery delta increases systematically
  with shift severity while exact recall remains high.

These are qualitative research criteria, not significance tests. They were not
changed after examining results.

## Method

Each replicate contains 20,000 FP32 database vectors from `N(0,I_64)`, 500
queries, one exact-FP32-built HNSW graph (`M=16`, `efConstruction=80`), and one
FAISS PQ32x8 model trained on all 20,000 database vectors. Search used `k=10`,
a bounded queue, relative-distance checking, one thread, and ascending query
IDs. Exhaustive FP32 squared-L2 top-10 ground truth was recomputed for every
query set.

One latent standard-normal query vector `z` was reused across severities so
the contrasts are paired. Mean-shift queries use
`q=z+alpha*sqrt(64)*u`, where one seeded unit direction `u` is shared within a
replicate. Radial queries use `q=sigma*z`. The IID row is the shared
`alpha=0`/`sigma=1` control.

Within a replicate, the database, graph, graph fingerprint, PQ codebooks, and
PQ codes were frozen across all query conditions. Their hashes were checked
after every condition. Exact and PQ native searches were repeated without and
with instrumentation; any result-ID or distance change was a fatal error.

The evaluated set contains every unique database ID passed to the FAISS
query-to-node `DistanceComputer`, including upper-level greedy descent and
level-0 exploration. Exact reranking is therefore an oracle over every
distance-evaluated node, not a practical reranking budget. Scalar values and
all result/ground-truth IDs are retained for all queries. Sorted evaluated ID
sets are retained for deterministic query IDs 0--24 in every condition; this
keeps the raw run at 79 MiB while preserving 16,000 complete scalar rows.

The run used Git commit `f56bdcbf462193b1f9d3e733e35a9b5d05401b0e`
with the Phase 2A changes uncommitted (`dirty_worktree=1`), Meta FAISS commit
`20f14b31a6d54e243a3d1de6ae193fc4c3ec18ed`, and
`rwcpu8.cse.ust.hk` (Linux 5.14.0-687.24.1.el9_8.x86_64, GCC 11.5.0). All
database/query/graph/PQ/direction seeds are in the resolved configuration and
replicate manifests.

## Fixed-ef observations

The values below are means across four replicate-level means, followed by
sample standard deviations. `PQ oracle` is exact reranking of the PQ traversal
evaluated set.

| Condition | Exact native | PQ native | PQ oracle | Discovery delta | Ranking delta | Recovery | Jaccard |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| IID | 0.9649 +/- 0.0037 | 0.8253 +/- 0.0049 | 0.9629 +/- 0.0047 | 0.0020 +/- 0.0023 | 0.1376 +/- 0.0032 | 0.9860 +/- 0.0163 | 0.8508 +/- 0.0011 |
| mean alpha=0.25 | 0.9608 +/- 0.0028 | 0.8231 +/- 0.0039 | 0.9597 +/- 0.0022 | 0.0010 +/- 0.0012 | 0.1367 +/- 0.0044 | 0.9925 +/- 0.0086 | 0.8514 +/- 0.0012 |
| mean alpha=0.50 | 0.9473 +/- 0.0013 | 0.8169 +/- 0.0025 | 0.9464 +/- 0.0016 | 0.0010 +/- 0.0018 | 0.1295 +/- 0.0023 | 0.9928 +/- 0.0134 | 0.8513 +/- 0.0007 |
| mean alpha=0.75 | 0.9298 +/- 0.0023 | 0.8145 +/- 0.0050 | 0.9287 +/- 0.0013 | 0.0010 +/- 0.0020 | 0.1143 +/- 0.0038 | 0.9912 +/- 0.0179 | 0.8523 +/- 0.0008 |
| mean alpha=1.00 | 0.9070 +/- 0.0092 | 0.8052 +/- 0.0080 | 0.9073 +/- 0.0077 | -0.0002 +/- 0.0026 | 0.1021 +/- 0.0054 | 1.0022 +/- 0.0260 | 0.8538 +/- 0.0013 |
| radial sigma=1.25 | 0.9314 +/- 0.0039 | 0.8165 +/- 0.0025 | 0.9305 +/- 0.0037 | 0.0008 +/- 0.0009 | 0.1140 +/- 0.0034 | 0.9927 +/- 0.0080 | 0.8552 +/- 0.0015 |
| radial sigma=1.50 | 0.8972 +/- 0.0053 | 0.8018 +/- 0.0044 | 0.8965 +/- 0.0041 | 0.0006 +/- 0.0021 | 0.0947 +/- 0.0064 | 0.9943 +/- 0.0208 | 0.8564 +/- 0.0013 |
| radial sigma=2.00 | 0.8353 +/- 0.0062 | 0.7679 +/- 0.0057 | 0.8349 +/- 0.0067 | 0.0005 +/- 0.0018 | 0.0670 +/- 0.0032 | 0.9939 +/- 0.0279 | 0.8562 +/- 0.0007 |

Mean total delta decreased from 0.1396 in IID to 0.1018 at alpha=1 and
0.0675 at sigma=2. This is not evidence that OOD improves PQ search: exact
HNSW recall also decreased, and the absolute room for an exact-versus-PQ gap
contracted. Mean discovery delta stayed between -0.0002 and 0.0020. Ranking
delta remained the numerically dominant component.

Per-query discovery deltas became less often exactly zero with severity: the
zero fraction was 0.891 in IID, 0.817 at alpha=1, and 0.756 at sigma=2.
Positive and negative cases increased together. At sigma=2 their fractions
were 0.124 and 0.121, leaving the mean near zero. Thus greater per-query
variation did not produce a systematic positive discovery loss.

All 21 shifted replicate/condition rows whose fixed-ef exact recall was at
least 0.90 satisfied criterion A. No condition satisfied criterion B, and the
severity curves do not satisfy criterion C.

## Exact-recall calibration and matched results

Seven replicate/condition pairs fell below 0.90 at fixed ef: all four
`sigma=2` pairs, two `sigma=1.5` pairs, and one `alpha=1` pair. The prescribed
exact-only sweep selected efSearch 1024 for every `sigma=2` pair and efSearch
512 for the other three. Every selected point restored exact Recall@10 to at
least 0.95.

| Replicate / condition | efSearch | Exact native | PQ native | PQ oracle | Discovery delta | Recovery |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R0 / sigma=2 | 1024 | 0.9602 | 0.8466 | 0.9620 | -0.0018 | 1.0158 |
| R1 / sigma=2 | 1024 | 0.9636 | 0.8402 | 0.9632 | 0.0006 | 0.9968 |
| R2 / sigma=1.5 | 512 | 0.9504 | 0.8304 | 0.9494 | 0.0010 | 0.9917 |
| R2 / sigma=2 | 1024 | 0.9604 | 0.8360 | 0.9578 | 0.0026 | 0.9791 |
| R3 / alpha=1 | 512 | 0.9550 | 0.8262 | 0.9536 | 0.0014 | 0.9891 |
| R3 / sigma=1.5 | 512 | 0.9524 | 0.8300 | 0.9504 | 0.0020 | 0.9837 |
| R3 / sigma=2 | 1024 | 0.9574 | 0.8396 | 0.9548 | 0.0026 | 0.9779 |

After exact recall was restored, discovery delta remained between -0.0018
and 0.0026 and recovery between 0.978 and 1.016. Increasing efSearch therefore
did not reveal a discovery breakdown hidden by exact-search difficulty. It
increased the evaluated-set Jaccard to about 0.890 at ef512 and 0.923--0.924
at ef1024.

## Geometric descriptors

Spearman correlations were calculated separately at each severity by pooling
the four replicates. No measured descriptor had a substantial association with
discovery delta. The largest absolute values outside IID were 0.0268 for the
raw `d10-d1` margin at mean alpha=0.75 and -0.0305 for query norm at radial
sigma=1.25. These exploratory correlations neither establish absence of a
non-monotone relationship nor support a causal interpretation.

## Correctness observations and confounders

- The four graph fingerprints, PQ-codebook hashes, and PQ-code hashes were
  distinct across replicates and invariant across conditions within each
  replicate. Instrumented and uninstrumented native outputs matched.
- The IID raw ground truth, native/oracle IDs, recall values, discovery delta,
  and Jaccard exactly reproduce Phase 1 robustness replicates 0--3.
- Sampled candidate sets were independently checked for uniqueness, counts,
  intersection/Jaccard, and coverage. Recall and all delta definitions were
  recomputed from raw IDs for every row.
- `delta_exact_control` was zero in 15,998 of 16,000 fixed-ef rows. Two rows
  had +0.1, giving an overall mean of 0.0000125 and a maximum of 0.1. The
  recorder includes upper-level greedy distance evaluations, whereas native
  results are emitted by level-0 search; an upper-level ground-truth node need
  not remain in the final heap. This creates a very small upward bias in the
  all-level oracle definition. It is retained, not silently corrected, and is
  far below the 0.03 emerging-evidence threshold. A future mechanism study
  should distinguish upper-level and level-0 evaluations explicitly.
- Standard-normal samples are reproducible for the recorded compiler and C++
  standard library; byte-for-byte reproduction is not asserted across all
  library implementations.
- This controlled family changes query mean or scale but leaves an isotropic
  Gaussian database. It does not test clustered or anisotropic database
  geometry.

## Answers to the Phase 2A questions

1. **No discovery degradation emerged under the tested shifts.** At every
   fixed condition, mean discovery delta stayed within 0.002 of zero, while
   exact reranking recovered about 99% of the exact/PQ gap.
2. There is therefore no observed emergence severity in the alpha or sigma
   grids.
3. The null discovery result is consistent across all four independently
   frozen indexes; no eligible condition met the emerging criterion in even
   one replicate.
4. Severe radial shift made exact HNSW harder. The predeclared efSearch sweep
   restored exact recall, but discovery delta still remained near zero.
5. Rerank recovery did not decrease systematically with severity. Its
   aggregate fixed-ef means stayed between 0.986 and 1.002; matched values
   stayed between 0.978 and 1.016.
6. None of the simple norm, centroid-distance, neighbor-distance, margin, or
   mean-direction-projection descriptors was meaningfully associated with
   discovery delta in this experiment; shifted-condition absolute Spearman
   values were at most about 0.031 (the IID maximum was 0.039).
7. These results do not justify full trajectory instrumentation yet because
   they provide no shift regime with a reproducible positive discovery signal
   to localize.

## Exactly one next experiment

Run **Phase 2B clustered/mixture database geometry** with IID in-distribution
queries, four pre-registered replicates, and the same fixed-topology
decomposition plus the same fixed-ef/matched-exact-recall control. This has
higher information gain than trajectory logging now: it tests whether
navigational discovery effects require non-isotropic graph geometry, while
changing one experimental factor and preserving the Phase 2A mechanism
measurements.

## Outputs

- Raw rows and manifests: `runs/phase2a_query_shift_v1/`
- The post-run `metadata_amendment.json` makes queue, ground-truth, and PQ
  training-size fields explicit; it records that no original raw row or
  manifest was modified.
- Fixed, matched, calibration, and correlation tables:
  `results/tables/phase2a_query_shift_*`
- Twelve requested severity and descriptor figures:
  `results/figures/phase2a_*`
