# Phase 5A — high-dimensional external-validity gatekeeper

## Primary execution — 2026-09-10

All frozen primary measurements and final independent audits are complete.
**Decision: Case A — scientific + systems external validity in this measured
workload/configuration.** This is not a claim of universality across embeddings,
compression ratios, hardware, or individual queries. No exploratory experiment
was run after the primary results.
No additional PQ, efSearch, L, normalization, or learned-policy condition is
permitted in this run. Historical acquisition notes below are retained, not
current prerequisites or current claims of non-execution.

### Frozen workload and reproduction

The binding pre-retrieval amendment is commit
`fbdf4784a505dfbf201be9e78f788e67d31fd4b2`. Dataset:
`Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M`, revision
`4a9731217921bc476a0f03544f11f22ae4903fa5`. This is held-out
**entity-to-entity nearest-neighbor search**, not natural-language-query
retrieval. All26 Arrow shards,1M rows and1536 dimensions pass validation.
Their local paths/checksums and exact role reconstruction are recorded in
[the amendment](phase5a_dataset_amendment.md) and the run's `amendment/` manifest.

PCG64 split seed20260909 permutes all source row IDs: first990000 form the
base, last10000 the evaluation queries. Training seed95000011 samples65536
base rows only, in the frozen order. Queries are excluded from both base and
training; all training rows exactly match their archived base positions.
The frozen role-ID hashes are:

| Role | SHA-256 |
|---|---|
| Base | `0b65c6b76ebcb691fdfe199eb0c642ee556f0878f01f17a1006d6b0d16e36a40` |
| Query | `4bc58919e88432cfa96f8450ed52616ffb95585a486977d9edf39ff8bb0f8797` |
| Training | `c348d26740f319d8061549e51b12b21d9c6fd09701c4a80782fabea24f208204` |

Preprocessing is unchanged: cast source embeddings to FP32 FIRST, compute
norm and division in FP64 from those FP32 values, then cast once to archived
FP32. Every experiment reads the same arrays. All values are finite; maximum
unit-norm error is2.995396664040584e-08. Squared L2 on these normalized vectors
is the cosine-equivalent exact ranking convention. PQ ADC approximates that
same squared L2; decoded PQ vectors are not renormalized. Minor finite-precision
norm deviations are covered by the independent cosine/L2 validation below.

The experiment parent Git commit is the amendment; new implementation was
uncommitted at execution. `execution_provenance.json` records every executed
source/binary hash, compiler flags, pinned FAISS revision, environment and
physical-core assignments. This is not a claim that fbdf478 alone contains
the subsequently implemented runner. FAISS revision:
`20f14b31a6d54e243a3d1de6ae193fc4c3ec18ed`.

### Exact ground truth

All10000 queries were exhaustively evaluated against990000 base vectors
using `faiss::fvec_L2sqr` FP32. Top11 are archived, first10 defining strict
ID-intersection Recall@10; rank11 is a tie diagnostic. Equal exact scores use
ascending base ID. Ground-truth ID SHA-256:
`27d79de76e344ab0237dde665d38a0b400e5847e5ef4893823dbfc9f4b5fd495`.
Kernel elapsed time551.1513s; no ANN approximation enters these labels.

The latest user requirement expands the old configuration's32 validation
queries to100 deterministic samples (seed95000071), without changing any
data/model parameter. Separate exhaustive scalar FP64 L2 and NumPy FP64 cosine
implementations both reproduce top10 membership on all100. Maximum primary
FP32-vs-FP64 distance difference4.811764e-07, below the frozen2e-5 tolerance.
There are zero primary FP32 rank10/rank11 ties across all10000 queries.
Tolerance is only a validation diagnostic; recall remains strict ID recall.
Details: `gt_validation.json`.

### One graph, one quantizer

One exact FP32 HNSW graph: M16, efConstruction80, seed20260907,24 construction
threads, one graph.add call. Build time1243.666647s; serialized size6225374202B.
Runner stage2752.7s also includes serialization/hash/I/O and is not build time.
Parallel graph construction is not guaranteed bit-reproducible from seed alone;
the frozen serialized graph is retained for exact replay.

Graph fingerprint:
`cd95032163d55937f2d4e90b1ab6e33d5509c938987d10a3c2cacc16d01b8f5d`.
File SHA-256:
`ddc7a331a1a664bf41ae67408b6e7e35321fbb47c62b64ace18c2c8d1eb36a2b`.

One PQ768x8, dsub2,768B/vector versus6144B FP32, compression8x. Frozen65536
training rows, seed95000037,25 iterations,1 redo,24 training threads; training
799.073615s, encoding70.626760s. Database codes preserve graph vector-ID order.
Codebook SHA-256:
`78c2dbd08ae1deabd177022609382ff56e1f3928c718419eb2be5d211ae035e0`;
codes SHA-256:
`a50637f62259fb08ed3425868194ef9d42aaeb3dfba73a8b1b5fe7f1fb85f51b`.

Pre-ANN deterministic sanity sampling (10000 pairs/order comparisons,
seed95000053): mean absolute relative distance error0.00861728, median0.00847855,
p950.01607440; reconstruction L2 norm mean0.08872676, median0.08829002,
p950.09582436; random strict pairwise inversion rate0.0254. ADC versus independent
decoded-vector L2 maximum error2.980232e-06 (<2e-5), with sampled full-code ID
alignment verified. These diagnostics selected no parameters.

### L0 semantics and correctness

The same frozen topology, entry state, levels and neighbor lists are used at
ef32/64/128. Only exact versus PQ distance computation changes. Full candidate
oracle uses unique L0 distance-evaluated IDs, never upper-level-only IDs.
Bounded retention keeps16 unique PQ-best L0 candidates during the same search,
using the validated score/first-evaluation-ordinal tie convention, then exactly
reranks those16. It neither records the full pool online nor performs a second
search. Full recording exists only in the offline reference/control path.

For every query/ef: native IDs and scores match ordinary FAISS; bounded top16
IDs/scores match full-record extraction and independent offline extraction;
bounded final IDs match full-record ALL16; graph fingerprint remains unchanged.
All30000 exact-control discrepancies are zero. No top16-boundary PQ ties occur.
Every candidate list is unique; coverage and ID-based recalls independently
recompute. Seven CTests and three frozen split/preprocessing tests passed before
benchmarking. Full raw lists and audit results are preserved.

### Primary recall results (observations)

| efSearch | Exact | PQ native | Candidate oracle | Bounded ALL16 | Discovery loss | Ranking loss | Oracle recovery | L16 recovery |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | .89641 | .87758 | .89657 | .89655 | -.00016 | .01899 | 100.85% | 99.895% |
| 64 | .94202 | .91685 | .94087 | .94084 | .00115 | .02402 | 95.43% | 99.875% |
| 128 | .96602 | .93799 | .96588 | .96585 | .00014 | .02789 | 99.50% | 99.892% |

Discovery=Exact−Oracle, ranking=Oracle−PQ, total=Exact−PQ. Oracle recovery
is ranking/total; L16 recovery=(ALL16−PQ)/ranking. All denominators are positive.
No signs or ratios are clipped. The slightly better PQ candidate oracle at
ef32 reflects different candidate discoveries, not better exhaustive truth;
zero exact-control discrepancies and no GT boundary ties were observed.
ef32 exact recall .89641 is slightly below .90; high-recall gate interpretation
rests primarily on ef64/128. Their oracle recovery passes preregistered H1≥90%.
H2≥60% L16 recovery passes at all three points. Rankings account for most
observed loss; this does not imply per-query discovery effects never occur.
Signed per-query distributions, counts and raw IDs remain available, not just
means. These are one-index/one-PQ outcomes, not multi-seed confidence claims.

An important limitation of the mean recovery criterion is cancellation.
Discovery loss is positive/zero/negative for3.60%/92.72%/3.68% of ef32 queries,
2.04%/96.35%/1.61% at ef64, and1.03%/98.07%/.90% at ef128. Individual discovery
loss ranges from−1 to+1 at ef32/64 and−.6 to+.8 at ef128. Thus average discovery
robustness is supported, not universal per-query navigability preservation.
Signed raw cases are retained and no adverse cases were removed. This primary
run does not investigate or optimize those rare path differences.

### Systems measurement protocol

Primary endpoints: native and bounded ALL16 at ef32/64/128, one worker.
Five measured repetitions per endpoint; each has four complete10000-query
passes and512 separate warmups drawn from frozen training rows, not evaluation
queries. Policy/ef order is deterministically shuffled within each repetition
(seed95000089). After single-thread analysis, only ef64 is measured with1/8
workers, five repetitions, order seed95000090. No optional policy is measured.

Each worker runs the complete query path on a distinct physical core (CPUs2..9,
coordinatorCPU0), with private retention state and FAISS visited TLS. OpenMP,
OpenBLAS/MKL/BLIS are limited to1; live thread count must equal workers+1.
Actual per-query wall-clock times include ADC setup, traversal, bounded heap,
exact16 distances and final top10 where applicable. QPS includes dispatch,
result-copy and worker completion overhead; it is not inferred from latency
quantiles. Loading/training/construction/hash checks are outside timed regions.
Native/final/top16 IDs are checked against validated offline references on
every pass; checks, raw-file writes and index hashing happen outside timing.

Hardware: Intel Core i9-10920X,12 physical/24 logical CPUs, one NUMA node,
approximately30GiB RAM,1MiB L2/core and19712KiB shared L3. Workers use physical
cores, not SMT siblings. Compiler GCC11.5.0, Release `-O3 -DNDEBUG`, C++20,
OpenMP, existing generic FAISS CPU build. Exact flags, governor/turbo values,
topology and caches are preserved in `execution_provenance.json`; no new
dataset-specific kernel optimization was introduced. Host load and CPU/memory
counters are captured around every cell. The command sandbox cannot enumerate
all host jobs, so exclusive host ownership cannot be guaranteed.
Recorded CPU2 governor is `performance`; `intel_pstate/no_turbo=0` (turbo enabled).
Frequency is not locked. CPU2..9 sibling lists are2/14 through9/21; selected
CPUs have distinct physical core IDs. No measured worker uses an SMT sibling
of another measured worker.

ALL16 accesses up to16×1536×4=98304B (96KiB) exact vector payload/query versus
8192B (8KiB) on SIFT:12x more. Each PQ ADC table is768×256×4=786432B (768KiB),
also12x the SIFT PQ64 table. Both endpoints pay their real ADC/traversal costs;
bounded refinement carries its own retention cost. Payload estimates are not
measured DRAM traffic, and no bandwidth bottleneck is inferred without counters.

### Single-thread results (five repetitions)

All30 benchmark cells pass,1200000 timed queries total. Values below are means
of each repetition's statistic, not best-run values; QPS uncertainty is sample
SD across five repetitions. Full mean/std/min/max for every latency statistic,
including p90, are in `phase5a_endpoints.csv`; every repetition is separately
listed in `phase5a_benchmark_repetitions.csv`.

| ef | Policy | Recall | QPS ± SD | Mean µs | Median µs | p95 µs | p99 µs |
|---:|---|---:|---:|---:|---:|---:|---:|
| 32 | Native | .87758 | 959.64 ± 1.35 | 1041.82 | 1035.03 | 1223.87 | 1353.57 |
| 32 | ALL16 | .89655 | 928.88 ± 4.75 | 1076.37 | 1068.55 | 1274.94 | 1401.33 |
| 64 | Native | .91685 | 740.08 ± 2.78 | 1350.96 | 1353.91 | 1604.42 | 1744.34 |
| 64 | ALL16 | .94084 | 726.92 ± 1.41 | 1375.46 | 1378.82 | 1636.37 | 1777.76 |
| 128 | Native | .93799 | 520.09 ± 4.10 | 1922.58 | 1941.50 | 2366.32 | 2494.87 |
| 128 | ALL16 | .96585 | 516.42 ± 3.50 | 1936.25 | 1955.21 | 2379.59 | 2507.83 |

| ef | ALL16 throughput penalty | Mean latency overhead | p95 overhead | p99 overhead |
|---:|---:|---:|---:|---:|
| 32 | 3.206% | 3.317% | 4.172% | 3.529% |
| 64 | 1.779% | 1.813% | 1.991% | 1.916% |
| 128 | .706% | .711% | .561% | .519% |

The smallest relative effects should be read alongside repetition variation;
these are descriptive ratios, not paired significance claims. No run was
discarded. At ef64, absolute mean increment is24.50µs, versus approximately
12.39µs in SIFT4D. The high-dimensional baseline is also much slower, so a
larger exact payload does not imply a larger *relative* end-to-end penalty.
The current measurements do not isolate bandwidth or ADC setup causality.

Observed, non-interpolated frontier comparison: ALL16 ef64 versus native ef128
has Recall+.00285 and QPS ratio1.39767. Native ef128 is dominated among the six
measured points. The other five points are nondominated. ALL16 ef32 versus
native ef64 instead trades Recall−.02030 for QPS ratio1.25510; it is not a
matched-recall superiority claim. ALL16 ef32 versus native ef128 trades
Recall−.04144 for QPS ratio1.78600. Connecting points in figures is visual only.
H3 receives an observed frontier improvement, not merely improvement at fixed
efSearch. No interpolation, extra ef point or post-result tuning was used.

### Limited ef64 concurrency results

All20 cells pass,800000 timed queries. This is a separate time block from the
primary six-point benchmark; its1-worker result is not silently substituted
for the primary single-thread measurements. Recall remains .91685 native and
.94084 ALL16 at both concurrency levels.

| Workers | Policy | QPS ± SD | Mean µs | p95 µs | p99 µs | Parallel efficiency |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Native | 735.43 ± 18.44 | 1360.16 | 1626.33 | 1769.45 | 100% |
| 1 | ALL16 | 719.97 ± 23.44 | 1389.92 | 1672.42 | 1821.58 | 100% |
| 8 | Native | 5591.49 ± 11.27 | 1419.02 | 1677.60 | 1826.64 | 95.04% |
| 8 | ALL16 | 5478.56 ± 19.81 | 1446.59 | 1712.33 | 1869.85 | 95.12% |

Throughput penalty is2.103% at1 worker and2.020% at8; mean latency overhead
2.188%/1.943%, p95 overhead2.834%/2.070%, p99 overhead2.946%/2.366%.
Speedups are7.603x native and7.609x ALL16. There is no observed widening
refinement-specific scalability gap or disproportionate tail problem through
eight physical workers. This is a closed-loop service-time test, not an
open-loop queueing/SLA test or an overload test.

The concurrency time block has visible1-worker drift: native QPS706.93–755.45,
ALL16 QPS681.32–742.08; repetition3 is slower for both. Its ALL16 p99 is2005.21µs,
versus1875.70µs native. No repetition was excluded or rerun selectively.
Eight-worker QPS is narrower: native5583.43–5610.98, ALL165450.30–5499.48.
Frequency/host effects cannot be distinguished from these measurements alone.

Across50 before/after environment snapshots,1-minute host load spans2.0–10.5,
minimum available memory21.44GiB, maximum host-wide iowait fraction4.91% over
the captured cell interval (which includes warmup). Four cells show+256KiB
free-swap changes; none shows a decrease. These are whole-host counters, not
proof of benchmark paging or a refinement bandwidth bottleneck. All snapshots
and descriptive environment tables are retained; exclusive host ownership and
fixed frequency were not guaranteed. The small percentage timing differences
are engineering observations, not publication-quality inference.

### Cross-dataset comparison

The systems rows below use ef64, existing generic CPU implementation, and
one-worker measurements. No adverse difference is normalized away.

| Quantity | SIFT1M Phase4B/C/D stream | DBPedia/OpenAI3 Phase5A |
|---|---:|---:|
| Workload | Image descriptors | Held-out entity embeddings |
| Dimension | 128 | 1536 |
| Metric | Squared L2 | Normalized squared L2 / cosine equivalent |
| PQ | 64x8 | 768x8 |
| Dimensions/subquantizer | 2 | 2 |
| FP32 compression | 8x | 8x |
| Exact native recall | Not measured on this cohort | .94202 |
| PQ native recall | .850309 | .916850 |
| Candidate oracle recall | .950365 | .940870 |
| Discovery loss | Not measured on this cohort | .001150 |
| Ranking loss | .100056 | .024020 |
| Candidate-oracle recovery | Not measured on this cohort | 95.431% |
| ALL16 recall | .939973 | .940840 |
| L16 recovery | 89.614% | 99.875% |
| Native QPS | 6118.50 | 740.08 |
| ALL16 QPS | 5687.78 | 726.92 |
| Throughput penalty | 7.040% | 1.779% |
| p99 overhead | 8.833% | 1.916% |

### Cross-dataset comparability constraint

SIFT Phase4B/C/D uses14252 queries, while Phase4A's scientific exact control
uses a different10000-query cohort. The systems comparison table uses matching
4B/C/D native/ALL16/oracle measurements and leaves unmeasured exact/discovery
entries null. It must not borrow Phase4A exact recall and imply it was measured
on the4D stream.

For clearly separate scientific context, Phase4A exact=.952760, PQ=.850560,
candidate oracle=.951390, ALL16=.941020. Under the Phase5A native-exact-based
formula this gives discovery=.001370 and ranking=.100830. The original
Phase4A discovery=.000930 instead used its exact-L0-oracle=.952320 baseline;
these definitions are not silently mixed. Phase4A has no wall-clock benchmark.

### Hypotheses and interpretation

**H1 passes at ef64/128:** full PQ-candidate reranking recovers95.43%/99.50%
of exact-vs-PQ loss; discovery is much smaller than ranking in aggregate.
ef32 is a lower-exact-recall context, not the main high-recall support.
**H2 passes:** L16 recovers99.875–99.895% of the ranking gap.
**H3 passes on observed points:** ALL16 ef64 dominates native ef128 in both
recall and QPS. **H4's possible larger relative overhead is not observed:**
payload and central absolute increment are larger, but relative end-to-end
overhead is smaller than SIFT. This is not a contradiction of higher-dimensional
exact-distance work; both workload and baseline search cost changed.

Case A is best supported at this frozen8x-compression/dsub2 operating regime.
The evidence favors high precision for final selection while PQ traversal
preserves most useful discoveries on average. It supports further external
validation of bounded shallow refinement; it does not establish novelty or
make a publication-ready multi-dataset/multi-compression claim. Native quality
is already relatively high here, and the recoverable gap is only .019–.028,
substantially smaller than SIFT. Rare signed discovery failures, one graph/PQ
realization, entity-to-entity queries, cache-warm repeated streams, one CPU and
uncontrolled host/frequency variation limit generalization.

**Exactly one next experiment (not run):** a preregistered compression-stress
replication on this identical DBPedia split/graph, changing only to standard
PQ384x8 (dsub4,16x compression), keeping the same training rows/seed,
ef32/64/128 and L16, and repeating the same decomposition/Recall-QPS endpoints.
Use current PQ768x8 as the frozen reference. This tests whether discovery
robustness and almost-complete shallow recovery depend on the unusually fine
two-dimensional subspaces shared by both successful workloads. It is one
controlled compression test, not a quantizer sweep or a new algorithm.

### Output and audit completion

Final independent audit: all2000000 timed query outputs across50 cells match
offline references;24 frozen implementation/config/binary files and complete
graph/PQ file hashes remain unchanged. All30,000 recall rows and candidate
controls pass. No exact-control or bounded-retention discrepancy remains.
Fresh raw-data regeneration produces byte-identical29 CSV/JSON/PNG artifacts.
See `final_independent_audit.json` and `analysis_regeneration.json`.

Primary figures:

- [Recall–QPS frontier](../results/figures/phase5a_recall_qps.png)
- [Four-way recall](../results/figures/phase5a_recall_decomposition.png)
- [Discovery versus ranking loss](../results/figures/phase5a_discovery_ranking.png)
- [Oracle and L16 recovery](../results/figures/phase5a_recovery.png)
- [Relative latency/throughput overhead](../results/figures/phase5a_overhead.png)
- [Cross-dataset recall](../results/figures/phase5a_cross_dataset.png)
- [1/8-worker QPS](../results/figures/phase5a_concurrency.png)

Primary tables: `results/tables/phase5a_{recall,per_query_distributions,pq_sanity,
benchmark_repetitions,endpoints,overhead,observed_frontier,observed_cross_ef_pairs,
scaling,cross_dataset,environment}.{csv,json}`. Raw measurements remain in the
requested run root; no prior run is overwritten. The primary phase is stopped:
no L8/32/64, alternative PQ, OPQ, extra efSearch, new training size or learned
model has been evaluated.

### Reproduction and retained artifacts

Configuration: `configs/indexes/phase5a_highdim_external_validity.conf`,
unchanged from the amendment. Raw root:
`runs/phase5a_highdim_external_validity_v1/`. It contains prepared arrays,
role/source manifests, GT, graph, PQ, sanity pairs, all per-query recalls and
candidate IDs/scores, benchmark schedules/per-query timings/returned IDs,
environment snapshots and gate outputs. No raw results were deleted.

Run commands (expanded child commands/environment in `execution_logs/`):

```bash
/rwproject/kdd-db/kluaq/miniconda3/envs/build_env/bin/python -u scripts/run_phase5a.py --stage offline
/rwproject/kdd-db/kluaq/miniconda3/envs/build_env/bin/python -u scripts/run_phase5a.py --stage systems
```

Execution refuses to overwrite existing raw runs. Analysis is separately
regenerable from raw outputs with `scripts/analyze_phase5a.py --stage final`,
using the existing plotting environment. The experiment log records preparation
I/O recoveries: partial direct-NFS output preserved, same arithmetic completed
on NVMe and copied/checksummed; stalled child-Git metadata recovered by direct
Git-dir lookup. Neither changed vectors or preregistered scientific parameters.

## Archived 2026-09-09 amendment checkpoint

The remainder is historical acquisition/pre-run documentation. Its statements
of pending work are superseded by the completed primary results above; its
old ada-002 roles are superseded by the binding amendment.

The user approved the locally cached Qdrant text-embedding-3-large1536 version
instead of ada-002, before any ANN results. See
[binding amendment](phase5a_dataset_amendment.md) for the990000/10000 split,
base-only PQ training and cast-first normalization. The source/role checksum
manifest is under `runs/phase5a_highdim_external_validity_v1/amendment/`.
The following acquisition narrative is historical; no further ada-002
download is required for the amended experiment. Scientific/system gates
remain unevaluated. This workload is held-out entity-to-entity NN.
The full26-shard local validation now passes:1000000 rows, dimension1536,
and frozen990000 base/10000 query/65536 base-only training IDs. Three split/
preprocessing unit tests and full-size deterministic ID regeneration pass.
At that checkpoint, no GT/PQ training/graph build/ANN evaluation had run.

## Historical 2026-09-08 preflight

Status on 2026-09-08: **not run; no validated complete dataset available**.
The design is preregistered in
[`preregistration.md`](../runs/phase5a_highdim_external_validity_v1/preregistration.md)
and [`configuration`](../configs/indexes/phase5a_highdim_external_validity.conf).
No scientific or systems gate has been evaluated. No previous run is changed.

## Historical sources inspected

- [Upstream DBPedia/OpenAI dataset](https://huggingface.co/datasets/KShivendu/dbpedia-entities-openai-1M),
  revision `af9b8869cc2d8debbd254d77737865bb09a2067f`: source metadata describes
  one million 1536-dimensional text-embedding-ada-002 vectors. It exposes
  26 Parquet files and a train split, not a separate official query split.
- [ANN-Benchmarks generator](https://github.com/erikbern/ann-benchmarks/blob/main/ann_benchmarks/datasets.py)
  defines a deterministic 10,000-query split for this source. Generator
  inspection is not proof of the downloaded artifact's counts.
- [Qdrant benchmark registry](https://github.com/qdrant/vector-db-benchmark/blob/master/datasets/datasets.json)
  names `dbpedia-openai-1M-1536-angular`, cosine, with
  [this GCS archive](https://storage.googleapis.com/ann-filtered-benchmark/datasets/dbpedia_openai_1M.tgz).
  HTTP metadata: 5,098,686,329 bytes, generation1699276546881421,
  MD5 `dc5fecd77592b669643a5e1ea0541887`. A full transfer failed after
  15,113 bytes with curl92 (HTTP/2 stream closed). The partial file remains
  preserved under `runs/.../source/`; it is not an experimental input.
- The original ANN-Benchmarks HDF5 URL returned404. A matching
  [HDF5 mirror object](https://storage.googleapis.com/ann-datasets/ann-benchmarks/dbpedia-openai-1000k-angular.hdf5)
  returns200: 6,160,008,192 bytes, generation1695799664928427,
  MD5 `02bc868fc7c77442ef8eaad14c93daf3`.
  A 1MiB range downloaded in12.4s (~85KB/s); this does not validate the full
  object. An HTTP/1.1 attempt failed with curl35 TLS EOF. Whole-file SHA-256,
  content checks and GT remain pending. Prefix-only HDF5 metadata inspection
  confirms `train=(990000,1536)`, `test=(10000,1536)`, FP32, angular. Neighbor
  and distance object metadata are outside the downloaded prefix. This is
  explicitly not validation of any vector data.
  A full HDF5 transfer probe received2,867,200 bytes in30s, then stopped at
  the deliberately configured timeout (curl28), rather than a server error.
  At that measured rate the6.16GB transfer would take about18h; this is an
  estimate, not proof the dataset is inaccessible. No download is left running.

## Historical pre-amendment decisions and unresolved prerequisites

Primary PQ768×8 is selected arithmetically, not by recall: 1536×4=6144B
FP32, 768B codes, 8× compression, subvector dimension2. Standard FAISS permits
this layout; performance and training feasibility are not yet measured.
Its 768KiB ADC table/query must count toward online cost. The machine has
approximately27GiB available RAM, enough to attempt memory-conscious loading
of this scale, not a guarantee that an unbounded loader fits.

Reserve65,536 official base-split IDs exclusively for training and keep all
official queries. This explicitly reduces the benchmark base; all GT must
be recomputed. Counts in the preregistration are expectations, not verified
measurements. Freeze actual role-ID checksums before training. No test query
will select parameters. All scoring uses archived unit-normalized FP32 L2.

The former ef64-only guard needs a small, reference-tested generalization;
no search code has been modified yet. Do not interpret the preflight as
completion of loader, bounded-retention, concurrency or GT tests.

## Historical preflight decision (superseded)

Questions about discovery robustness, ranking dominance, L16 recovery,
Recall/QPS frontier and paper-readiness are **unanswered** for this dataset.
Neither Case A/B/C nor a scientific next-experiment recommendation is justified
without the requested primary measurements. Required next operational step:
obtain a complete checksum-verified high-dimensional artifact through a
working transfer or a supplied local path, then execute the frozen design.
