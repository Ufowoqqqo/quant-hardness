# Phase 5B — compression robustness

Status: all frozen primary experiments complete; no extra quantizer or
refinement depth was run. **Case A is supported for these measured aggregate
operating points**, with increased per-query discovery variation and historical
cross-phase timing explicitly retained as limitations.
See [frozen plan](../runs/phase5b_compression_robustness_v1/preregistration.md)
and [configuration](../configs/indexes/phase5b_compression_robustness.conf).

Reuse Phase5A vectors/GT/graph, train one PQ384x8 with the same65536 rows and
initialization seed. Only ef32/64/128,L16 and the specified1/8-worker check.
Candidate rank diagnostic is descriptive only, not L32/L64 refinement.
No new algorithm or uncertainty policy is authorized.

## Frozen inputs and implementation

All consumed Phase5A arrays, role IDs, exact GT/reference files, graph/PQ files
and historical result tables match the committed metadata. Source manifest
and preprocessing implementation hashes also match; no Arrow reprocessing or
new split occurs. Phase5B links the original normalized arrays and graph.
`input_verification.json` records all hashes and the pre-model timestamp.

Graph fingerprint:
`cd95032163d55937f2d4e90b1ab6e33d5509c938987d10a3c2cacc16d01b8f5d`;
graph file SHA-256:
`ddc7a331a1a664bf41ae67408b6e7e35321fbb47c62b64ace18c2c8d1eb36a2b`.
Training source-ID SHA-256:
`c348d26740f319d8061549e51b12b21d9c6fd09701c4a80782fabea24f208204`;
training normalized-array SHA-256:
`176b72ea78a75926add9f8aa915a3de70963f9e3a6f13d9e3f4e0627274a8235`.
GT IDs SHA-256:
`27d79de76e344ab0237dde665d38a0b400e5847e5ef4893823dbfc9f4b5fd495`.
GT was exhaustively generated for all10000 queries in5A and independently
validated using100 scalar FP64 L2/cosine reference queries; zero GT boundary
ties. It is reused, not recomputed or redefined.

The data are1536-dimensional held-out entity-to-entity OpenAI3 embeddings,
not natural-language-query retrieval. Original FP32-cast/FP64-normalize/FP32
archive policy is unchanged. Normalized squared L2 is used for exact graph,
GT, PQ ADC and refinement; reconstructed PQ vectors are not renormalized.
Graph M16,efConstruction80,seed20260907, original serialized topology.

Only the experiment driver architecture guard, PQ constructor and code stride
were generalized from768 to configurable384/768; graph/GT construction modes
are explicitly forbidden for5B. Candidate-retention and search implementations
remain byte-identical to5A source hashes. Native exact IDs are reverified
against5A for all queries/ef. Full-record, ordinary FAISS and bounded searches
serve correctness references; timed bounded mode does not record a full pool
or perform a second search. PQ-score ties retain first-evaluation order.

Seven CTests passed before PQ training, including both d1536/PQ384 andPQ768
fixture comparisons against full sorting and simple exact reranking. Two
additional Python tests check signed ratios and deterministic candidate-rank
tie/membership diagnostics. Fixtures do not train alternative experimental PQ
models. Source/binary/config/environment hashes are archived before execution.

## Frozen measurement and accounting

One standard PQ384x8,65536 frozen rows,seed95000037,25 iterations,1 redo,
24 training threads; FAISS20f14b31a6d54e243a3d1de6ae193fc4c3ec18ed.
No training subset resampling. Sanity pairs/order seed95000053,10000 samples,
exact same sampled identities as5A, with ADC-vs-decoded-L2 tolerance2e-5 and
sampled full code/ID checks. Poorer numerical quality is an outcome, not a
selection criterion. Primary ef32/64/128,k10,L16 only.

All per-query deltas are signed. Discovery=R_exact−R_oracle,
ranking=R_oracle−R_native; total is their sum. Oracle recovery=ranking/total
for positive denominator>1e-12; L16 recovery=(R_ALL16−R_native)/ranking for
positive denominator>1e-12. Undefined ratios are null, not zero. ef64/128 have
frozen high-quality exact controls; ef32 remains reported without retuning.

Single-thread: six cells,5 repetitions,40000 queries/cell,512 separate
training-row warmups/worker, seed95000089 randomized policy order. Then only
ef64 with1/8 workers,5 repetitions, order seed95000090. CPUs2..9 physical,
coordinatorCPU0; internal OpenMP/BLAS1; no nested parallelism. Actual query
wall times include ADC setup, traversal, retention, refinement/final selection.
QPS includes dispatch/completion overhead; load/train/hash/output I/O are outside
the timing region. Every timed native/final/top16 ID is checked against the
offline reference. Per-query latencies and every repetition are retained.

Hardware and compiler remain Phase5A's Intel i9-10920X,12 physical/24 logical
CPUs,one NUMA node,approximately30GiB RAM,GCC11.5.0 generic Release/O3 build.
Same compiler flags are checked; governor/turbo,CPU topology and load snapshots
are recorded anew, not assumed identical. No competing heavy agent work will
run during timed cells; other host jobs and frequency cannot be fully controlled.

990000 PQ codes use760320000B at768B/vector and380160000B at384B/vector.
The raw FP32 base remains6082560000B and the serialized graph including those
vectors6225374202B; its non-vector remainder142814202B is not a precise heap/RSS
measurement. Codebooks use1572864B in either PQ layout. ADC tables halve from
786432B to393216B/worker. ALL16 exact payload stays98304B/query. Deployment
still retains raw vectors for exact refinement: **16x is PQ-code payload
compression, not16x compression of total deployed index memory**. Report graph,
raw vectors,codes,codebook and worker scratch separately; no bandwidth causality
is claimed from these accounting estimates.

## Primary scientific observations

The one PQ384 model trained530.700027s and encoded36.558373s. Codebook SHA-256
`25b67a5fb81c84f25f2d61dcc16f29847dedb86b27e063653390da03044c5061`;
encoded base SHA-256
`883b61f52605d9ac3e2366aa77638ca12c9bea5673916c751914407ae0094aa2`;
serialized model SHA-256
`bfbe2bf9b612b66e3cafecaa12111f69de9e3ea2d12456e6c29e885d3cf34db4`.
No second model was trained.

| Sanity mean | PQ768 (8x) | PQ384 (16x) |
|---|---:|---:|
| Reconstruction L2 norm | .088727 | .295673 |
| Absolute relative distance error | .008617 | .047844 |
| Random pairwise inversion rate | .0254 | .0625 |
| ADC-vs-decoded exact reference error | 6.449e-7 | 4.729e-7 |

All sampled identities and exact scores match5A. Larger approximation error
is recorded as the intervention outcome, not used to substitute a quantizer.
Full median/p95/min/max values are in `phase5b_sanity_comparison.csv`.

| ef | PQ | Exact | Native | Candidate oracle | ALL16 | Discovery loss | Ranking loss | Oracle recovery | L16 recovery |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 768 | .89641 | .87758 | .89657 | .89655 | -.00016 | .01899 | 100.850% | 99.895% |
| 32 | 384 | .89641 | .83320 | .89709 | .89341 | -.00068 | .06389 | 101.076% | 94.240% |
| 64 | 768 | .94202 | .91685 | .94087 | .94084 | .00115 | .02402 | 95.431% | 99.875% |
| 64 | 384 | .94202 | .86500 | .94159 | .93606 | .00043 | .07659 | 99.442% | 92.780% |
| 128 | 768 | .96602 | .93799 | .96588 | .96585 | .00014 | .02789 | 99.501% | 99.892% |
| 128 | 384 | .96602 | .88098 | .96537 | .95864 | .00065 | .08439 | 99.236% | 92.025% |

Moving8x→16x increases total quantization loss by .04438/.05185/.05701 at
ef32/64/128. Ranking increases .04490/.05257/.05650; discovery changes
−.00052/−.00072/+.00051. Thus stronger compression's extra loss is overwhelmingly
ranking-side in this run. Negative discovery deltas and recoveries>100% are
retained: alternative search paths sometimes discover better useful sets.

Every query/ef passes native/full/bounded output identity, offline top16
extraction and unique L0 semantics. Recomputed exact IDs equal frozen5A IDs.
All30000 exact-control discrepancies are zero; no top16 boundary PQ-score ties
occur. Graph fingerprint remains the original. No tolerance recall was used.
The aggregate scientific thresholds H1/H2/H3 pass at high-recall ef64/128;
H3 also passes at ef32. This does not establish zero discovery loss per query
or robustness across independently trained codebooks.

The conversation usage interruption did not require redoing completed work:
all recall/audit artifacts were already complete on return. Only the remaining
systems stage was launched; no raw output was overwritten.

## Primary single-thread systems observations

All30 cells pass,1200000 timed queries. Means below average five repetitions;
QPS±SD is across repetitions. Median/p95/p99 are means of each repetition's
quantile. Full means/std/min/max and all individual repetitions are retained.

| ef | Policy | Recall | QPS ± SD | Mean µs | Median µs | p95 µs | p99 µs |
|---:|---|---:|---:|---:|---:|---:|---:|
| 32 | PQ384 native | .83320 | 1656.37 ± 1.13 | 603.50 | 599.13 | 702.52 | 772.11 |
| 32 | PQ384 ALL16 | .89341 | 1568.21 ± 4.64 | 637.48 | 628.71 | 763.44 | 832.04 |
| 64 | PQ384 native | .86500 | 1297.42 ± 1.67 | 770.51 | 771.38 | 906.53 | 989.64 |
| 64 | PQ384 ALL16 | .93606 | 1233.50 ± 13.02 | 810.56 | 809.67 | 968.84 | 1054.31 |
| 128 | PQ384 native | .88098 | 915.14 ± 1.85 | 1092.45 | 1101.41 | 1327.65 | 1402.28 |
| 128 | PQ384 ALL16 | .95864 | 892.29 ± 5.12 | 1120.51 | 1129.93 | 1361.25 | 1439.47 |

At ef32/64/128, ALL16 throughput penalty is5.323%/4.927%/2.497%, versus5A's
3.206%/1.779%/.706%. Mean latency overhead5.629%/5.198%/2.568%; p99 overhead
7.761%/6.534%/2.652%. Thus exact refinement becomes a larger relative cost
when PQ traversal is cheaper, but these measured recall/QPS points remain
useful. This is descriptive accounting, not isolated ADC/cache causality.

Among all12 observed points, the nondominated set is exactly:

| PQ | Policy | ef | Recall | QPS |
|---|---|---:|---:|---:|
| 384 | Native | 32 | .83320 | 1656.37 |
| 384 | ALL16 | 32 | .89341 | 1568.21 |
| 384 | ALL16 | 64 | .93606 | 1233.50 |
| 384 | ALL16 | 128 | .95864 | 892.29 |
| 768 | ALL16 | 128 | .96585 | 516.42 |

The other seven points are dominated in recall and mean QPS. This includes
all PQ768 native points and PQ384 native ef64/128. At the very highest observed
recall, PQ768 ALL16 ef128 still has value: PQ384 is not universally preferable.
No interpolated operating point is used. Error bars show within-phase repeat
variation, not uncertainty from separate benchmark dates or graph/PQ seeds.

Concrete observed comparisons:

| PQ384 ALL16 point | Reference native | Recall difference | QPS ratio | Strict observed dominance? |
|---|---|---:|---:|---|
| ef32 | PQ768 ef32 | +.01583 | 1.634x | Yes |
| ef64 | PQ768 ef64 | +.01921 | 1.667x | Yes |
| ef64 | PQ768 ef128 | -.00193 | 2.372x | No; nearby-recall tradeoff |
| ef128 | PQ768 ef128 | +.02065 | 1.716x | Yes |
| ef32 | PQ384 ef64 | +.02841 | 1.209x | Yes, within Phase5B |

Against the old shallow-refinement endpoint, PQ384 ALL16 ef128 has recall
.01780 above PQ768 ALL16 ef64 and QPS approximately1.228x higher. These remain
separate-phase measurements, not a contemporaneous randomized comparison.
The within5B dominance shows a tradeoff improvement without that cross-phase
timing confound, although it does not by itself isolate the8x→16x difference.

Estimated serialized-graph-plus-code/codebook payload is6987267066B forPQ768
versus6607107066B forPQ384, before query buffers, visited tables, allocation
overhead and other process state. Halving code bytes reduces this deployed
payload estimate by only about5.44%, not50%, because raw vectors remain.

## Limited ef64 concurrency observations

All20 cells and800000 timed queries pass. Values are mean±sample SD across
five repetitions; quantiles are means of repetition quantiles. Recall stays
.86500 native and .93606 ALL16 at both worker counts. This is a separate time
block; its1-worker mean is not substituted for the primary single-thread table.

| Workers | Policy | QPS ± SD | Mean µs | Median µs | p95 µs | p99 µs | Efficiency |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | Native | 1297.21 ± 2.68 | 770.65 | 771.53 | 907.27 | 989.59 | 100% |
| 1 | ALL16 | 1242.11 ± 2.22 | 804.88 | 804.02 | 961.34 | 1046.87 | 100% |
| 8 | Native | 9730.02 ± 5.26 | 818.11 | 818.95 | 958.77 | 1047.10 | 93.76% |
| 8 | ALL16 | 9285.84 ± 8.95 | 856.86 | 855.83 | 1024.18 | 1115.03 | 93.45% |

Throughput penalty4.247% at1 worker and4.565% at8; mean latency overhead
4.442%/4.737%; p95 overhead5.959%/6.822%; p99 overhead5.789%/6.488%.
Speedups7.501x/7.476x. Relative refinement overhead is greater than5A's8-worker
2.020% throughput and2.366% p99 overhead, but does not expand dramatically
with worker count. No new severe tail or concurrency problem is observed.
This is closed-loop service-time measurement, not open-loop queueing/SLA proof.

Host load snapshots range1.33–3.65; minimum available memory23012284KiB;
maximum host-wide iowait fraction.187%, with no free-swap change. Environment
counters include warmup and are not per-kernel profiling. Different host state
from5A and unlocked turbo/frequency remain cross-phase confounders. All runs
are retained, including the relatively more variable PQ384 ALL16 ef64
single-thread endpoint (QPS SD13.02). No best-run selection.

## Authorized candidate-rank diagnostic

Population: each exact candidate-oracle top10 member that also belongs to GT
top10. Fractions are item-weighted across queries, not fractions of queries.
Queries with no relevant oracle member contribute no items and are counted
explicitly. Scores are sorted ascending, ties by first L0 evaluation order.
All300000 oracle-member records, including GT-relevance flags, are preserved
in `candidate_oracle_item_ranks.csv`; no extra search/refinement is involved.

| ef | Relevant items | Zero-relevant queries | Within10 | Within16 | Within32 | Within64 | Max rank |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 89709 | 66 | 92.8781% | 99.5898% | 100% | 100% | 31 |
| 64 | 94159 | 15 | 91.8659% | 99.4127% | 99.9979% | 100% | 34 |
| 128 | 96537 | 3 | 91.2583% | 99.3029% | 99.9969% | 100% | 46 |

There are368/553/673 relevant oracle members outside top16. Their contribution
to strict recall is .00368/.00553/.00673, exactly the observed oracle−ALL16
residual gap. Independent stable-sort rank auditing verifies this equality
per query, not only on average. Thus L16 is sufficient by the frozen80%
recovery criterion, but is not perfect; its remaining truncation loss is
explained by useful evaluated candidates outside approximate top16. These
rank32/64 columns do not represent executed L32/L64 refinement results.

The supplementary all-oracle-member population has100000 items per ef and
top16 fractions99.168%/99.129%/99.113%; full distributions are in the tables.
These are different denominators from ranking-gap recovery92.0–94.2% and
must not be conflated.

## Important adverse per-query evidence

Average discovery robustness coexists with larger signed per-query variation:

| ef | PQ384 discovery positive | Zero | Negative | p99 | Range |
|---:|---:|---:|---:|---:|---:|
| 32 | 7.58% | 85.44% | 6.98% | .4 | [-1,1] |
| 64 | 4.56% | 91.53% | 3.91% | .2 | [-1,1] |
| 128 | 2.40% | 95.54% | 2.06% | .1 | [-1,.9] |

For comparison, positive/negative fractions underPQ768 at ef64 were2.04%/1.61%.
At ef64 the mean positive part of discovery loss increases .00391→.00769,
while the signed negative part changes−.00276→−.00726. Their sum is .00043.
At ef32 the correspondingPQ384 components are .01529 and−.01597; at ef128,
.00387 and−.00322. These components are diagnostics only: no signed per-query
metric was clipped or replaced. Thus the net-discovery gate passes partly
with cancellation; stronger compression is not harmless to every query's
candidate discovery. All extreme cases remain in raw output. This is not
evidence of a stable hard-query class without replication, nor a reason to
resume uncertainty prediction in this phase.

## Decision and limits

H1 passes at ef64/128: oracle recovers99.442%/99.236% of total loss.
H2 passes: ranking increases by .05257/.05650, versus discovery changes
−.00072/+.00051. H3 primary80% and secondary60% thresholds pass at all three
points. H4's direction is resolved empirically: PQ384 ALL16 adds useful
measured frontier points despite increased relative refinement overhead.

**Case A — robust precision-placement principle at these measured points**
is the best description. Cases B/C/D are not supported as aggregate outcomes:
L16 still recovers92–94% of ranking loss, oracle recovery remains near100%,
and new frontier points exist. This does not establish universal navigability,
an optimal depth, or robustness at yet stronger compression. PQ768 ALL16
ef128 retains the highest-recall endpoint. Only one PQ384 seed, one graph,
one embedding workload and one hardware platform were tested. Cross-phase
historical timing does not establish causal speedup from compression alone.

The combined5A/5B evidence is sufficient to **proceed to baseline comparison**,
not to claim a new state of the art or paper-level novelty. An optimized,
graph/quantizer-co-designed baseline may outperform these generic FAISS paths.

**Exactly one next experiment, not run:** a preregistered same-machine,
contemporaneously interleaved end-to-end comparison of PQ384 native/ALL16
against a pinned author implementation of SymphonyQG on this same normalized
DBPedia split and GT. Use each system's documented native graph, explicit
matched-ISA/compiler treatment, observed Recall–QPS curves and full resident
index-memory accounting; predeclare settings before outcomes. Label this a
systems baseline comparison, not a same-topology causal ablation. It directly
tests whether the within-FAISS frontier gain is competitive with a co-designed
quantized graph system. No SymphonyQG run, alternative PQ or L sweep has begun.

## Baseline context inspected (not a Phase5B experiment)

The [SymphonyQG paper](https://arxiv.org/abs/2411.12229) describes jointly
designing graph/quantization behavior around FastScan and avoiding a separate
reranking stage. The [authors' repository](https://github.com/gouyt13/SymphonyQG)
states an AVX512 requirement and points to its implementation in RaBitQ-Library;
the original repository is archived. These sources were inspected only to
ground the requested final baseline-comparison decision; no baseline was
downloaded, built or benchmarked in Phase5B. Our within-FAISS fixed-graph
mechanism test cannot establish superiority over that different system.
Any future head-to-head comparison needs explicit ISA/compiler and total-memory
accounting, and must distinguish an algorithm's native graph from the fixed
topology control. No novelty claim follows from current repository experiments.

## Outputs and reproducibility

Raw root: `runs/phase5b_compression_robustness_v1/`, including input checksum
verification, preregistration/config hashes, model provenance, complete L0
candidate IDs/PQ scores, all query result IDs/metrics,300000 oracle-item rank
rows,50 benchmark cells with per-query latencies and result IDs, environment
snapshots and independent audits. Reused prepared vectors/graph are symlinks
to original5A artifacts, not duplicate or regenerated data. The large trained
PQ index is retained locally; its checksum is tracked. No raw result was deleted.

Code commit at execution is541a0eb8c812bf9926f0bc5de3699d908dfa20a4; Phase5B
implementation changes are uncommitted at run time, explicitly marked dirty
with source/binary hashes. The parent commit alone does not contain the new
runner. `execution_provenance.json` and `analysis_freeze.json` identify the
actual code; later doc/audit additions do not change the timed implementation.

Primary commands, expanded logs and environment are in `execution_logs/` and
[the experiment log](experiment_log.md):

```bash
cmake --build build --target faiss_phase5a faiss_phase5a_bench phase5a_retention_correctness -j 4
/rwproject/kdd-db/kluaq/miniconda3/envs/build_env/bin/python -u scripts/prepare_phase5b.py
/rwproject/kdd-db/kluaq/miniconda3/envs/build_env/bin/python -u scripts/run_phase5b.py --stage model
/rwproject/kdd-db/kluaq/miniconda3/envs/build_env/bin/python -u scripts/run_phase5b.py --stage recall
/rwproject/kdd-db/kluaq/miniconda3/envs/build_env/bin/python -u scripts/run_phase5b.py --stage systems
```

These run stages refuse raw-output overwrite. To regenerate only analysis:

```bash
MPLCONFIGDIR=/tmp/phase5b-mpl-cache /tmp/phase3b-plot-env/bin/python scripts/analyze_phase5b.py --stage final --tables /tmp/phase5b-regeneration-KkuxnE/tables --figures /tmp/phase5b-regeneration-KkuxnE/figures
```

The existing plot environment uses NumPy1.23.5/Matplotlib3.9.4; input/audit
environment uses NumPy1.26.4. No learned model or additional quantizer library
is introduced. Separate environment-summary regeneration and byte comparison
commands are recorded in the experiment log. All34 generated CSV/JSON/PNG
artifacts match their fresh regeneration exactly. Independent timed-ID audit
checks14 frozen files and all2000000 queries, with all50 cells passing.
Final post-run control audit also passes: all51 frozen input/reference hashes
are unchanged, exact controls match5A byte-for-byte, exact/PQ oracle recalls
equal candidate coverage for every query, and all300000 item ranks match an
independent stable-sort implementation. Per-query ALL16 residual recall loss
equals the count of relevant oracle items outside top16 divided by10.
See `postrun_controls.json`. All experiment and verification processes have
completed; no new experimental phase was launched.

Required figures:

1. [8x/16x Recall–QPS families](../results/figures/phase5b_compression_frontier.png)
2. [Exact/native/oracle/ALL16 recall](../results/figures/phase5b_recall_decomposition.png)
3. [Discovery versus ranking loss](../results/figures/phase5b_discovery_ranking.png)
4. [Candidate-oracle recovery](../results/figures/phase5b_candidate_oracle_recovery.png)
5. [L16 recovery](../results/figures/phase5b_L16_recovery.png)
6. [Measured Pareto points](../results/figures/phase5b_pareto.png)
7. [Relevant-oracle-item PQ-rank CDF](../results/figures/phase5b_candidate_rank_cdf.png)
8. [1/8-worker QPS](../results/figures/phase5b_concurrency.png)

Tables: `results/tables/phase5b_{recall_comparison,loss_transition,
per_query_distributions,sanity_comparison,memory_accounting,
benchmark_repetitions,endpoints,overhead,combined_frontier,observed_comparisons,
candidate_ranks,scaling,environment}.{csv,json}`. All prior5A measurements remain
unchanged. No PQ256/PQ512,alternative seed,OPQ,L32/L64 refinement,additional ef,
HNSW construction change or new dataset was run.
