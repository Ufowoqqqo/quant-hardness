# Phase 4A preregistration — before held-out outcomes

Primary: raw PQ ADC rank-12 minus rank-9 gap, ascending score; smaller
is higher risk. Normalized gap is secondary only. No label-based changes,
calibration, training after results, new traversal instrumentation or timing benchmark.

Use the fixed-revision SIFT learning file and PCG64 split_seed=94000011.
Filter exact vector-content duplicates of historical standard queries and
base vectors, and deduplicate learning vectors deterministically by first ID.
The base exclusion is deliberately conservative: prevents self matches and
overlap with any historical base-trained PQ model. Record excluded IDs/counts.
Permute remaining IDs once: first 65536 train, next 10000 test; fail if
insufficient. Store complete original learning IDs and SHA256. No query
labels are inspected to define this split. Train ONE standard PQ64x8 on
these learning training vectors (seed 94000037, 25 iterations, 24 threads).
This fixes identical-source disjoint roles, rather than retaining an old
base-trained model. Encode all base IDs in original order. No PQ selection.
Retain existing FP32 HNSW graph exactly, M16/efConstruction80; efSearch64,k10.

Only actual PQ L0 evaluated candidates define every refinement pool. Exact
traversal is a separate discovery/control measurement, never a policy input.
Record native IDs/scores, PQ candidates/scores and offline exact distances.
Compare instrumented and native searches bit-for-bit. Exact-control or
ground-truth anomalies must be investigated before interpretation. Exhaustive
IndexFlat FP32 GT is independently checked on 100 seeded queries using direct
FP64 squared differences across all base vectors, allowing only equal-distance
boundary ties. Exact label ties are preserved and explicitly counted.

Use stable first-L0-evaluation-order ties for candidate scoring and reranking;
GAP score ties use query ID, never labels. If fewer than 12 candidates: set
gap to +infinity (last priority), count/report. Refinement depth is min(L,|V|)
with actual counts reported. If any |V|<64, STOP before policy interpretation
because nominal equal budgets would not necessarily imply equal costs.

All L=16,32,64 and fractions=.10,.25,.50 reported. Select floor(f*n) queries.
100 RANDOM replicates use PCG64 SeedSequence([94000059,scope,replicate]);
one permutation shared across L and fractions, prefix selection. Shards
are query IDs modulo 4; rerun identical policies within each shard with no
tuning. All selectors select the SAME number at each budget.

ORACLE-QUERY (nonimplementable): rank by actual depth-specific recoverable
gain Recall(exact top-L rerank)-Recall(native), descending, query-ID ties.
This is the true fixed-L selector upper bound. Also report a separately named
full-pool-loss selector to resolve the alternative interpretation of ranking
by full candidate ranking_recall_loss. Never use either label in GAP/RANDOM.
Keep signed gains; ratios null if denominator<=1e-12.

Useful held-out stratification: for at least TWO L, BOTH .25 and .50 GAP
recall exceed random mean AND its empirical 97.5th percentile (two-sided
95% interval = linear quantiles 2.5%,97.5%). These are random-allocation
intervals, not population uncertainty or statistical significance. Signal
failure: most budgets indistinguishable or inconsistent signs. Strong
success: useful criterion plus GAP exceeds random at most nonzero budgets,
advantage broadly stable with budget and meaningful oracle headroom capture;
last two remain qualitative, not invented post-hoc numerical thresholds.

Report PQ native, full PQ-pool exact oracle, top-L all queries, every policy
gain/recovery/cost/efficiency, four shards, PQ-gap quintiles (score-only equal
count with query-ID ties), Spearman and harmful fractions. Primary results
must be saved before secondary rescued-candidate diagnostics. No exact
descriptor chooses queries/candidates. Full oracle costs actual pool size;
validation/GT offline costs excluded from deployment refinement budget.

Phase 4A tests workload-level prioritization, not a deployable streaming
threshold, wall-clock speedup, or external dataset generalization. One fixed
model is a gatekeeper, not multi-codebook replication.
