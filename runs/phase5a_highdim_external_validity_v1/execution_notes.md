# Phase 5A execution notes, before primary outcomes

Amendment commit: fbdf4784a505dfbf201be9e78f788e67d31fd4b2.
The latest user request explicitly raises deterministic independent exhaustive
GT validation from32 to at least100 queries. The execution uses100 with the
same seed95000071; frozen ANN/PQ/data/split parameters remain unchanged.
The committed config/manifest stay byte-identical; execution provenance
records this requested validation-count extension separately.

PQ ADC approximates squared L2 on the normalized input vectors. Decoded PQ
centroids/codes are NOT renormalized: doing so would change the frozen scoring
function. Independent cosine validation applies to the exact input vectors,
not to a redefined cosine metric on PQ reconstructions.

Ground truth uses exhaustive `faiss::fvec_L2sqr` on archived normalized FP32,
top11 retained to diagnose the rank10 boundary, ties by ascending base ID.
Recall is strict ID intersection/10. Independent validation uses scalar FP64
exhaustive L2 on100 queries and a separate NumPy FP64 cosine scan. Numerical
disagreements are documented and must be within2e-5; tolerance does not alter
recall labels or scores. Graph construction uses one add call, unchanged
M16/efConstruction80/seed20260907/24 threads; save actual graph fingerprint.
PQ768×8 trains one model with the frozen65536 rows, seed95000037,25 iterations.
Sanity sampling uses deterministic mt19937_64 seed95000053,10000 query/base
pairs and10000 candidate pair order comparisons. Checks include reference
ADC against decoded-vector L2 and direct code/ID alignment.

Only adapter change: permit ef32/128 alongside64. L0 retention/comparator,
HNSW traversal implementation, score tie order and L16 remain unchanged.
A synthetic d1536/PQ768 fixture (not an alternative trained dataset model)
checks full-sort/top16/final reranking/native IDs/fingerprint before timing.

## Technical issue: preprocessing I/O

The initial reconstruction wrote random destination rows directly to NFS.
After78924 rows it was interrupted for I/O impracticality; partial files are
preserved in `prepared_interrupted_nfs_writes/`, never valid inputs. The
replacement writes identical rows with the identical frozen normalization
function to a newly allocated /tmp NVMe scratch, then sequentially copies
finished arrays to `prepared/`. This changes transport only, not vector
values, ID order or experimental parameters. Scratch is preserved too.

## Technical issue: build worktree inspection

Unbounded CMake `git status --porcelain` hung scanning the large NFS worktree.
The interrupted configure made no ANN results. It now has a30-second timeout;
failure/timeout records dirty=true conservatively, never claims a clean tree.
Compiler/optimization flags remain unchanged. Record source/binary hashes
in execution provenance because implementation follows the amendment commit.

## Technical issue: provenance Git lookup

After all1000000 rows and array copies completed, preparation stalled at
`git -C third_party/faiss rev-parse HEAD`, NOT in array hashing. A bounded
probe confirmed that command timed out while root `git rev-parse` completed
in0.11s. `git --git-dir=third_party/faiss/.git rev-parse HEAD` returns the
expected20f14b31a6d54e243a3d1de6ae193fc4c3ec18ed immediately. A64MiB archive read
took0.0093s (cache diagnostic only; not ANN performance).
The stalled process was interrupted; no arrays were deleted or changed.
`prepare_phase5a.py --finalize-existing /tmp/phase5a-normalization-aifuhbw5`
recovers provenance only after full archived/scratch checksum comparisons,
finite/unit-norm checks of all outputs and exact training/base row alignment.
This makes no split/arithmetic/model change. Record this recovery explicitly.

No exploratory L, PQ, efSearch, OPQ or learned-policy experiments are authorized.
After primary single-thread analysis, run only ef64 with1/8 workers, then STOP.
