# Phase 5B preregistration — before new training or retrieval

Parent Phase5A commit:541a0eb8c812bf9926f0bc5de3699d908dfa20a4.
Dataset amendment:fbdf4784a505dfbf201be9e78f788e67d31fd4b2.
No Phase5B retrieval results exist at this checkpoint.

Reuse the exact Phase5A normalized990000 base/10000 queries/65536 training
rows, role IDs, exhaustive GT, warmups and serialized graph. Verify checksums
against committed metadata before any new model training. Do not rebuild or
renormalize. PQ architecture is the only scientific intervention:
PQ768x8/dsub2/768B/8x becomes PQ384x8/dsub4/384B/16x. One seed95000037,
25 training iterations,1 redo,24 training threads, identical training rows.
No OPQ or alternative quantizer. Standard FAISS pinned20f14b31a6d54e243a3d1de6ae193fc4c3ec18ed.

Hypotheses, fixed before retrieval:

- H1: candidate-oracle recovery>=.90 at practical operating points.
- H2: ranking_loss384>discovery_loss384; stronger compression principally
  increases ranking-side loss rather than discovery-side loss.
- H3: L16 recovery>=.80 at at least one practical point; secondary>=.60
  across most practical points.
- H4: systems value can improve or weaken as ADC/storage gets cheaper and
  ranking noise increases. Do not presume improvement.

Practical/high-quality exact control interpretation emphasizes ef64/128,
whose frozen exact recall is .94202/.96602; ef32(.89641) remains fully reported.
No threshold will be adjusted from these Phase5A-known values.

Run ef32/64/128 only, k10, same graph. Exact results are reverified against
Phase5A IDs. Real PQ384 L0 evaluated pools define all oracles/refinements.
Record native/exact/oracle/ALL16 IDs, complete unique L0 IDs/PQ scores and
signed per-query losses. Bound16 uses exactly existing score/first-evaluation
ordinal tie semantics; compare every top16 and final10 against full recording
and offline score sort. Strict GT-ID recall never replaced by tie tolerance.
Sanity uses Phase5A seed95000053 and10000 identical sample pairs/order pairs;
numerically worse PQ quality is an outcome, not grounds to retune.

Single-thread six cells,5 repetitions,4 full query passes per repetition,
512 separate training-row warmups/worker. Random order seed95000089.
Identical compiler flags/physical affinity CPU2, coordinatorCPU0; internal
OpenMP/BLAS1. Then ef64 only with1/8 workers,5 repetitions, order seed95000090,
CPU2..9 physical cores. All returned IDs must match offline references.
Report actual per-query wall times/QPS and every repetition, not best runs.
Phase5A performance is a historical measured reference, not contemporaneously
randomized with Phase5B; record this temporal/frequency confound explicitly.

Combine12 observed operating points; Pareto=not dominated in both recall/QPS
with at least one strict improvement. No interpolation establishes dominance.
PQ-code payload compression is separate from graph/raw-vector deployment
memory. Keep original graph and exact vector backing in memory accounting.

Authorized diagnostic after primary single-thread analysis: PQ approximate
rank of each candidate-oracle top10 item that is also a GT top10 member.
Store all oracle-top10 ranks with a GT-relevance flag as a sensitivity record.
Rank ties follow score then first evaluation ordinal. Report fractions within
10/16/32/64 and empirical CDF, including zero-relevant-query handling. This
does not perform L32/L64 reranking or a second graph search.

Interpretation A: discovery robust, ranking grows, L16 effective, useful Pareto
points. B: discovery robust but top16 insufficient. C: material discovery
failure. D: scientific robustness but no useful systems frontier advantage.
Do not force A. Preserve rare negative and positive discovery effects.
After primary results, stop. Exactly one next-experiment recommendation only.
