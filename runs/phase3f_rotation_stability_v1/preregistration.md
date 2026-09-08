# Phase 3F preregistration — before rotations or PQ results

Fixed SIFT1M identities, original graph, ef64/k10, and all 10,000 saved
FP32 L0 candidate pools/exact scores/GT from Phase 3C. No traversal.
PQ64x8, 65536 training IDs: Phase 3E subset A (FAISS rand_perm seed80400011).
Five bases x three matched initialization seeds80500011/29/47. Train all
15 models independently, including identity. No model selection.

## Hypotheses

H1. Exact candidate geometry, ranking, oracle recall remain numerically
invariant under every rotation.
H2. Basis changes reduce harmful-query stability more than initialization.
H3. A nontrivial subset remains majority-harmful across most/all rotations.
H4. Invariant local top-k gaps associate with rotation-averaged loss.
H5. Inversion pairs are less stable across rotations than harmful queries.
These are exploratory directional hypotheses, not significance tests.
Do not retrospectively assign numeric pass thresholds to H2-H5.

## Rotation and validation gate

R0 identity; R1-R4: independent NumPy PCG64 standard-normal128x128 matrices,
QR in float64, multiply columns of Q by sign(diag(R_QR)) (zero sign=1).
The resulting Q is called R. Column convention x_R=R x; stored row arrays
use X_R=X @ R.T. Save float64 matrices; compute transform in float64 then
round once to float32 for FAISS. Same transform for base, queries, training.
No centering, normalization, learned rotation, or PQ changes.

Before ANY PQ training: validate all five bases, all candidate scores/rankings,
and exhaustive top10 for100 deterministic queries over all1m database items.
Save sample IDs. Original exact SIFT distances have integer values; ties can
change ordering or membership after float rounding. Compare raw rotated
ordering, explicitly count/save all tie changes, and reject any non-tied
inversion beyond declared tolerance. Retain original exact scores/order as
the common reference for all scoring conditions. Oracle recall must be
identical for this canonical, original-tie-order reference; report raw
rotation-tie effects separately rather than forcing raw equality.

Orthogonality max absolute error <=1e-12. Float64 transforms on1000
deterministic norm/base-pair/query-base checks: atol1e-8+rtol1e-12.
Float32 stored vectors with float64 distance accumulation: squared-distance
atol0.005+rtol5e-7; additional FP32 accumulation check rtol2e-6, same atol.
Use squared norms as the norm-invariance check and also report L2 norm error.
GT exhaustive validation uses float64 dot products and checks provided
membership modulo exact ties as well as rotated membership. Candidate
validation uses direct float64 squared differences for EVERY saved candidate.
STOP before training/interpretation if a gate fails; no tolerance widening.

## Analysis fixed in advance

Use prior strict cross-boundary definition (exact_i<exact_j, pq_i>pq_j),
stable candidate-order tie breaking, same512 sampled pairs/query as3C.
Preserve signed loss, all per-query metrics, replacement and union-pair data.
Harmful rotation = >=2/3 harmful initializations. Classes by number of harmful
rotations: robust0, occasional1-2, usual3-4, persistent5. Robust here need
not mean never harmful in any individual model.

Population variance: W=mean_r Var_i(L_ri), B=Var_r(mean_i L_ri), T=W+B.
Also save sample-variance signed contrast B_sample-W_sample/3, not clipped,
not formal random-effects inference. Matched initialization seeds across
bases motivate descriptive two-way rotation/init/interaction decomposition;
do not treat B as a pure causal geometry or alignment percentage.

All model pairs, separated within/between bases; additionally R0-to-random,
random-to-random, matched-init vs unmatched-init, oracle=1 and no-boundary-tie
sensitivities. Report marginal harmful prevalences and independent-marginal
Jaccard baselines: worse global PQ quality can mechanically raise overlap.
Do not discard globally poor rotations. Correlations exploratory only.

After primary checkpoint ONLY: average I1 across five bases, and average
I1/I2/I3 within each base. Report each ensemble's member baseline and loss
recovery. Five-versus-three diversity is a confound, not a method comparison.
No weight tuning, performance benchmarking, or proposed algorithm.
