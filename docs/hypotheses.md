# Hypotheses

These are provisional hypotheses to test, not established findings.

## H0: no query-specific hardness effect

After controlling for graph and search parameters, quantization does not create
a stable subset of disproportionately damaged queries. Observed differences
are explained by aggregate accuracy loss or measurement noise.

## H1: local distance-order instability

Queries with small distance margins among relevant candidates experience more
quantization-induced neighbor-order changes and larger paired search damage.

## H2: graph-decision instability

Quantization errors near traversal decision boundaries alter graph expansion
paths, increasing per-query search cost or reducing recall.

## H3: distribution-shift sensitivity

Predictors learned or calibrated in-distribution do not retain their predictive
relationship under query distribution shift or OOD evaluation.

## Falsification priorities

1. Test whether paired per-query damage exceeds run-to-run variation.
2. Test whether apparent hard-query rankings are stable across deterministic
   repetitions and search budgets.
3. Test simple competing explanations before adding predictors or methods.
4. Report null and contradictory results with the same per-query detail.

