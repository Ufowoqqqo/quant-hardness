# Research objective

This repository studies quantization-induced query hardness in graph-based
approximate nearest-neighbor search.

The primary question is:

> When and why does quantization turn an easy graph-ANN query into a hard one?

Do **not** assume that the hypothesis is true.

## Scientific rules

1. Separate observation from interpretation.
2. Never claim novelty based only on experiments in this repository.
3. Never change an experimental metric or dataset preprocessing silently.
4. Exact-vector and quantized-vector search must be comparable under the same
   graph and search parameters unless the experiment explicitly studies graph
   construction.
5. Preserve raw per-query measurements. Never report only averages.
6. All experiments must record:
   - Git commit
   - dataset
   - random seed
   - graph parameters
   - search parameters
   - quantizer parameters
   - machine information
7. Every optimization must be validated against a simple reference
   implementation before benchmarking.
8. Prefer falsifying the hypothesis over producing positive-looking plots.
9. Never delete raw experimental results.
10. Do not introduce a new algorithm unless requested explicitly.

## Current phase

We are in the phenomenon-validation phase.

Priorities, in order:

1. Establish exact-search and quantized-search baselines.
2. Measure per-query quantization damage.
3. Identify predictors of quantization-induced hardness.
4. Test distribution shift / OOD.
5. Only then consider methods.

## Coding principles

- Keep instrumentation minimally invasive.
- Separate measurement code from ANN implementation.
- Use deterministic seeds where possible.
- Add unit tests for distance approximation and recall computation.
- Use configuration files rather than hard-coded experiment parameters.
- Scripts must support non-interactive execution on a server.

## Experiment reporting

After every meaningful experiment:

1. Append the exact command and configuration to `docs/experiment_log.md`.
2. Summarize results without interpretation.
3. List anomalies and possible confounders.
4. Propose the smallest next experiment that distinguishes competing
   explanations.
