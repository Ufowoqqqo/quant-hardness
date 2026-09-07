# Findings

This document contains only findings supported by recorded experiments. Keep
observations separate from interpretations and link every item to its raw run
directory and experiment-log entry.

## Phase 3A gatekeeper observation

At SIFT1M exact-recall operating points 0.95319 and 0.98379, PQ64x8 produced
mean discovery deltas 0.00119 and 0.00092; exact reranking of PQ-evaluated L0
candidates recovered 0.98819 and 0.99201 of the native recall gap. Together
with the three controlled synthetic studies, this is negative evidence for
frequent PQ-induced candidate-discovery failure under the tested conditions.
See `docs/phase3a_sift1m.md` and `runs/phase3a_sift1m_v1/`.
