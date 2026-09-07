# Third-party implementations

Place upstream ANN libraries, submodules, and clearly documented patches here.
Do not add repository-owned instrumentation directly to upstream source trees.
Record the exact upstream revision used by every experiment.

## Pinned dependencies

| Dependency | Tag | Exact commit | Purpose |
| --- | --- | --- | --- |
| Meta FAISS | `v1.15.0` | `20f14b31a6d54e243a3d1de6ae193fc4c3ec18ed` | Phase 1 CPU HNSW and PQ reference backend |

Initialize dependencies with `git submodule update --init`. The superproject
gitlink and top-level CMake revision check enforce the exact FAISS commit; no
floating branch is configured.
