#pragma once

#include "graph/faiss_shared_hnsw.h"

#include <faiss/Index.h>
#include <faiss/IndexHNSW.h>

#include <vector>

namespace quant_hardness {

// Results from replaying FAISS's native upper-level greedy navigation and
// native level-0 search as two explicit stages. The level-0 entry point is
// re-evaluated only for phase attribution; this does not affect search state.
struct PhaseSeparatedSearchResults {
  SearchResults native;
  std::vector<std::vector<faiss::idx_t>> upper_only_evaluated_ids;
  std::vector<std::vector<faiss::idx_t>> level0_evaluated_ids;
};

PhaseSeparatedSearchResults search_with_level0_recording(
    faiss::IndexHNSW &graph, faiss::Index &traversal_storage,
    const float *queries, faiss::idx_t query_count, faiss::idx_t k,
    const faiss::SearchParametersHNSW &parameters,
    bool validate_native_identity = true);

} // namespace quant_hardness
