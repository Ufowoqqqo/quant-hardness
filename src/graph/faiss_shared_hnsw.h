#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include <faiss/Index.h>
#include <faiss/IndexHNSW.h>

namespace quant_hardness {

struct SearchResults {
  std::vector<float> distances;
  std::vector<faiss::idx_t> ids;
};

// SHA-256 over the canonical faiss-hnsw-struct-v1 representation.
std::string graph_fingerprint(const faiss::HNSW &hnsw);

// Research-only, single-threaded adapter. The sole HNSW topology remains in
// graph_index; traversal_storage supplies only query-to-node distances.
SearchResults search_with_storage(faiss::IndexHNSW &graph_index,
                                  faiss::Index &traversal_storage,
                                  const float *queries,
                                  faiss::idx_t query_count, faiss::idx_t k,
                                  const faiss::SearchParametersHNSW &params);

} // namespace quant_hardness
