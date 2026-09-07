#include "metrics/ground_truth.h"

#include <faiss/IndexFlat.h>

#include <stdexcept>

namespace quant_hardness {

GroundTruthResults exhaustive_l2_top_k(const float *base_vectors,
                                       faiss::idx_t base_count,
                                       const float *queries,
                                       faiss::idx_t query_count, int dimension,
                                       faiss::idx_t k) {
  if (base_vectors == nullptr || queries == nullptr || base_count <= 0 ||
      query_count <= 0 || dimension <= 0 || k <= 0 || k > base_count) {
    throw std::invalid_argument("invalid exhaustive ground-truth dimensions");
  }

  faiss::IndexFlatL2 exhaustive_index(dimension);
  exhaustive_index.add(base_count, base_vectors);

  GroundTruthResults output;
  output.distances.resize(static_cast<std::size_t>(query_count * k));
  output.ids.resize(static_cast<std::size_t>(query_count * k));
  exhaustive_index.search(query_count, queries, k, output.distances.data(),
                          output.ids.data());
  return output;
}

} // namespace quant_hardness
