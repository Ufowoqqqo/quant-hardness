#pragma once

#include <vector>

#include <faiss/Index.h>

namespace quant_hardness {

struct GroundTruthResults {
  std::vector<float> distances;
  std::vector<faiss::idx_t> ids;
};

GroundTruthResults exhaustive_l2_top_k(const float *base_vectors,
                                       faiss::idx_t base_count,
                                       const float *queries,
                                       faiss::idx_t query_count, int dimension,
                                       faiss::idx_t k);

} // namespace quant_hardness
