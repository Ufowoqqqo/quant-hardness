#pragma once

#include <faiss/Index.h>

#include <cstddef>
#include <span>
#include <vector>

namespace quant_hardness {

struct ExactRerankResults {
  std::vector<float> distances;
  std::vector<faiss::idx_t> ids;
};

ExactRerankResults exact_rerank_l2(const float *base_vectors,
                                   faiss::idx_t base_count, int dimension,
                                   const float *query,
                                   std::span<const faiss::idx_t> candidate_ids,
                                   std::size_t k);

// Exact reranking that preserves candidate first-evaluation order for equal
// distances. Non-tied rankings are identical to exact_rerank_l2.
ExactRerankResults exact_rerank_l2_stable(
    const float *base_vectors, faiss::idx_t base_count, int dimension,
    const float *query, std::span<const faiss::idx_t> candidate_ids,
    std::size_t k);

double candidate_coverage_at_k(std::span<const faiss::idx_t> candidate_ids,
                               std::span<const faiss::idx_t> ground_truth_ids,
                               std::size_t k);

} // namespace quant_hardness
