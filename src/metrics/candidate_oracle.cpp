#include "metrics/candidate_oracle.h"

#include <algorithm>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace quant_hardness {
namespace {

float squared_l2(const float *left, const float *right, int dimension) {
  float distance = 0.0F;
  for (int component = 0; component < dimension; ++component) {
    const float difference = left[component] - right[component];
    distance += difference * difference;
  }
  return distance;
}

void require_unique_valid(std::span<const faiss::idx_t> ids,
                          faiss::idx_t upper_bound) {
  std::unordered_set<faiss::idx_t> unique;
  for (const faiss::idx_t id : ids) {
    if (id < 0 || id >= upper_bound || !unique.insert(id).second) {
      throw std::invalid_argument("candidate IDs must be unique and in range");
    }
  }
}

} // namespace

ExactRerankResults exact_rerank_l2(const float *base_vectors,
                                   faiss::idx_t base_count, int dimension,
                                   const float *query,
                                   std::span<const faiss::idx_t> candidate_ids,
                                   std::size_t k) {
  if (base_vectors == nullptr || query == nullptr || base_count <= 0 ||
      dimension <= 0 || k == 0 || candidate_ids.size() < k) {
    throw std::invalid_argument("invalid exact-reranking input");
  }
  require_unique_valid(candidate_ids, base_count);
  std::vector<std::pair<float, faiss::idx_t>> ranked;
  ranked.reserve(candidate_ids.size());
  for (const faiss::idx_t id : candidate_ids) {
    ranked.emplace_back(
        squared_l2(query, base_vectors + id * dimension, dimension), id);
  }
  std::partial_sort(ranked.begin(), ranked.begin() + k, ranked.end());

  ExactRerankResults output;
  output.distances.reserve(k);
  output.ids.reserve(k);
  for (std::size_t i = 0; i < k; ++i) {
    output.distances.push_back(ranked[i].first);
    output.ids.push_back(ranked[i].second);
  }
  return output;
}

double candidate_coverage_at_k(std::span<const faiss::idx_t> candidate_ids,
                               std::span<const faiss::idx_t> ground_truth_ids,
                               std::size_t k) {
  if (k == 0 || ground_truth_ids.size() < k) {
    throw std::invalid_argument("invalid candidate-coverage input");
  }
  std::unordered_set<faiss::idx_t> candidates;
  for (const faiss::idx_t id : candidate_ids) {
    if (id < 0 || !candidates.insert(id).second) {
      throw std::invalid_argument("candidate IDs must be unique and valid");
    }
  }
  std::unordered_set<faiss::idx_t> truth;
  std::size_t covered = 0;
  for (std::size_t i = 0; i < k; ++i) {
    const faiss::idx_t id = ground_truth_ids[i];
    if (id < 0 || !truth.insert(id).second) {
      throw std::invalid_argument("ground-truth IDs must be unique and valid");
    }
    covered += candidates.contains(id) ? 1 : 0;
  }
  return static_cast<double>(covered) / static_cast<double>(k);
}

} // namespace quant_hardness
