#include "metrics/recall.h"

#include <cmath>
#include <stdexcept>
#include <unordered_set>

namespace quant_hardness {

double recall_at_k(std::span<const faiss::idx_t> result_ids,
                   std::span<const faiss::idx_t> ground_truth_ids,
                   std::size_t k) {
  if (k == 0 || result_ids.size() < k || ground_truth_ids.size() < k) {
    throw std::invalid_argument(
        "Recall@k requires two lists of at least k IDs");
  }

  std::unordered_set<faiss::idx_t> truth;
  std::unordered_set<faiss::idx_t> results;
  for (std::size_t i = 0; i < k; ++i) {
    if (ground_truth_ids[i] < 0 || !truth.insert(ground_truth_ids[i]).second) {
      throw std::invalid_argument(
          "ground-truth top-k IDs must be unique and valid");
    }
    if (result_ids[i] < 0 || !results.insert(result_ids[i]).second) {
      throw std::invalid_argument("search top-k IDs must be unique and valid");
    }
  }

  std::size_t intersection = 0;
  for (const faiss::idx_t id : results) {
    intersection += truth.contains(id) ? 1 : 0;
  }
  return static_cast<double>(intersection) / static_cast<double>(k);
}

double delta_recall(double recall_exact, double recall_quantized) {
  if (!std::isfinite(recall_exact) || !std::isfinite(recall_quantized) ||
      recall_exact < 0.0 || recall_exact > 1.0 || recall_quantized < 0.0 ||
      recall_quantized > 1.0) {
    throw std::invalid_argument(
        "recall values must be finite and within [0, 1]");
  }
  return recall_exact - recall_quantized;
}

void require_identical_query_order(std::span<const std::uint64_t> exact_order,
                                   std::span<const std::uint64_t> pq_order) {
  if (exact_order.size() != pq_order.size()) {
    throw std::invalid_argument("paired query counts differ");
  }
  for (std::size_t i = 0; i < exact_order.size(); ++i) {
    if (exact_order[i] != pq_order[i]) {
      throw std::invalid_argument("paired query ordering differs");
    }
  }
}

} // namespace quant_hardness
