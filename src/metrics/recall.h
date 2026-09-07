#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include <faiss/Index.h>

namespace quant_hardness {

double recall_at_k(std::span<const faiss::idx_t> result_ids,
                   std::span<const faiss::idx_t> ground_truth_ids,
                   std::size_t k);

double delta_recall(double recall_exact, double recall_quantized);

void require_identical_query_order(std::span<const std::uint64_t> exact_order,
                                   std::span<const std::uint64_t> pq_order);

} // namespace quant_hardness
