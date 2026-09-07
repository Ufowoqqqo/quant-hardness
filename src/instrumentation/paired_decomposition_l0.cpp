#include "instrumentation/paired_decomposition_l0.h"

#include "instrumentation/faiss_level0_recorder.h"
#include "metrics/candidate_oracle.h"
#include "metrics/recall.h"

#include <cmath>
#include <span>
#include <stdexcept>

namespace quant_hardness {
namespace {

std::size_t intersection_size(std::span<const faiss::idx_t> first,
                              std::span<const faiss::idx_t> second) {
  std::size_t common = 0, left = 0, right = 0;
  while (left < first.size() && right < second.size()) {
    if (first[left] == second[right]) {
      ++common; ++left; ++right;
    } else if (first[left] < second[right]) {
      ++left;
    } else {
      ++right;
    }
  }
  return common;
}

} // namespace

std::vector<QueryDecompositionL0> measure_paired_decomposition_l0(
    faiss::IndexHNSW &graph, faiss::Index &exact_storage,
    faiss::Index &pq_storage, const float *base_vectors,
    faiss::idx_t base_count, const float *queries, faiss::idx_t query_count,
    int dimension, faiss::idx_t k,
    const std::vector<faiss::idx_t> &ground_truth_ids,
    const faiss::SearchParametersHNSW &parameters,
    bool validate_native_identity) {
  if (ground_truth_ids.size() != static_cast<std::size_t>(query_count * k)) {
    throw std::invalid_argument("ground-truth size does not match queries");
  }
  const std::string fingerprint = graph_fingerprint(graph.hnsw);
  auto exact = search_with_level0_recording(
      graph, exact_storage, queries, query_count, k, parameters,
      validate_native_identity);
  auto pq = search_with_level0_recording(
      graph, pq_storage, queries, query_count, k, parameters,
      validate_native_identity);
  if (graph_fingerprint(graph.hnsw) != fingerprint) {
    throw std::runtime_error("L0 paired decomposition changed graph topology");
  }

  std::vector<QueryDecompositionL0> output;
  output.reserve(query_count);
  for (faiss::idx_t query_id = 0; query_id < query_count; ++query_id) {
    const std::size_t offset = static_cast<std::size_t>(query_id * k);
    const auto truth = std::span(ground_truth_ids).subspan(offset, k);
    const float *query = queries + query_id * dimension;
    auto exact_oracle = exact_rerank_l2(
        base_vectors, base_count, dimension, query,
        exact.level0_evaluated_ids[query_id], k);
    auto pq_oracle = exact_rerank_l2(
        base_vectors, base_count, dimension, query,
        pq.level0_evaluated_ids[query_id], k);

    QueryDecompositionL0 row{};
    row.ground_truth_ids.assign(truth.begin(), truth.end());
    row.exact_native_ids.assign(exact.native.ids.begin() + offset,
                                exact.native.ids.begin() + offset + k);
    row.pq_native_ids.assign(pq.native.ids.begin() + offset,
                             pq.native.ids.begin() + offset + k);
    row.exact_l0_oracle_ids = std::move(exact_oracle.ids);
    row.pq_l0_oracle_ids = std::move(pq_oracle.ids);
    row.exact_l0_evaluated_ids =
        std::move(exact.level0_evaluated_ids[query_id]);
    row.pq_l0_evaluated_ids = std::move(pq.level0_evaluated_ids[query_id]);
    row.exact_upper_only_evaluated_ids =
        std::move(exact.upper_only_evaluated_ids[query_id]);
    row.pq_upper_only_evaluated_ids =
        std::move(pq.upper_only_evaluated_ids[query_id]);
    row.recall_exact_native = recall_at_k(row.exact_native_ids, truth, k);
    row.recall_pq_native = recall_at_k(row.pq_native_ids, truth, k);
    row.recall_exact_l0_oracle =
        recall_at_k(row.exact_l0_oracle_ids, truth, k);
    row.recall_pq_l0_oracle = recall_at_k(row.pq_l0_oracle_ids, truth, k);
    row.coverage_exact_l0 =
        candidate_coverage_at_k(row.exact_l0_evaluated_ids, truth, k);
    row.coverage_pq_l0 =
        candidate_coverage_at_k(row.pq_l0_evaluated_ids, truth, k);
    if (row.recall_exact_l0_oracle != row.coverage_exact_l0 ||
        row.recall_pq_l0_oracle != row.coverage_pq_l0) {
      throw std::runtime_error("L0 oracle recall differs from L0 coverage");
    }
    row.delta_total = row.recall_exact_native - row.recall_pq_native;
    row.delta_discovery =
        row.recall_exact_l0_oracle - row.recall_pq_l0_oracle;
    row.delta_ranking = row.recall_pq_l0_oracle - row.recall_pq_native;
    row.delta_exact_control =
        row.recall_exact_l0_oracle - row.recall_exact_native;
    if (std::abs(row.delta_total - (row.delta_discovery + row.delta_ranking -
                                    row.delta_exact_control)) > 1e-12) {
      throw std::runtime_error("L0 decomposition identity failed");
    }
    row.evaluated_l0_intersection_size = intersection_size(
        row.exact_l0_evaluated_ids, row.pq_l0_evaluated_ids);
    const std::size_t union_size = row.exact_l0_evaluated_ids.size() +
                                   row.pq_l0_evaluated_ids.size() -
                                   row.evaluated_l0_intersection_size;
    row.evaluated_l0_jaccard =
        static_cast<double>(row.evaluated_l0_intersection_size) / union_size;
    output.push_back(std::move(row));
  }
  return output;
}

} // namespace quant_hardness
