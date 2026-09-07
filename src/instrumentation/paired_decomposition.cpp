#include "instrumentation/paired_decomposition.h"

#include "instrumentation/faiss_distance_recorder.h"
#include "metrics/candidate_oracle.h"
#include "metrics/recall.h"

#include <cmath>
#include <span>
#include <stdexcept>

namespace quant_hardness {
namespace {

std::size_t intersection_size(std::span<const faiss::idx_t> first,
                              std::span<const faiss::idx_t> second) {
  std::size_t common = 0;
  std::size_t left = 0;
  std::size_t right = 0;
  while (left < first.size() && right < second.size()) {
    if (first[left] == second[right]) {
      ++common;
      ++left;
      ++right;
    } else if (first[left] < second[right]) {
      ++left;
    } else {
      ++right;
    }
  }
  return common;
}

} // namespace

std::vector<QueryDecomposition> measure_paired_decomposition(
    faiss::IndexHNSW &graph, faiss::Index &exact_storage,
    faiss::Index &pq_storage, const float *base_vectors,
    faiss::idx_t base_count, const float *queries, faiss::idx_t query_count,
    int dimension, faiss::idx_t k,
    const std::vector<faiss::idx_t> &ground_truth_ids,
    const faiss::SearchParametersHNSW &parameters,
    bool validate_instrumentation_identity) {
  if (ground_truth_ids.size() != static_cast<std::size_t>(query_count * k)) {
    throw std::invalid_argument("ground-truth size does not match queries");
  }
  const std::string fingerprint = graph_fingerprint(graph.hnsw);
  SearchResults exact_reference;
  SearchResults pq_reference;
  if (validate_instrumentation_identity) {
    exact_reference = search_with_storage(graph, exact_storage, queries,
                                          query_count, k, parameters);
    pq_reference = search_with_storage(graph, pq_storage, queries, query_count,
                                       k, parameters);
  }

  RecordingIndex exact_recording(exact_storage, queries, query_count);
  const SearchResults exact_native = search_with_storage(
      graph, exact_recording, queries, query_count, k, parameters);
  RecordingIndex pq_recording(pq_storage, queries, query_count);
  const SearchResults pq_native = search_with_storage(
      graph, pq_recording, queries, query_count, k, parameters);
  if (validate_instrumentation_identity &&
      (exact_reference.ids != exact_native.ids ||
       exact_reference.distances != exact_native.distances ||
       pq_reference.ids != pq_native.ids ||
       pq_reference.distances != pq_native.distances)) {
    throw std::runtime_error("instrumentation changed native search results");
  }
  if (graph_fingerprint(graph.hnsw) != fingerprint) {
    throw std::runtime_error("paired decomposition changed graph fingerprint");
  }

  std::vector<QueryDecomposition> output;
  output.reserve(static_cast<std::size_t>(query_count));
  for (faiss::idx_t query_id = 0; query_id < query_count; ++query_id) {
    const std::size_t offset = static_cast<std::size_t>(query_id * k);
    const auto truth = std::span(ground_truth_ids)
                           .subspan(offset, static_cast<std::size_t>(k));
    std::vector<faiss::idx_t> exact_evaluated =
        exact_recording.sorted_evaluated_ids(query_id);
    std::vector<faiss::idx_t> pq_evaluated =
        pq_recording.sorted_evaluated_ids(query_id);
    if (exact_evaluated.size() < static_cast<std::size_t>(k) ||
        pq_evaluated.size() < static_cast<std::size_t>(k)) {
      throw std::runtime_error("fewer than k unique node distances evaluated");
    }
    const float *query = queries + query_id * dimension;
    ExactRerankResults exact_oracle = exact_rerank_l2(
        base_vectors, base_count, dimension, query, exact_evaluated, k);
    ExactRerankResults pq_oracle = exact_rerank_l2(
        base_vectors, base_count, dimension, query, pq_evaluated, k);

    QueryDecomposition row{};
    row.ground_truth_ids.assign(truth.begin(), truth.end());
    row.exact_native_ids.assign(exact_native.ids.begin() + offset,
                                exact_native.ids.begin() + offset + k);
    row.pq_native_ids.assign(pq_native.ids.begin() + offset,
                             pq_native.ids.begin() + offset + k);
    row.exact_oracle_ids = std::move(exact_oracle.ids);
    row.pq_oracle_ids = std::move(pq_oracle.ids);
    row.exact_evaluated_ids = std::move(exact_evaluated);
    row.pq_evaluated_ids = std::move(pq_evaluated);
    row.recall_exact_native = recall_at_k(row.exact_native_ids, truth, k);
    row.recall_pq_native = recall_at_k(row.pq_native_ids, truth, k);
    row.recall_exact_oracle = recall_at_k(row.exact_oracle_ids, truth, k);
    row.recall_pq_oracle = recall_at_k(row.pq_oracle_ids, truth, k);
    row.coverage_exact =
        candidate_coverage_at_k(row.exact_evaluated_ids, truth, k);
    row.coverage_pq = candidate_coverage_at_k(row.pq_evaluated_ids, truth, k);
    if (std::abs(row.recall_exact_oracle - row.coverage_exact) > 1e-12 ||
        std::abs(row.recall_pq_oracle - row.coverage_pq) > 1e-12) {
      throw std::runtime_error("oracle recall differs from candidate coverage");
    }
    row.delta_total = row.recall_exact_native - row.recall_pq_native;
    row.delta_discovery = row.recall_exact_oracle - row.recall_pq_oracle;
    row.delta_ranking = row.recall_pq_oracle - row.recall_pq_native;
    row.delta_exact_control = row.recall_exact_oracle - row.recall_exact_native;
    if (std::abs(row.delta_total - (row.delta_discovery + row.delta_ranking -
                                    row.delta_exact_control)) > 1e-12) {
      throw std::runtime_error("decomposition identity failed");
    }
    row.evaluated_intersection_size =
        intersection_size(row.exact_evaluated_ids, row.pq_evaluated_ids);
    const std::size_t union_size = row.exact_evaluated_ids.size() +
                                   row.pq_evaluated_ids.size() -
                                   row.evaluated_intersection_size;
    row.evaluated_jaccard =
        static_cast<double>(row.evaluated_intersection_size) /
        static_cast<double>(union_size);
    output.push_back(std::move(row));
  }
  return output;
}

} // namespace quant_hardness
