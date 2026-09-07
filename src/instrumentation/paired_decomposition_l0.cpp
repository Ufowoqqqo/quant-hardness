#include "instrumentation/paired_decomposition_l0.h"

#include "instrumentation/faiss_level0_recorder.h"
#include "metrics/candidate_oracle.h"
#include "metrics/recall.h"

#include <cmath>
#include <algorithm>
#include <span>
#include <sstream>
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

float squared_l2(const float *left, const float *right, int dimension) {
  float distance = 0.0F;
  for (int component = 0; component < dimension; ++component) {
    const float difference = left[component] - right[component];
    distance += difference * difference;
  }
  return distance;
}

bool mismatch_is_boundary_tie(
    const std::vector<faiss::idx_t> &candidates,
    const std::vector<faiss::idx_t> &oracle,
    std::span<const faiss::idx_t> truth, const float *base_vectors,
    const float *query, int dimension) {
  if (oracle.empty()) return false;
  const float cutoff = squared_l2(
      query, base_vectors + oracle.back() * dimension, dimension);
  bool found_missing = false;
  for (const faiss::idx_t truth_id : truth) {
    if (std::binary_search(candidates.begin(), candidates.end(), truth_id) &&
        std::find(oracle.begin(), oracle.end(), truth_id) == oracle.end()) {
      found_missing = true;
      if (squared_l2(query, base_vectors + truth_id * dimension, dimension) !=
          cutoff) {
        return false;
      }
    }
  }
  return found_missing;
}

bool native_is_equivalent_exact_rerank(
    const std::vector<faiss::idx_t> &native,
    const std::vector<faiss::idx_t> &sorted_candidates,
    const std::vector<float> &reference_distances, const float *base_vectors,
    const float *query, int dimension) {
  std::vector<float> native_distances;
  native_distances.reserve(native.size());
  for (const faiss::idx_t id : native) {
    if (!std::binary_search(sorted_candidates.begin(), sorted_candidates.end(),
                            id))
      return false;
    native_distances.push_back(
        squared_l2(query, base_vectors + id * dimension, dimension));
  }
  auto expected = reference_distances;
  std::sort(native_distances.begin(), native_distances.end());
  std::sort(expected.begin(), expected.end());
  return native_distances == expected;
}

} // namespace

std::vector<QueryDecompositionL0> measure_paired_decomposition_l0(
    faiss::IndexHNSW &graph, faiss::Index &exact_storage,
    faiss::Index &pq_storage, const float *base_vectors,
    faiss::idx_t base_count, const float *queries, faiss::idx_t query_count,
    int dimension, faiss::idx_t k,
    const std::vector<faiss::idx_t> &ground_truth_ids,
    const faiss::SearchParametersHNSW &parameters,
    bool validate_native_identity, bool allow_verified_boundary_ties,
    bool stable_evaluation_order_ties) {
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
    auto exact_oracle = stable_evaluation_order_ties
        ? exact_rerank_l2_stable(base_vectors, base_count, dimension, query,
                                 exact.level0_evaluation_order_ids[query_id], k)
        : exact_rerank_l2(base_vectors, base_count, dimension, query,
                          exact.level0_evaluated_ids[query_id], k);
    auto pq_oracle = stable_evaluation_order_ties
        ? exact_rerank_l2_stable(base_vectors, base_count, dimension, query,
                                 pq.level0_evaluation_order_ids[query_id], k)
        : exact_rerank_l2(base_vectors, base_count, dimension, query,
                          pq.level0_evaluated_ids[query_id], k);

    QueryDecompositionL0 row{};
    row.ground_truth_ids.assign(truth.begin(), truth.end());
    row.exact_native_ids.assign(exact.native.ids.begin() + offset,
                                exact.native.ids.begin() + offset + k);
    row.pq_native_ids.assign(pq.native.ids.begin() + offset,
                             pq.native.ids.begin() + offset + k);
    row.exact_l0_oracle_ids = std::move(exact_oracle.ids);
    row.pq_l0_oracle_ids = std::move(pq_oracle.ids);
    if (stable_evaluation_order_ties) {
      if (!native_is_equivalent_exact_rerank(
              row.exact_native_ids, exact.level0_evaluated_ids[query_id],
              exact_oracle.distances, base_vectors, query, dimension))
        throw std::runtime_error(
            "exact native result is not an equivalent exact L0 rerank");
      // Native exact is a valid realization of the exact top-k under boundary
      // ties. Use that realization so the exact-control metric is well-defined.
      row.exact_l0_oracle_ids = row.exact_native_ids;
    }
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
    row.exact_oracle_coverage_tie_mismatch =
        row.recall_exact_l0_oracle != row.coverage_exact_l0;
    row.pq_oracle_coverage_tie_mismatch =
        row.recall_pq_l0_oracle != row.coverage_pq_l0;
    const bool exact_tie = !row.exact_oracle_coverage_tie_mismatch ||
        mismatch_is_boundary_tie(row.exact_l0_evaluated_ids,
                                 row.exact_l0_oracle_ids, truth, base_vectors,
                                 query, dimension);
    const bool pq_tie = !row.pq_oracle_coverage_tie_mismatch ||
        mismatch_is_boundary_tie(row.pq_l0_evaluated_ids,
                                 row.pq_l0_oracle_ids, truth, base_vectors,
                                 query, dimension);
    if ((!allow_verified_boundary_ties &&
         (row.exact_oracle_coverage_tie_mismatch ||
          row.pq_oracle_coverage_tie_mismatch)) ||
        !exact_tie || !pq_tie) {
      std::ostringstream message;
      message << "L0 oracle recall differs from L0 coverage at query "
              << query_id << ": exact_oracle=" << row.recall_exact_l0_oracle
              << " exact_coverage=" << row.coverage_exact_l0
              << " pq_oracle=" << row.recall_pq_l0_oracle
              << " pq_coverage=" << row.coverage_pq_l0 << " truth=";
      for (const auto id : row.ground_truth_ids) message << id << ',';
      message << " exact_oracle_ids=";
      for (const auto id : row.exact_l0_oracle_ids) message << id << ',';
      message << " pq_oracle_ids=";
      for (const auto id : row.pq_l0_oracle_ids) message << id << ',';
      throw std::runtime_error(message.str());
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
