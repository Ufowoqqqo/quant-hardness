#pragma once

#include "graph/faiss_shared_hnsw.h"

#include <faiss/Index.h>
#include <faiss/IndexHNSW.h>

#include <cstddef>
#include <vector>

namespace quant_hardness {

struct QueryDecompositionL0 {
  double recall_exact_native;
  double recall_pq_native;
  double recall_exact_l0_oracle;
  double recall_pq_l0_oracle;
  double coverage_exact_l0;
  double coverage_pq_l0;
  double delta_total;
  double delta_discovery;
  double delta_ranking;
  double delta_exact_control;
  bool exact_oracle_coverage_tie_mismatch;
  bool pq_oracle_coverage_tie_mismatch;
  std::size_t evaluated_l0_intersection_size;
  double evaluated_l0_jaccard;
  std::vector<faiss::idx_t> ground_truth_ids;
  std::vector<faiss::idx_t> exact_native_ids;
  std::vector<faiss::idx_t> pq_native_ids;
  std::vector<faiss::idx_t> exact_l0_oracle_ids;
  std::vector<faiss::idx_t> pq_l0_oracle_ids;
  std::vector<faiss::idx_t> exact_l0_evaluated_ids;
  std::vector<faiss::idx_t> pq_l0_evaluated_ids;
  std::vector<faiss::idx_t> exact_upper_only_evaluated_ids;
  std::vector<faiss::idx_t> pq_upper_only_evaluated_ids;
};

std::vector<QueryDecompositionL0> measure_paired_decomposition_l0(
    faiss::IndexHNSW &graph, faiss::Index &exact_storage,
    faiss::Index &pq_storage, const float *base_vectors,
    faiss::idx_t base_count, const float *queries, faiss::idx_t query_count,
    int dimension, faiss::idx_t k,
    const std::vector<faiss::idx_t> &ground_truth_ids,
    const faiss::SearchParametersHNSW &parameters,
    bool validate_native_identity = true,
    bool allow_verified_boundary_ties = false,
    bool stable_evaluation_order_ties = false);

} // namespace quant_hardness
