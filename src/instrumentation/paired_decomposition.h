#pragma once

#include "graph/faiss_shared_hnsw.h"

#include <faiss/Index.h>
#include <faiss/IndexHNSW.h>

#include <cstddef>
#include <vector>

namespace quant_hardness {

struct QueryDecomposition {
  double recall_exact_native;
  double recall_pq_native;
  double recall_exact_oracle;
  double recall_pq_oracle;
  double coverage_exact;
  double coverage_pq;
  double delta_total;
  double delta_discovery;
  double delta_ranking;
  double delta_exact_control;
  std::size_t evaluated_intersection_size;
  double evaluated_jaccard;
  std::vector<faiss::idx_t> ground_truth_ids;
  std::vector<faiss::idx_t> exact_native_ids;
  std::vector<faiss::idx_t> pq_native_ids;
  std::vector<faiss::idx_t> exact_oracle_ids;
  std::vector<faiss::idx_t> pq_oracle_ids;
  std::vector<faiss::idx_t> exact_evaluated_ids;
  std::vector<faiss::idx_t> pq_evaluated_ids;
};

std::vector<QueryDecomposition> measure_paired_decomposition(
    faiss::IndexHNSW &graph, faiss::Index &exact_storage,
    faiss::Index &pq_storage, const float *base_vectors,
    faiss::idx_t base_count, const float *queries, faiss::idx_t query_count,
    int dimension, faiss::idx_t k,
    const std::vector<faiss::idx_t> &ground_truth_ids,
    const faiss::SearchParametersHNSW &parameters,
    bool validate_instrumentation_identity = true);

} // namespace quant_hardness
