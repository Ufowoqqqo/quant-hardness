#include "graph/faiss_shared_hnsw.h"
#include "instrumentation/faiss_distance_recorder.h"
#include "metrics/candidate_oracle.h"
#include "metrics/ground_truth.h"
#include "metrics/recall.h"

#include <faiss/IndexFlat.h>
#include <faiss/IndexPQ.h>

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void test_exact_reranking_and_coverage() {
  const std::vector<float> base{0.0F, 3.0F, 10.0F, 20.0F, 2.5F};
  const float query = 2.0F;
  const std::vector<faiss::idx_t> candidates{0, 1, 3, 4};
  const auto reranked =
      quant_hardness::exact_rerank_l2(base.data(), 5, 1, &query, candidates, 3);
  require(reranked.ids == std::vector<faiss::idx_t>({4, 1, 0}),
          "exact reranking IDs are wrong");
  require(reranked.distances == std::vector<float>({0.25F, 1.0F, 4.0F}),
          "exact reranking distances are wrong");
  const std::vector<faiss::idx_t> truth{4, 1, 0};
  require(quant_hardness::candidate_coverage_at_k(candidates, truth, 3) == 1.0,
          "candidate coverage is wrong");
  require(quant_hardness::recall_at_k(reranked.ids, truth, 3) == 1.0,
          "oracle recall differs from candidate coverage");
}

std::vector<float> normal_vectors(std::uint32_t seed, std::size_t count) {
  std::mt19937 generator(seed);
  std::normal_distribution<float> distribution(0.0F, 1.0F);
  std::vector<float> values(count);
  for (float &value : values) {
    value = distribution(generator);
  }
  return values;
}

void test_instrumented_search_controls() {
  constexpr int dimension = 8;
  constexpr faiss::idx_t base_count = 512;
  constexpr faiss::idx_t query_count = 5;
  constexpr faiss::idx_t k = 10;
  const auto base = normal_vectors(41, base_count * dimension);
  const auto queries = normal_vectors(43, query_count * dimension);
  const auto truth = quant_hardness::exhaustive_l2_top_k(
      base.data(), base_count, queries.data(), query_count, dimension, k);

  faiss::IndexFlatL2 exact_storage(dimension);
  faiss::IndexHNSW graph(&exact_storage, 8);
  graph.hnsw.efConstruction = 40;
  graph.hnsw.rng = faiss::RandomGenerator(47);
  graph.add(base_count, base.data());
  const std::string fingerprint = quant_hardness::graph_fingerprint(graph.hnsw);

  faiss::IndexPQ pq_storage(dimension, 2, 4, faiss::METRIC_L2);
  pq_storage.pq.cp.seed = 53;
  pq_storage.train(base_count, base.data());
  pq_storage.add(base_count, base.data());

  faiss::SearchParametersHNSW parameters;
  parameters.efSearch = 24;
  parameters.bounded_queue = true;
  parameters.check_relative_distance = true;
  const auto exact_uninstrumented = quant_hardness::search_with_storage(
      graph, exact_storage, queries.data(), query_count, k, parameters);
  const auto pq_uninstrumented = quant_hardness::search_with_storage(
      graph, pq_storage, queries.data(), query_count, k, parameters);

  quant_hardness::RecordingIndex exact_recording(exact_storage, queries.data(),
                                                 query_count);
  const auto exact_instrumented = quant_hardness::search_with_storage(
      graph, exact_recording, queries.data(), query_count, k, parameters);
  quant_hardness::RecordingIndex pq_recording(pq_storage, queries.data(),
                                              query_count);
  const auto pq_instrumented = quant_hardness::search_with_storage(
      graph, pq_recording, queries.data(), query_count, k, parameters);

  require(exact_uninstrumented.ids == exact_instrumented.ids &&
              exact_uninstrumented.distances == exact_instrumented.distances,
          "instrumentation changed exact native results");
  require(pq_uninstrumented.ids == pq_instrumented.ids &&
              pq_uninstrumented.distances == pq_instrumented.distances,
          "instrumentation changed PQ native results");
  require(quant_hardness::graph_fingerprint(graph.hnsw) == fingerprint,
          "instrumented exact/PQ search changed graph fingerprint");

  for (faiss::idx_t query_id = 0; query_id < query_count; ++query_id) {
    const auto exact_ids = exact_recording.sorted_evaluated_ids(query_id);
    const auto pq_ids = pq_recording.sorted_evaluated_ids(query_id);
    require(std::adjacent_find(exact_ids.begin(), exact_ids.end()) ==
                    exact_ids.end() &&
                std::adjacent_find(pq_ids.begin(), pq_ids.end()) ==
                    pq_ids.end(),
            "recorded evaluated IDs are not unique");
    const auto ground_truth = std::span(truth.ids).subspan(query_id * k, k);
    const float *query = queries.data() + query_id * dimension;
    const auto exact_oracle = quant_hardness::exact_rerank_l2(
        base.data(), base_count, dimension, query, exact_ids, k);
    const auto pq_oracle = quant_hardness::exact_rerank_l2(
        base.data(), base_count, dimension, query, pq_ids, k);
    require(std::abs(
                quant_hardness::recall_at_k(exact_oracle.ids, ground_truth, k) -
                quant_hardness::candidate_coverage_at_k(exact_ids, ground_truth,
                                                        k)) < 1e-12,
            "exact oracle recall differs from coverage");
    require(
        std::abs(quant_hardness::recall_at_k(pq_oracle.ids, ground_truth, k) -
                 quant_hardness::candidate_coverage_at_k(pq_ids, ground_truth,
                                                         k)) < 1e-12,
        "PQ oracle recall differs from coverage");
  }
}

} // namespace

int main() {
  try {
    omp_set_num_threads(1);
    test_exact_reranking_and_coverage();
    test_instrumented_search_controls();
    std::cout << "status=PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "status=FAIL\nerror=" << error.what() << '\n';
    return 1;
  }
}
