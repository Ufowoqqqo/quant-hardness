#include "graph/faiss_shared_hnsw.h"
#include "instrumentation/faiss_distance_recorder.h"
#include "instrumentation/faiss_level0_recorder.h"
#include "instrumentation/paired_decomposition_l0.h"
#include "metrics/candidate_oracle.h"
#include "metrics/ground_truth.h"
#include "metrics/recall.h"

#include <faiss/IndexFlat.h>
#include <faiss/IndexPQ.h>

#include <omp.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const std::string &message) {
  if (!condition) throw std::runtime_error(message);
}

std::vector<float> normal_vectors(std::uint32_t seed, std::size_t count) {
  std::mt19937 generator(seed);
  std::normal_distribution<float> distribution(0.0F, 1.0F);
  std::vector<float> values(count);
  for (float &value : values) value = distribution(generator);
  return values;
}

std::vector<float> direction(std::uint32_t seed, int dimension) {
  auto output = normal_vectors(seed, dimension);
  double squared = 0.0;
  for (float value : output) squared += static_cast<double>(value) * value;
  const double norm = std::sqrt(squared);
  for (float &value : output) value = static_cast<float>(value / norm);
  return output;
}

struct Fixture {
  static constexpr int dimension = 64;
  static constexpr faiss::idx_t base_count = 20000;
  std::vector<float> base;
  std::vector<float> latent;
  faiss::IndexFlatL2 storage{dimension};
  faiss::IndexHNSW graph{&storage, 16};

  Fixture(std::uint32_t database_seed, std::uint32_t query_seed,
          std::uint32_t graph_seed)
      : base(normal_vectors(database_seed, base_count * dimension)),
        latent(normal_vectors(query_seed, 500 * dimension)) {
    graph.hnsw.efConstruction = 80;
    graph.hnsw.rng = faiss::RandomGenerator(graph_seed);
    graph.add(base_count, base.data());
  }
};

std::vector<float> one_query(const Fixture &fixture, faiss::idx_t query_id,
                             double scale,
                             const std::vector<float> *shift_direction) {
  std::vector<float> query(Fixture::dimension);
  for (int component = 0; component < Fixture::dimension; ++component) {
    query[component] = fixture.latent[query_id * Fixture::dimension + component];
    if (shift_direction) {
      query[component] += static_cast<float>(
          scale * std::sqrt(Fixture::dimension) * (*shift_direction)[component]);
    } else {
      query[component] = static_cast<float>(scale * query[component]);
    }
  }
  return query;
}

void verify_known_anomaly(Fixture &fixture, const std::vector<float> &query) {
  constexpr faiss::idx_t k = 10;
  faiss::SearchParametersHNSW parameters;
  parameters.efSearch = 256;
  parameters.bounded_queue = true;
  parameters.check_relative_distance = true;
  const auto truth = quant_hardness::exhaustive_l2_top_k(
      fixture.base.data(), Fixture::base_count, query.data(), 1,
      Fixture::dimension, k);
  quant_hardness::RecordingIndex legacy(fixture.storage, query.data(), 1);
  const auto native = quant_hardness::search_with_storage(
      fixture.graph, legacy, query.data(), 1, k, parameters);
  const auto all_ids = legacy.sorted_evaluated_ids(0);
  const auto all_oracle = quant_hardness::exact_rerank_l2(
      fixture.base.data(), Fixture::base_count, Fixture::dimension,
      query.data(), all_ids, k);
  const double old_control =
      quant_hardness::recall_at_k(all_oracle.ids, truth.ids, k) -
      quant_hardness::recall_at_k(native.ids, truth.ids, k);
  require(std::abs(old_control - 0.1) < 1e-12,
          "known Phase 2A all-level anomaly was not reproduced");

  const auto phased = quant_hardness::search_with_level0_recording(
      fixture.graph, fixture.storage, query.data(), 1, k, parameters);
  const auto l0_oracle = quant_hardness::exact_rerank_l2(
      fixture.base.data(), Fixture::base_count, Fixture::dimension,
      query.data(), phased.level0_evaluated_ids[0], k);
  const double corrected_control =
      quant_hardness::recall_at_k(l0_oracle.ids, truth.ids, k) -
      quant_hardness::recall_at_k(phased.native.ids, truth.ids, k);
  require(corrected_control == 0.0,
          "known Phase 2A anomaly remains under L0 semantics");
}

void verify_small_scientific_subsets(Fixture &fixture,
                                     const std::vector<float> &shift) {
  constexpr faiss::idx_t query_count = 500;
  constexpr faiss::idx_t k = 10;
  faiss::IndexPQ pq(Fixture::dimension, 32, 8, faiss::METRIC_L2);
  pq.pq.cp.seed = 301009;
  pq.train(Fixture::base_count, fixture.base.data());
  pq.add(Fixture::base_count, fixture.base.data());
  faiss::SearchParametersHNSW parameters;
  parameters.efSearch = 256;
  parameters.bounded_queue = true;
  parameters.check_relative_distance = true;
  for (bool shifted : {false, true}) {
    std::vector<float> queries(query_count * Fixture::dimension);
    for (faiss::idx_t query_id = 0; query_id < query_count; ++query_id) {
      for (int component = 0; component < Fixture::dimension; ++component) {
        queries[query_id * Fixture::dimension + component] =
            fixture.latent[query_id * Fixture::dimension + component] +
            (shifted ? static_cast<float>(
                           std::sqrt(Fixture::dimension) * shift[component])
                     : 0.0F);
      }
    }
    const auto truth = quant_hardness::exhaustive_l2_top_k(
        fixture.base.data(), Fixture::base_count, queries.data(), query_count,
        Fixture::dimension, k);
    const auto rows = quant_hardness::measure_paired_decomposition_l0(
        fixture.graph, fixture.storage, pq, fixture.base.data(),
        Fixture::base_count, queries.data(), query_count, Fixture::dimension,
        k, truth.ids, parameters);
    double discovery = 0.0, exact = 0.0, pq_native = 0.0, pq_oracle = 0.0;
    for (const auto &row : rows) {
      require(row.delta_exact_control == 0.0,
              "subset exact control is nonzero");
      discovery += row.delta_discovery;
      exact += row.recall_exact_native;
      pq_native += row.recall_pq_native;
      pq_oracle += row.recall_pq_l0_oracle;
    }
    discovery /= query_count;
    exact /= query_count;
    pq_native /= query_count;
    pq_oracle /= query_count;
    const double recovery = (pq_oracle - pq_native) / (exact - pq_native);
    std::cout << "subset=" << (shifted ? "mean_alpha_1" : "iid")
              << " mean_delta_discovery=" << discovery
              << " rerank_recovery=" << recovery << '\n';
    require(std::abs(discovery) <= 0.03 && recovery >= 0.80,
            "L0 correction materially changed Phase 1/2A subset conclusion");
  }
}

} // namespace

int main() {
  try {
    omp_set_num_threads(1);
    Fixture replicate0(1009, 101009, 201009);
    const auto shift0 = direction(401009, Fixture::dimension);
    verify_known_anomaly(replicate0, one_query(replicate0, 342, 1.0, &shift0));
    verify_small_scientific_subsets(replicate0, shift0);
    Fixture replicate1(2003, 102003, 202003);
    verify_known_anomaly(replicate1, one_query(replicate1, 447, 2.0, nullptr));
    std::cout << "status=PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "status=FAIL\nerror=" << error.what() << '\n';
    return 1;
  }
}
