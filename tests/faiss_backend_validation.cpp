#include "graph/faiss_shared_hnsw.h"

#include <faiss/IndexFlat.h>
#include <faiss/IndexPQ.h>

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Config {
  std::uint32_t seed;
  faiss::idx_t base_vectors;
  faiss::idx_t queries;
  int dimension;
  faiss::idx_t k;
  int hnsw_m;
  int ef_construction;
  int ef_search;
  int pq_m;
  int pq_nbits;
};

void require(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

Config read_config(const std::string &path) {
  std::ifstream input(path);
  require(input.good(), "cannot open config: " + path);

  std::map<std::string, long long> values;
  std::string line;
  while (std::getline(input, line)) {
    const std::size_t comment = line.find('#');
    if (comment != std::string::npos) {
      line.erase(comment);
    }
    const std::size_t equals = line.find('=');
    if (equals == std::string::npos) {
      continue;
    }
    const std::string key = line.substr(0, equals);
    const std::string value = line.substr(equals + 1);
    require(!key.empty() && !value.empty(), "invalid config line");
    require(values.emplace(key, std::stoll(value)).second,
            "duplicate config key: " + key);
  }

  const auto get = [&](const std::string &key) {
    const auto it = values.find(key);
    require(it != values.end(), "missing config key: " + key);
    require(it->second > 0, "config value must be positive: " + key);
    return it->second;
  };

  Config config{static_cast<std::uint32_t>(get("seed")),
                get("base_vectors"),
                get("queries"),
                static_cast<int>(get("dimension")),
                get("k"),
                static_cast<int>(get("hnsw_m")),
                static_cast<int>(get("ef_construction")),
                static_cast<int>(get("ef_search")),
                static_cast<int>(get("pq_m")),
                static_cast<int>(get("pq_nbits"))};
  require(config.dimension % config.pq_m == 0,
          "dimension must be divisible by pq_m");
  require(config.k <= config.base_vectors, "k exceeds base vector count");
  require(values.size() == 10, "config contains unknown keys");
  return config;
}

std::vector<float> random_vectors(std::mt19937 &generator, faiss::idx_t count,
                                  int dimension) {
  std::normal_distribution<float> distribution(0.0F, 1.0F);
  std::vector<float> vectors(static_cast<std::size_t>(count * dimension));
  for (float &value : vectors) {
    value = distribution(generator);
  }
  return vectors;
}

void require_valid_results(const quant_hardness::SearchResults &results,
                           const Config &config) {
  require(results.ids.size() ==
              static_cast<std::size_t>(config.queries * config.k),
          "wrong result count");
  for (const faiss::idx_t id : results.ids) {
    require(id >= 0 && id < config.base_vectors, "invalid result ID");
  }
  for (const float distance : results.distances) {
    require(std::isfinite(distance) && distance >= 0.0F, "invalid L2 distance");
  }
}

void require_identical_results(const quant_hardness::SearchResults &left,
                               const quant_hardness::SearchResults &right,
                               const std::string &context) {
  require(left.ids == right.ids, context + ": result IDs differ");
  require(left.distances == right.distances,
          context + ": result distances differ");
}

void require_fingerprint_covers_structure(faiss::HNSW &hnsw) {
  const std::string original = quant_hardness::graph_fingerprint(hnsw);
  const auto changes_fingerprint = [&](auto &field, const auto replacement) {
    const auto saved = field;
    field = replacement;
    require(quant_hardness::graph_fingerprint(hnsw) != original,
            "fingerprint omitted a structural field");
    field = saved;
    require(quant_hardness::graph_fingerprint(hnsw) == original,
            "fingerprint did not restore after test mutation");
  };

  changes_fingerprint(hnsw.entry_point, hnsw.entry_point + 1);
  changes_fingerprint(hnsw.max_level, hnsw.max_level + 1);
  require(!hnsw.levels.empty(), "levels unexpectedly empty");
  changes_fingerprint(hnsw.levels[0], hnsw.levels[0] + 1);
  require(!hnsw.offsets.empty(), "offsets unexpectedly empty");
  changes_fingerprint(hnsw.offsets[0], hnsw.offsets[0] + 1);
  require(hnsw.neighbors.size() > 0, "neighbors unexpectedly empty");
  changes_fingerprint(hnsw.neighbors[0], hnsw.neighbors[0] + 1);
  require(!hnsw.cum_nneighbor_per_level.empty(),
          "neighbor capacities unexpectedly empty");
  changes_fingerprint(hnsw.cum_nneighbor_per_level[0],
                      hnsw.cum_nneighbor_per_level[0] + 1);
}

} // namespace

int main(int argc, char **argv) {
  try {
    require(argc == 2, "usage: faiss_backend_validation CONFIG");
    const Config config = read_config(argv[1]);
    omp_set_num_threads(1);

    std::mt19937 generator(config.seed);
    std::vector<float> base =
        random_vectors(generator, config.base_vectors, config.dimension);
    std::vector<float> queries =
        random_vectors(generator, config.queries, config.dimension);

    faiss::IndexFlatL2 exact_storage(config.dimension);
    faiss::IndexHNSW graph(&exact_storage, config.hnsw_m);
    graph.hnsw.efConstruction = config.ef_construction;
    graph.hnsw.rng = faiss::RandomGenerator(config.seed);
    graph.train(config.base_vectors, base.data());
    graph.add(config.base_vectors, base.data());
    require(graph.ntotal == config.base_vectors,
            "FP32 HNSW construction has wrong ntotal");
    require(graph.storage == &exact_storage,
            "FP32 HNSW did not retain exact storage");

    const std::string frozen_fingerprint =
        quant_hardness::graph_fingerprint(graph.hnsw);
    require_fingerprint_covers_structure(graph.hnsw);
    require(quant_hardness::graph_fingerprint(graph.hnsw) == frozen_fingerprint,
            "fingerprint self-test changed graph structure");

    faiss::IndexPQ pq_storage(config.dimension, config.pq_m, config.pq_nbits,
                              faiss::METRIC_L2);
    pq_storage.pq.cp.seed = static_cast<int>(config.seed);
    pq_storage.train(config.base_vectors, base.data());
    pq_storage.add(config.base_vectors, base.data());
    require(pq_storage.ntotal == exact_storage.ntotal,
            "PQ and FP32 storage IDs are not aligned");
    require(quant_hardness::graph_fingerprint(graph.hnsw) == frozen_fingerprint,
            "training PQ changed the frozen graph");

    std::unique_ptr<faiss::DistanceComputer> exact_distance(
        exact_storage.get_distance_computer());
    std::unique_ptr<faiss::DistanceComputer> pq_distance(
        pq_storage.get_distance_computer());
    exact_distance->set_query(queries.data());
    pq_distance->set_query(queries.data());
    std::size_t differing_distances = 0;
    double total_absolute_difference = 0.0;
    for (faiss::idx_t id = 0; id < config.base_vectors; ++id) {
      const float exact = (*exact_distance)(id);
      const float approximate = (*pq_distance)(id);
      const double difference = std::abs(static_cast<double>(exact) -
                                         static_cast<double>(approximate));
      total_absolute_difference += difference;
      differing_distances += difference > 1e-5 ? 1 : 0;
    }
    require(differing_distances > 0,
            "PQ traversal distances unexpectedly equal FP32 distances");

    faiss::SearchParametersHNSW search_params;
    search_params.efSearch = config.ef_search;
    search_params.bounded_queue = true;
    search_params.check_relative_distance = true;

    quant_hardness::SearchResults native_exact_results;
    native_exact_results.distances.resize(
        static_cast<std::size_t>(config.queries * config.k));
    native_exact_results.ids.resize(
        static_cast<std::size_t>(config.queries * config.k));
    graph.search(config.queries, queries.data(), config.k,
                 native_exact_results.distances.data(),
                 native_exact_results.ids.data(), &search_params);

    const auto exact_results = quant_hardness::search_with_storage(
        graph, exact_storage, queries.data(), config.queries, config.k,
        search_params);
    require_identical_results(native_exact_results, exact_results,
                              "native FP32 versus guarded FP32 search");
    require(quant_hardness::graph_fingerprint(graph.hnsw) == frozen_fingerprint,
            "exact traversal changed graph fingerprint");

    faiss::IndexFlatL2 identity_storage(config.dimension);
    identity_storage.add(config.base_vectors, base.data());
    const auto identity_results = quant_hardness::search_with_storage(
        graph, identity_storage, queries.data(), config.queries, config.k,
        search_params);
    require_identical_results(exact_results, identity_results,
                              "FP32 alternate-storage identity control");

    const auto pq_results = quant_hardness::search_with_storage(
        graph, pq_storage, queries.data(), config.queries, config.k,
        search_params);
    require(graph.storage == &exact_storage,
            "PQ traversal did not restore exact storage");
    require(quant_hardness::graph_fingerprint(graph.hnsw) == frozen_fingerprint,
            "PQ traversal changed graph fingerprint");

    require_valid_results(exact_results, config);
    require_valid_results(pq_results, config);

    std::cout << "faiss_commit=" << QH_FAISS_COMMIT << '\n'
              << "graph_fingerprint=" << frozen_fingerprint << '\n'
              << "base_vectors=" << config.base_vectors << '\n'
              << "queries=" << config.queries << '\n'
              << "dimension=" << config.dimension << '\n'
              << "pq_distance_differences=" << differing_distances << '/'
              << config.base_vectors << '\n'
              << "pq_mean_absolute_distance_difference="
              << total_absolute_difference / config.base_vectors << '\n'
              << "identity_storage_control=PASS\n"
              << "exact_first_result_id=" << exact_results.ids.front() << '\n'
              << "pq_first_result_id=" << pq_results.ids.front() << '\n'
              << "status=PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "status=FAIL\nerror=" << error.what() << '\n';
    return 1;
  }
}
