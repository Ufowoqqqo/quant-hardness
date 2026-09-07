#include "graph/faiss_shared_hnsw.h"
#include "metrics/ground_truth.h"
#include "metrics/recall.h"

#include <faiss/IndexFlat.h>
#include <faiss/IndexPQ.h>

#include <omp.h>
#include <sys/utsname.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

struct Config {
  std::uint32_t database_seed;
  std::uint32_t query_seed;
  std::uint32_t graph_seed;
  std::uint32_t pq_seed;
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

  Config config{static_cast<std::uint32_t>(get("database_seed")),
                static_cast<std::uint32_t>(get("query_seed")),
                static_cast<std::uint32_t>(get("graph_seed")),
                static_cast<std::uint32_t>(get("pq_seed")),
                get("base_vectors"),
                get("queries"),
                static_cast<int>(get("dimension")),
                get("k"),
                static_cast<int>(get("hnsw_m")),
                static_cast<int>(get("ef_construction")),
                static_cast<int>(get("ef_search")),
                static_cast<int>(get("pq_m")),
                static_cast<int>(get("pq_nbits"))};
  require(values.size() == 13, "config contains unknown keys");
  require(config.dimension % config.pq_m == 0,
          "dimension must be divisible by pq_m");
  require(config.k == 10, "Phase 1 metric is fixed to Recall@10");
  require(config.ef_search >= config.k, "ef_search must be at least k");
  require(config.k <= config.base_vectors, "k exceeds database size");
  return config;
}

std::vector<float> normal_vectors(std::uint32_t seed, faiss::idx_t count,
                                  int dimension) {
  std::mt19937 generator(seed);
  std::normal_distribution<float> distribution(0.0F, 1.0F);
  std::vector<float> vectors(static_cast<std::size_t>(count * dimension));
  for (float &value : vectors) {
    value = distribution(generator);
  }
  return vectors;
}

std::string utc_now() {
  const std::time_t now =
      std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  std::tm utc{};
  gmtime_r(&now, &utc);
  std::ostringstream output;
  output << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
  return output.str();
}

std::string hostname() {
  char value[256]{};
  if (gethostname(value, sizeof(value)) != 0) {
    return "unknown";
  }
  value[sizeof(value) - 1] = '\0';
  return value;
}

std::string kernel() {
  struct utsname info {};
  if (uname(&info) != 0) {
    return "unknown";
  }
  return std::string(info.sysname) + " " + info.release + " " + info.machine;
}

template <typename Integer>
void write_id_array(std::ostream &output, const Integer *ids,
                    std::size_t size) {
  output << '[';
  for (std::size_t i = 0; i < size; ++i) {
    if (i != 0) {
      output << ',';
    }
    output << ids[i];
  }
  output << ']';
}

void write_manifest(const std::filesystem::path &path, const Config &config,
                    const std::string &fingerprint,
                    const std::string &created_at) {
  std::ofstream output(path);
  require(output.good(), "cannot create manifest: " + path.string());
  output << "{\n"
         << "  \"schema_version\": 1,\n"
         << "  \"run_id\": \"phase1_synthetic_paired_recall_v1\",\n"
         << "  \"created_at_utc\": \"" << created_at << "\",\n"
         << "  \"git_commit\": \"" << QH_GIT_COMMIT << "\",\n"
         << "  \"dirty_worktree\": " << QH_GIT_DIRTY << ",\n"
         << "  \"faiss_commit\": \"" << QH_FAISS_COMMIT << "\",\n"
         << "  \"graph_fingerprint_schema\": \"faiss-hnsw-struct-v1\",\n"
         << "  \"graph_fingerprint\": \"" << fingerprint << "\",\n"
         << "  \"dataset\": {\n"
         << "    \"distribution\": \"independent standard normal FP32\",\n"
         << "    \"base_vectors\": " << config.base_vectors << ",\n"
         << "    \"queries\": " << config.queries << ",\n"
         << "    \"dimension\": " << config.dimension << ",\n"
         << "    \"database_seed\": " << config.database_seed << ",\n"
         << "    \"query_seed\": " << config.query_seed << "\n"
         << "  },\n"
         << "  \"graph\": {\n"
         << "    \"construction_distance\": \"FP32 squared L2\",\n"
         << "    \"construction_count\": 1,\n"
         << "    \"seed\": " << config.graph_seed << ",\n"
         << "    \"M\": " << config.hnsw_m << ",\n"
         << "    \"ef_construction\": " << config.ef_construction << "\n"
         << "  },\n"
         << "  \"search\": {\n"
         << "    \"k\": " << config.k << ",\n"
         << "    \"ef_search\": " << config.ef_search << ",\n"
         << "    \"bounded_queue\": true,\n"
         << "    \"check_relative_distance\": true,\n"
         << "    \"threads\": 1,\n"
         << "    \"query_order\": \"ascending query_id, identical for both "
            "modes\",\n"
         << "    \"result_ranking\": \"native traversal-distance ranking\"\n"
         << "  },\n"
         << "  \"pq\": {\n"
         << "    \"implementation\": \"FAISS IndexPQ asymmetric distance\",\n"
         << "    \"seed\": " << config.pq_seed << ",\n"
         << "    \"M\": " << config.pq_m << ",\n"
         << "    \"nbits\": " << config.pq_nbits << ",\n"
         << "    \"training_data\": \"all synthetic database vectors\"\n"
         << "  },\n"
         << "  \"ground_truth\": \"exhaustive FP32 squared L2 top-10\",\n"
         << "  \"unresolved_confounder\": \"PQ results combine "
            "traversal-decision and approximate final-ranking effects\",\n"
         << "  \"machine\": {\n"
         << "    \"hostname\": \"" << hostname() << "\",\n"
         << "    \"kernel\": \"" << kernel() << "\",\n"
         << "    \"hardware_concurrency\": "
         << std::thread::hardware_concurrency() << ",\n"
         << "    \"compiler\": \"" << __VERSION__ << "\"\n"
         << "  }\n"
         << "}\n";
  require(output.good(), "failed while writing manifest");
}

} // namespace

int main(int argc, char **argv) {
  try {
    require(argc == 3,
            "usage: faiss_paired_recall CONFIG NEW_OUTPUT_DIRECTORY");
    const Config config = read_config(argv[1]);
    const std::filesystem::path output_directory(argv[2]);
    require(!std::filesystem::exists(output_directory),
            "refusing to overwrite existing run directory: " +
                output_directory.string());
    omp_set_num_threads(1);

    const std::vector<float> base = normal_vectors(
        config.database_seed, config.base_vectors, config.dimension);
    const std::vector<float> queries =
        normal_vectors(config.query_seed, config.queries, config.dimension);

    const quant_hardness::GroundTruthResults ground_truth =
        quant_hardness::exhaustive_l2_top_k(base.data(), config.base_vectors,
                                            queries.data(), config.queries,
                                            config.dimension, config.k);

    faiss::IndexFlatL2 exact_storage(config.dimension);
    faiss::IndexHNSW graph(&exact_storage, config.hnsw_m);
    graph.hnsw.efConstruction = config.ef_construction;
    graph.hnsw.rng = faiss::RandomGenerator(config.graph_seed);
    graph.train(config.base_vectors, base.data());
    graph.add(config.base_vectors, base.data());
    require(graph.ntotal == config.base_vectors,
            "FP32 HNSW construction has wrong ntotal");
    require(graph.storage == &exact_storage,
            "FP32 graph did not retain exact storage");
    const std::string frozen_fingerprint =
        quant_hardness::graph_fingerprint(graph.hnsw);

    faiss::IndexPQ pq_storage(config.dimension, config.pq_m, config.pq_nbits,
                              faiss::METRIC_L2);
    pq_storage.pq.cp.seed = static_cast<int>(config.pq_seed);
    pq_storage.train(config.base_vectors, base.data());
    pq_storage.add(config.base_vectors, base.data());
    require(pq_storage.ntotal == exact_storage.ntotal,
            "PQ and FP32 storage IDs are not aligned");
    require(quant_hardness::graph_fingerprint(graph.hnsw) == frozen_fingerprint,
            "PQ preparation changed graph fingerprint");

    faiss::SearchParametersHNSW search_params;
    search_params.efSearch = config.ef_search;
    search_params.bounded_queue = true;
    search_params.check_relative_distance = true;

    std::vector<std::uint64_t> exact_order(
        static_cast<std::size_t>(config.queries));
    std::vector<std::uint64_t> pq_order(
        static_cast<std::size_t>(config.queries));
    for (std::size_t i = 0; i < exact_order.size(); ++i) {
      exact_order[i] = i;
      pq_order[i] = i;
    }
    quant_hardness::require_identical_query_order(exact_order, pq_order);

    require(quant_hardness::graph_fingerprint(graph.hnsw) == frozen_fingerprint,
            "fingerprint changed before exact search");
    const quant_hardness::SearchResults exact_results =
        quant_hardness::search_with_storage(graph, exact_storage,
                                            queries.data(), config.queries,
                                            config.k, search_params);
    require(quant_hardness::graph_fingerprint(graph.hnsw) == frozen_fingerprint,
            "fingerprint changed after exact search");

    const quant_hardness::SearchResults pq_results =
        quant_hardness::search_with_storage(graph, pq_storage, queries.data(),
                                            config.queries, config.k,
                                            search_params);
    require(graph.storage == &exact_storage,
            "exact storage was not restored after PQ search");
    require(quant_hardness::graph_fingerprint(graph.hnsw) == frozen_fingerprint,
            "fingerprint changed after PQ search");

    std::vector<double> exact_recall(static_cast<std::size_t>(config.queries));
    std::vector<double> pq_recall(static_cast<std::size_t>(config.queries));
    std::vector<double> deltas(static_cast<std::size_t>(config.queries));
    for (faiss::idx_t query = 0; query < config.queries; ++query) {
      const std::size_t offset = static_cast<std::size_t>(query * config.k);
      exact_recall[query] = quant_hardness::recall_at_k(
          std::span(exact_results.ids).subspan(offset, config.k),
          std::span(ground_truth.ids).subspan(offset, config.k), config.k);
      pq_recall[query] = quant_hardness::recall_at_k(
          std::span(pq_results.ids).subspan(offset, config.k),
          std::span(ground_truth.ids).subspan(offset, config.k), config.k);
      deltas[query] =
          quant_hardness::delta_recall(exact_recall[query], pq_recall[query]);
    }

    std::filesystem::create_directories(output_directory);
    const std::string created_at = utc_now();
    write_manifest(output_directory / "manifest.json", config,
                   frozen_fingerprint, created_at);

    std::ofstream rows(output_directory / "queries.jsonl");
    require(rows.good(), "cannot create queries.jsonl");
    rows << std::setprecision(17);
    for (faiss::idx_t query = 0; query < config.queries; ++query) {
      const std::size_t offset = static_cast<std::size_t>(query * config.k);
      rows << "{\"schema_version\":1,\"query_id\":" << query
           << ",\"query_order\":" << query
           << ",\"recall_exact_at_10\":" << exact_recall[query]
           << ",\"recall_pq_at_10\":" << pq_recall[query]
           << ",\"delta_recall_at_10\":" << deltas[query]
           << ",\"exact_result_ids\":";
      write_id_array(rows, exact_results.ids.data() + offset, config.k);
      rows << ",\"pq_result_ids\":";
      write_id_array(rows, pq_results.ids.data() + offset, config.k);
      rows << ",\"ground_truth_ids\":";
      write_id_array(rows, ground_truth.ids.data() + offset, config.k);
      rows << "}\n";
    }
    require(rows.good(), "failed while writing queries.jsonl");

    std::ofstream resolved(output_directory / "resolved_config.conf");
    require(resolved.good(), "cannot create resolved_config.conf");
    std::ifstream source_config(argv[1]);
    resolved << source_config.rdbuf();
    require(resolved.good(), "failed while copying resolved config");

    std::cout << "run_directory=" << output_directory.string() << '\n'
              << "queries_written=" << config.queries << '\n'
              << "graph_fingerprint=" << frozen_fingerprint << '\n'
              << "status=PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "status=FAIL\nerror=" << error.what() << '\n';
    return 1;
  }
}
