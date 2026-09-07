#include "graph/faiss_shared_hnsw.h"
#include "metrics/ground_truth.h"
#include "metrics/recall.h"

#include <faiss/IndexFlat.h>
#include <faiss/IndexPQ.h>
#include <faiss/impl/DistanceComputer.h>

#include <omp.h>
#include <sys/utsname.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

struct PQConfig {
  int subquantizers;
  int bits;

  std::string id() const {
    return "pq_m" + std::to_string(subquantizers) + "_nbits" +
           std::to_string(bits);
  }
};

struct Config {
  std::uint32_t database_seed;
  std::uint32_t query_seed;
  std::uint32_t graph_seed;
  std::uint32_t pq_seed;
  std::uint32_t distance_sample_seed;
  faiss::idx_t base_vectors;
  faiss::idx_t queries;
  int dimension;
  faiss::idx_t k;
  int hnsw_m;
  int ef_construction;
  std::vector<int> ef_search_values;
  std::vector<PQConfig> pq_configs;
  int sample_queries;
  int distance_pairs_per_query;
  int order_pairs_per_query;
};

void require(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

std::vector<std::string> split(const std::string &value, char delimiter) {
  std::vector<std::string> parts;
  std::stringstream stream(value);
  std::string part;
  while (std::getline(stream, part, delimiter)) {
    require(!part.empty(), "empty list item in config");
    parts.push_back(part);
  }
  return parts;
}

Config read_config(const std::string &path) {
  std::ifstream input(path);
  require(input.good(), "cannot open config: " + path);
  std::map<std::string, std::string> values;
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
    require(
        values.emplace(line.substr(0, equals), line.substr(equals + 1)).second,
        "duplicate config key");
  }
  const auto get = [&](const std::string &key) -> const std::string & {
    const auto it = values.find(key);
    require(it != values.end() && !it->second.empty(),
            "missing config key: " + key);
    return it->second;
  };
  const auto positive = [&](const std::string &key) {
    const long long parsed = std::stoll(get(key));
    require(parsed > 0, "config value must be positive: " + key);
    return parsed;
  };

  Config config{static_cast<std::uint32_t>(positive("database_seed")),
                static_cast<std::uint32_t>(positive("query_seed")),
                static_cast<std::uint32_t>(positive("graph_seed")),
                static_cast<std::uint32_t>(positive("pq_seed")),
                static_cast<std::uint32_t>(positive("distance_sample_seed")),
                positive("base_vectors"),
                positive("queries"),
                static_cast<int>(positive("dimension")),
                positive("k"),
                static_cast<int>(positive("hnsw_m")),
                static_cast<int>(positive("ef_construction")),
                {},
                {},
                static_cast<int>(positive("sample_queries")),
                static_cast<int>(positive("distance_pairs_per_query")),
                static_cast<int>(positive("order_pairs_per_query"))};
  for (const std::string &item : split(get("ef_search_values"), ',')) {
    const int ef = std::stoi(item);
    require(ef >= config.k, "efSearch must be at least k");
    config.ef_search_values.push_back(ef);
  }
  for (const std::string &item : split(get("pq_configs"), ',')) {
    const std::size_t separator = item.find('x');
    require(separator != std::string::npos, "PQ config must be Mxnbits");
    PQConfig pq{std::stoi(item.substr(0, separator)),
                std::stoi(item.substr(separator + 1))};
    require(pq.subquantizers > 0 && config.dimension % pq.subquantizers == 0,
            "invalid PQ subquantizer count");
    require(pq.bits > 0 && pq.bits <= 8, "PQ bits must be in [1, 8]");
    config.pq_configs.push_back(pq);
  }
  require(values.size() == 16, "config contains unknown keys");
  require(config.k == 10, "calibration metric is fixed to Recall@10");
  require(config.sample_queries <= config.queries,
          "sample_queries exceeds query count");
  require(std::set<int>(config.ef_search_values.begin(),
                        config.ef_search_values.end())
                  .size() == config.ef_search_values.size(),
          "duplicate efSearch value");
  std::set<std::string> pq_ids;
  for (const PQConfig &pq : config.pq_configs) {
    require(pq_ids.insert(pq.id()).second, "duplicate PQ config");
  }
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

float exact_squared_l2(const float *left, const float *right, int dimension) {
  float total = 0.0F;
  for (int component = 0; component < dimension; ++component) {
    const float difference = left[component] - right[component];
    total += difference * difference;
  }
  return total;
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
  return gethostname(value, sizeof(value)) == 0 ? value : "unknown";
}

std::string kernel() {
  struct utsname info {};
  return uname(&info) == 0 ? std::string(info.sysname) + " " + info.release +
                                 " " + info.machine
                           : "unknown";
}

template <typename Integer>
void write_id_array(std::ostream &output, const Integer *ids,
                    std::size_t size) {
  output << '[';
  for (std::size_t i = 0; i < size; ++i) {
    output << (i == 0 ? "" : ",") << ids[i];
  }
  output << ']';
}

void write_manifest(const std::filesystem::path &path, const Config &config,
                    const std::string &fingerprint) {
  std::ofstream output(path);
  require(output.good(), "cannot create manifest");
  output << "{\n  \"schema_version\": 1,\n"
         << "  \"run_id\": \"phase1_calibration_v1\",\n"
         << "  \"created_at_utc\": \"" << utc_now() << "\",\n"
         << "  \"git_commit\": \"" << QH_GIT_COMMIT << "\",\n"
         << "  \"dirty_worktree\": " << QH_GIT_DIRTY << ",\n"
         << "  \"faiss_commit\": \"" << QH_FAISS_COMMIT << "\",\n"
         << "  \"graph_fingerprint_schema\": \"faiss-hnsw-struct-v1\",\n"
         << "  \"graph_fingerprint\": \"" << fingerprint << "\",\n"
         << "  \"dataset\": {\"distribution\": \"independent standard "
            "normal FP32\", \"base_vectors\": "
         << config.base_vectors << ", \"queries\": " << config.queries
         << ", \"dimension\": " << config.dimension
         << ", \"database_seed\": " << config.database_seed
         << ", \"query_seed\": " << config.query_seed << "},\n"
         << "  \"graph\": {\"construction_count\": 1, "
            "\"construction_distance\": \"FP32 squared L2\", \"seed\": "
         << config.graph_seed << ", \"M\": " << config.hnsw_m
         << ", \"ef_construction\": " << config.ef_construction << "},\n"
         << "  \"search\": {\"k\": " << config.k << ", \"ef_search_values\": [";
  for (std::size_t i = 0; i < config.ef_search_values.size(); ++i) {
    output << (i == 0 ? "" : ",") << config.ef_search_values[i];
  }
  output << "], \"bounded_queue\": true, \"check_relative_distance\": "
            "true, \"threads\": 1, \"query_order\": \"ascending query_id"
            "\"},\n  \"pq_configs\": [";
  for (std::size_t i = 0; i < config.pq_configs.size(); ++i) {
    const PQConfig &pq = config.pq_configs[i];
    output << (i == 0 ? "" : ",") << "{\"id\":\"" << pq.id()
           << "\",\"M\":" << pq.subquantizers << ",\"nbits\":" << pq.bits
           << ",\"code_bytes\":" << (pq.subquantizers * pq.bits + 7) / 8
           << ",\"training_size\":" << config.base_vectors
           << ",\"seed\":" << config.pq_seed << '}';
  }
  output << "],\n  \"distance_sampling\": {\"query_selection\": \"first "
         << config.sample_queries << " query IDs\", \"database_id_seed\": "
         << config.distance_sample_seed << ", \"distance_pairs_per_query\": "
         << config.distance_pairs_per_query
         << ", \"order_pairs_per_query\": " << config.order_pairs_per_query
         << ", \"relative_error\": \"absolute error / exact distance for "
            "exact distance > 1e-12\", \"inversion\": \"strict reversal; "
            "PQ ties reported separately\"},\n"
         << "  \"ground_truth\": \"exhaustive FP32 squared L2 top-10\",\n"
         << "  \"unresolved_confounder\": \"PQ results combine traversal "
            "and native approximate-ranking effects\",\n"
         << "  \"machine\": {\"hostname\": \"" << hostname()
         << "\", \"kernel\": \"" << kernel() << "\", \"hardware_concurrency\": "
         << std::thread::hardware_concurrency() << ", \"compiler\": \""
         << __VERSION__ << "\"}\n}\n";
  require(output.good(), "failed while writing manifest");
}

struct SampleIds {
  std::vector<faiss::idx_t> distance_ids;
  std::vector<std::pair<faiss::idx_t, faiss::idx_t>> order_ids;
};

std::vector<SampleIds> make_samples(const Config &config) {
  std::mt19937 generator(config.distance_sample_seed);
  std::vector<SampleIds> samples(config.sample_queries);
  for (SampleIds &per_query : samples) {
    for (int i = 0; i < config.distance_pairs_per_query; ++i) {
      per_query.distance_ids.push_back(generator() % config.base_vectors);
    }
    for (int i = 0; i < config.order_pairs_per_query; ++i) {
      faiss::idx_t first = generator() % config.base_vectors;
      faiss::idx_t second = generator() % config.base_vectors;
      while (second == first) {
        second = generator() % config.base_vectors;
      }
      per_query.order_ids.emplace_back(first, second);
    }
  }
  return samples;
}

} // namespace

int main(int argc, char **argv) {
  try {
    require(argc == 3, "usage: faiss_calibration CONFIG NEW_OUTPUT_DIRECTORY");
    const Config config = read_config(argv[1]);
    const std::filesystem::path output_directory(argv[2]);
    require(!std::filesystem::exists(output_directory),
            "refusing to overwrite existing output directory");
    omp_set_num_threads(1);

    const std::vector<float> base = normal_vectors(
        config.database_seed, config.base_vectors, config.dimension);
    const std::vector<float> queries =
        normal_vectors(config.query_seed, config.queries, config.dimension);
    const auto truth = quant_hardness::exhaustive_l2_top_k(
        base.data(), config.base_vectors, queries.data(), config.queries,
        config.dimension, config.k);

    faiss::IndexFlatL2 exact_storage(config.dimension);
    faiss::IndexHNSW graph(&exact_storage, config.hnsw_m);
    graph.hnsw.efConstruction = config.ef_construction;
    graph.hnsw.rng = faiss::RandomGenerator(config.graph_seed);
    graph.add(config.base_vectors, base.data());
    const std::string frozen_fingerprint =
        quant_hardness::graph_fingerprint(graph.hnsw);

    std::map<int, quant_hardness::SearchResults> exact_results;
    std::map<int, std::vector<double>> exact_recalls;
    for (const int ef : config.ef_search_values) {
      faiss::SearchParametersHNSW params;
      params.efSearch = ef;
      params.bounded_queue = true;
      params.check_relative_distance = true;
      exact_results.emplace(ef, quant_hardness::search_with_storage(
                                    graph, exact_storage, queries.data(),
                                    config.queries, config.k, params));
      std::vector<double> recalls(config.queries);
      for (faiss::idx_t query = 0; query < config.queries; ++query) {
        const std::size_t offset = query * config.k;
        recalls[query] = quant_hardness::recall_at_k(
            std::span(exact_results.at(ef).ids).subspan(offset, config.k),
            std::span(truth.ids).subspan(offset, config.k), config.k);
      }
      exact_recalls.emplace(ef, std::move(recalls));
      require(quant_hardness::graph_fingerprint(graph.hnsw) ==
                  frozen_fingerprint,
              "graph changed during exact efSearch sweep");
    }

    std::filesystem::create_directories(output_directory);
    write_manifest(output_directory / "manifest.json", config,
                   frozen_fingerprint);
    std::ifstream source_config(argv[1]);
    std::ofstream resolved(output_directory / "resolved_config.conf");
    resolved << source_config.rdbuf();

    std::ofstream exact_rows(output_directory / "exact_queries.jsonl");
    std::ofstream paired_rows(output_directory / "paired_queries.jsonl");
    std::ofstream distance_rows(output_directory / "distance_samples.jsonl");
    std::ofstream order_rows(output_directory / "order_samples.jsonl");
    require(exact_rows.good() && paired_rows.good() && distance_rows.good() &&
                order_rows.good(),
            "cannot create calibration raw outputs");
    exact_rows << std::setprecision(17);
    paired_rows << std::setprecision(17);
    distance_rows << std::setprecision(17);
    order_rows << std::setprecision(17);

    for (const int ef : config.ef_search_values) {
      for (faiss::idx_t query = 0; query < config.queries; ++query) {
        const std::size_t offset = query * config.k;
        exact_rows << "{\"query_id\":" << query << ",\"query_order\":" << query
                   << ",\"ef_search\":" << ef
                   << ",\"recall_exact_at_10\":" << exact_recalls.at(ef)[query]
                   << ",\"exact_result_ids\":";
        write_id_array(exact_rows, exact_results.at(ef).ids.data() + offset,
                       config.k);
        exact_rows << ",\"ground_truth_ids\":";
        write_id_array(exact_rows, truth.ids.data() + offset, config.k);
        exact_rows << "}\n";
      }
    }

    const std::vector<SampleIds> samples = make_samples(config);
    for (const PQConfig &pq_config : config.pq_configs) {
      faiss::IndexPQ pq(config.dimension, pq_config.subquantizers,
                        pq_config.bits, faiss::METRIC_L2);
      pq.pq.cp.seed = static_cast<int>(config.pq_seed);
      pq.train(config.base_vectors, base.data());
      pq.add(config.base_vectors, base.data());
      require(pq.ntotal == exact_storage.ntotal,
              "PQ and FP32 storage IDs differ");

      for (const int ef : config.ef_search_values) {
        faiss::SearchParametersHNSW params;
        params.efSearch = ef;
        params.bounded_queue = true;
        params.check_relative_distance = true;
        const auto pq_results = quant_hardness::search_with_storage(
            graph, pq, queries.data(), config.queries, config.k, params);
        for (faiss::idx_t query = 0; query < config.queries; ++query) {
          const std::size_t offset = query * config.k;
          const double pq_recall = quant_hardness::recall_at_k(
              std::span(pq_results.ids).subspan(offset, config.k),
              std::span(truth.ids).subspan(offset, config.k), config.k);
          const double delta = quant_hardness::delta_recall(
              exact_recalls.at(ef)[query], pq_recall);
          paired_rows << "{\"query_id\":" << query
                      << ",\"query_order\":" << query << ",\"ef_search\":" << ef
                      << ",\"pq_id\":\"" << pq_config.id()
                      << "\",\"pq_m\":" << pq_config.subquantizers
                      << ",\"pq_nbits\":" << pq_config.bits
                      << ",\"recall_exact_at_10\":"
                      << exact_recalls.at(ef)[query]
                      << ",\"recall_pq_at_10\":" << pq_recall
                      << ",\"delta_recall_at_10\":" << delta
                      << ",\"exact_result_ids\":";
          write_id_array(paired_rows, exact_results.at(ef).ids.data() + offset,
                         config.k);
          paired_rows << ",\"pq_result_ids\":";
          write_id_array(paired_rows, pq_results.ids.data() + offset, config.k);
          paired_rows << ",\"ground_truth_ids\":";
          write_id_array(paired_rows, truth.ids.data() + offset, config.k);
          paired_rows << "}\n";
        }
      }

      std::unique_ptr<faiss::DistanceComputer> computer(
          pq.get_distance_computer());
      for (int query = 0; query < config.sample_queries; ++query) {
        const float *query_vector =
            queries.data() + static_cast<std::size_t>(query * config.dimension);
        computer->set_query(query_vector);
        for (std::size_t sample = 0;
             sample < samples[query].distance_ids.size(); ++sample) {
          const faiss::idx_t database_id = samples[query].distance_ids[sample];
          const float exact = exact_squared_l2(
              query_vector,
              base.data() +
                  static_cast<std::size_t>(database_id * config.dimension),
              config.dimension);
          const float approximate = (*computer)(database_id);
          distance_rows << "{\"pq_id\":\"" << pq_config.id()
                        << "\",\"query_id\":" << query
                        << ",\"sample_id\":" << sample
                        << ",\"database_id\":" << database_id
                        << ",\"exact_distance\":" << exact
                        << ",\"pq_distance\":" << approximate
                        << ",\"absolute_error\":"
                        << std::abs(approximate - exact);
          if (exact > 1e-12F) {
            distance_rows << ",\"absolute_relative_error\":"
                          << std::abs(approximate - exact) / exact;
          } else {
            distance_rows << ",\"absolute_relative_error\":null";
          }
          distance_rows << "}\n";
        }
        for (std::size_t sample = 0; sample < samples[query].order_ids.size();
             ++sample) {
          const auto [first, second] = samples[query].order_ids[sample];
          const float exact_first = exact_squared_l2(
              query_vector,
              base.data() + static_cast<std::size_t>(first * config.dimension),
              config.dimension);
          const float exact_second = exact_squared_l2(
              query_vector,
              base.data() + static_cast<std::size_t>(second * config.dimension),
              config.dimension);
          const float pq_first = (*computer)(first);
          const float pq_second = (*computer)(second);
          const bool inversion =
              (exact_first < exact_second && pq_first > pq_second) ||
              (exact_first > exact_second && pq_first < pq_second);
          order_rows << "{\"pq_id\":\"" << pq_config.id()
                     << "\",\"query_id\":" << query
                     << ",\"sample_id\":" << sample
                     << ",\"first_database_id\":" << first
                     << ",\"second_database_id\":" << second
                     << ",\"exact_first\":" << exact_first
                     << ",\"exact_second\":" << exact_second
                     << ",\"pq_first\":" << pq_first
                     << ",\"pq_second\":" << pq_second
                     << ",\"strict_inversion\":"
                     << (inversion ? "true" : "false") << ",\"pq_tie\":"
                     << (pq_first == pq_second ? "true" : "false") << "}\n";
        }
      }
      require(quant_hardness::graph_fingerprint(graph.hnsw) ==
                  frozen_fingerprint,
              "graph changed during PQ calibration");
    }
    require(exact_rows.good() && paired_rows.good() && distance_rows.good() &&
                order_rows.good() && resolved.good(),
            "failed while writing calibration outputs");

    std::cout << "run_directory=" << output_directory.string() << '\n'
              << "graph_fingerprint=" << frozen_fingerprint << '\n'
              << "exact_query_rows="
              << config.queries * config.ef_search_values.size() << '\n'
              << "paired_query_rows="
              << config.queries * config.ef_search_values.size() *
                     config.pq_configs.size()
              << '\n'
              << "status=PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "status=FAIL\nerror=" << error.what() << '\n';
    return 1;
  }
}
