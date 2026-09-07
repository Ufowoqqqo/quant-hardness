#include "graph/faiss_shared_hnsw.h"
#include "instrumentation/faiss_distance_recorder.h"
#include "metrics/candidate_oracle.h"
#include "metrics/ground_truth.h"
#include "metrics/recall.h"

#include <faiss/IndexFlat.h>
#include <faiss/IndexPQ.h>

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
#include <random>
#include <set>
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
  std::vector<int> ef_search_values;
  int pq_m;
  int pq_nbits;
  std::string expected_graph_fingerprint;
};

void require(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

std::vector<std::string> split(const std::string &value, char delimiter) {
  std::stringstream stream(value);
  std::string item;
  std::vector<std::string> items;
  while (std::getline(stream, item, delimiter)) {
    require(!item.empty(), "empty config-list item");
    items.push_back(item);
  }
  return items;
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
    const auto iterator = values.find(key);
    require(iterator != values.end() && !iterator->second.empty(),
            "missing config key: " + key);
    return iterator->second;
  };
  const auto positive = [&](const std::string &key) {
    const long long value = std::stoll(get(key));
    require(value > 0, "config value must be positive: " + key);
    return value;
  };
  Config config{static_cast<std::uint32_t>(positive("database_seed")),
                static_cast<std::uint32_t>(positive("query_seed")),
                static_cast<std::uint32_t>(positive("graph_seed")),
                static_cast<std::uint32_t>(positive("pq_seed")),
                positive("base_vectors"),
                positive("queries"),
                static_cast<int>(positive("dimension")),
                positive("k"),
                static_cast<int>(positive("hnsw_m")),
                static_cast<int>(positive("ef_construction")),
                {},
                static_cast<int>(positive("pq_m")),
                static_cast<int>(positive("pq_nbits")),
                get("expected_graph_fingerprint")};
  for (const std::string &item : split(get("ef_search_values"), ',')) {
    config.ef_search_values.push_back(std::stoi(item));
  }
  require(values.size() == 14, "config contains unknown keys");
  require(config.k == 10, "decomposition is fixed to Recall@10");
  require(config.dimension % config.pq_m == 0,
          "dimension must be divisible by pq_m");
  require(config.pq_m == 32 && config.pq_nbits == 8,
          "primary decomposition requires PQ32x8");
  require(config.ef_search_values == std::vector<int>({160, 256, 384}),
          "primary decomposition requires efSearch 160,256,384");
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
  return gethostname(value, sizeof(value)) == 0 ? value : "unknown";
}

std::string kernel() {
  struct utsname info {};
  return uname(&info) == 0 ? std::string(info.sysname) + " " + info.release +
                                 " " + info.machine
                           : "unknown";
}

template <typename Integer>
void write_array(std::ostream &output, const Integer *values,
                 std::size_t size) {
  output << '[';
  for (std::size_t i = 0; i < size; ++i) {
    output << (i == 0 ? "" : ",") << values[i];
  }
  output << ']';
}

std::size_t intersection_size(std::span<const faiss::idx_t> first,
                              std::span<const faiss::idx_t> second) {
  std::size_t intersection = 0;
  std::size_t left = 0;
  std::size_t right = 0;
  while (left < first.size() && right < second.size()) {
    if (first[left] == second[right]) {
      ++intersection;
      ++left;
      ++right;
    } else if (first[left] < second[right]) {
      ++left;
    } else {
      ++right;
    }
  }
  return intersection;
}

void write_manifest(const std::filesystem::path &path, const Config &config,
                    const std::string &fingerprint) {
  std::ofstream output(path);
  require(output.good(), "cannot create manifest");
  output << "{\n"
         << "  \"schema_version\": 1,\n"
         << "  \"run_id\": \"phase1_decomposition_v1\",\n"
         << "  \"created_at_utc\": \"" << utc_now() << "\",\n"
         << "  \"git_commit\": \"" << QH_GIT_COMMIT << "\",\n"
         << "  \"dirty_worktree\": " << QH_GIT_DIRTY << ",\n"
         << "  \"faiss_commit\": \"" << QH_FAISS_COMMIT << "\",\n"
         << "  \"calibration_reference_commit\": \"776bc6a\",\n"
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
         << "  \"search\": {\"k\": 10, \"ef_search_values\": "
            "[160,256,384], \"threads\": 1, \"bounded_queue\": true, "
            "\"check_relative_distance\": true, \"query_order\": "
            "\"ascending query_id\"},\n"
         << "  \"pq\": {\"M\": " << config.pq_m
         << ", \"nbits\": " << config.pq_nbits
         << ", \"code_bytes\": 32, \"seed\": " << config.pq_seed
         << ", \"training_size\": " << config.base_vectors << "},\n"
         << "  \"distance_evaluation_semantics\": \"unique database IDs "
            "passed to DistanceComputer::operator() or distances_batch_4 "
            "after set_query during HNSW search; symmetric database-to-"
            "database distances excluded\",\n"
         << "  \"candidate_id_order\": \"ascending database ID; evaluation "
            "order intentionally not retained\",\n"
         << "  \"ground_truth\": \"exhaustive FP32 squared L2 top-10\",\n"
         << "  \"machine\": {\"hostname\": \"" << hostname()
         << "\", \"kernel\": \"" << kernel() << "\", \"hardware_concurrency\": "
         << std::thread::hardware_concurrency() << ", \"compiler\": \""
         << __VERSION__ << "\"}\n"
         << "}\n";
}

} // namespace

int main(int argc, char **argv) {
  try {
    require(argc == 3,
            "usage: faiss_decomposition CONFIG NEW_OUTPUT_DIRECTORY");
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
    const std::string fingerprint =
        quant_hardness::graph_fingerprint(graph.hnsw);
    require(fingerprint == config.expected_graph_fingerprint,
            "graph fingerprint differs from calibration");

    faiss::IndexPQ pq_storage(config.dimension, config.pq_m, config.pq_nbits,
                              faiss::METRIC_L2);
    pq_storage.pq.cp.seed = static_cast<int>(config.pq_seed);
    pq_storage.train(config.base_vectors, base.data());
    pq_storage.add(config.base_vectors, base.data());
    require(pq_storage.ntotal == exact_storage.ntotal,
            "PQ and exact storage IDs differ");
    require(quant_hardness::graph_fingerprint(graph.hnsw) == fingerprint,
            "PQ preparation changed graph");

    std::filesystem::create_directories(output_directory);
    write_manifest(output_directory / "manifest.json", config, fingerprint);
    std::ifstream source_config(argv[1]);
    std::ofstream resolved(output_directory / "resolved_config.conf");
    resolved << source_config.rdbuf();
    std::ofstream rows(output_directory / "queries.jsonl");
    require(rows.good() && resolved.good(), "cannot create raw outputs");
    rows << std::setprecision(17);

    for (const int ef_search : config.ef_search_values) {
      faiss::SearchParametersHNSW parameters;
      parameters.efSearch = ef_search;
      parameters.bounded_queue = true;
      parameters.check_relative_distance = true;

      quant_hardness::RecordingIndex exact_recording(
          exact_storage, queries.data(), config.queries);
      const auto exact_native = quant_hardness::search_with_storage(
          graph, exact_recording, queries.data(), config.queries, config.k,
          parameters);
      quant_hardness::RecordingIndex pq_recording(pq_storage, queries.data(),
                                                  config.queries);
      const auto pq_native = quant_hardness::search_with_storage(
          graph, pq_recording, queries.data(), config.queries, config.k,
          parameters);
      require(quant_hardness::graph_fingerprint(graph.hnsw) == fingerprint,
              "instrumented search changed graph");

      for (faiss::idx_t query_id = 0; query_id < config.queries; ++query_id) {
        const std::size_t offset = query_id * config.k;
        const auto ground_truth =
            std::span(truth.ids).subspan(offset, config.k);
        const std::vector<faiss::idx_t> exact_evaluated =
            exact_recording.sorted_evaluated_ids(query_id);
        const std::vector<faiss::idx_t> pq_evaluated =
            pq_recording.sorted_evaluated_ids(query_id);
        require(exact_evaluated.size() >= static_cast<std::size_t>(config.k) &&
                    pq_evaluated.size() >= static_cast<std::size_t>(config.k),
                "fewer than k unique distances evaluated");
        const float *query = queries.data() + query_id * config.dimension;
        const auto exact_oracle = quant_hardness::exact_rerank_l2(
            base.data(), config.base_vectors, config.dimension, query,
            exact_evaluated, config.k);
        const auto pq_oracle = quant_hardness::exact_rerank_l2(
            base.data(), config.base_vectors, config.dimension, query,
            pq_evaluated, config.k);

        const double recall_exact_native = quant_hardness::recall_at_k(
            std::span(exact_native.ids).subspan(offset, config.k), ground_truth,
            config.k);
        const double recall_pq_native = quant_hardness::recall_at_k(
            std::span(pq_native.ids).subspan(offset, config.k), ground_truth,
            config.k);
        const double recall_exact_oracle = quant_hardness::recall_at_k(
            exact_oracle.ids, ground_truth, config.k);
        const double recall_pq_oracle =
            quant_hardness::recall_at_k(pq_oracle.ids, ground_truth, config.k);
        const double coverage_exact = quant_hardness::candidate_coverage_at_k(
            exact_evaluated, ground_truth, config.k);
        const double coverage_pq = quant_hardness::candidate_coverage_at_k(
            pq_evaluated, ground_truth, config.k);
        require(std::abs(recall_exact_oracle - coverage_exact) < 1e-12,
                "exact oracle recall differs from candidate coverage");
        require(std::abs(recall_pq_oracle - coverage_pq) < 1e-12,
                "PQ oracle recall differs from candidate coverage");

        const double delta_total = recall_exact_native - recall_pq_native;
        const double delta_discovery = recall_exact_oracle - recall_pq_oracle;
        const double delta_ranking = recall_pq_oracle - recall_pq_native;
        const double delta_exact_control =
            recall_exact_oracle - recall_exact_native;
        require(std::abs(delta_total - (delta_discovery + delta_ranking -
                                        delta_exact_control)) < 1e-12,
                "decomposition identity failed");
        const std::size_t common =
            intersection_size(exact_evaluated, pq_evaluated);
        const std::size_t union_size =
            exact_evaluated.size() + pq_evaluated.size() - common;
        const double jaccard =
            static_cast<double>(common) / static_cast<double>(union_size);

        rows << "{\"query_id\":" << query_id << ",\"query_order\":" << query_id
             << ",\"ef_search\":" << ef_search
             << ",\"recall_exact_native\":" << recall_exact_native
             << ",\"recall_pq_native\":" << recall_pq_native
             << ",\"recall_exact_oracle\":" << recall_exact_oracle
             << ",\"recall_pq_oracle\":" << recall_pq_oracle
             << ",\"coverage_exact\":" << coverage_exact
             << ",\"coverage_pq\":" << coverage_pq
             << ",\"delta_total\":" << delta_total
             << ",\"delta_discovery\":" << delta_discovery
             << ",\"delta_ranking\":" << delta_ranking
             << ",\"delta_exact_control\":" << delta_exact_control
             << ",\"exact_unique_distance_evaluations\":"
             << exact_evaluated.size()
             << ",\"pq_unique_distance_evaluations\":" << pq_evaluated.size()
             << ",\"evaluated_intersection_size\":" << common
             << ",\"evaluated_jaccard\":" << jaccard
             << ",\"ground_truth_ids\":";
        write_array(rows, ground_truth.data(), config.k);
        rows << ",\"exact_native_result_ids\":";
        write_array(rows, exact_native.ids.data() + offset, config.k);
        rows << ",\"pq_native_result_ids\":";
        write_array(rows, pq_native.ids.data() + offset, config.k);
        rows << ",\"exact_oracle_result_ids\":";
        write_array(rows, exact_oracle.ids.data(), config.k);
        rows << ",\"pq_oracle_result_ids\":";
        write_array(rows, pq_oracle.ids.data(), config.k);
        rows << ",\"exact_evaluated_ids\":";
        write_array(rows, exact_evaluated.data(), exact_evaluated.size());
        rows << ",\"pq_evaluated_ids\":";
        write_array(rows, pq_evaluated.data(), pq_evaluated.size());
        rows << "}\n";
      }
    }
    require(rows.good() && resolved.good(), "failed while writing raw outputs");
    std::cout << "run_directory=" << output_directory.string() << '\n'
              << "rows_written="
              << config.queries * config.ef_search_values.size() << '\n'
              << "graph_fingerprint=" << fingerprint << '\n'
              << "status=PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "status=FAIL\nerror=" << error.what() << '\n';
    return 1;
  }
}
