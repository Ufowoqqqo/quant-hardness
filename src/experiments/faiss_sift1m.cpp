#include "datasets/xvecs.h"
#include "graph/faiss_shared_hnsw.h"
#include "instrumentation/paired_decomposition_l0.h"
#include "instrumentation/faiss_level0_recorder.h"
#include "metrics/ground_truth.h"
#include "metrics/recall.h"

#include <faiss/IndexFlat.h>
#include <faiss/IndexPQ.h>
#include <faiss/index_io.h>

#include <omp.h>
#include <openssl/evp.h>
#include <sys/utsname.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

struct Config {
  std::string base_path, query_path, ground_truth_path, dataset_source_url;
  std::string dataset_revision, base_sha256, query_sha256;
  std::string ground_truth_sha256, cache_directory;
  faiss::idx_t base_count, query_count, k;
  int dimension, gt_width, hnsw_m, ef_construction, graph_seed;
  int construction_threads, search_threads, pq_nbits, pq_seed;
  int gt_validation_queries, gt_validation_seed, quality_queries;
  int distance_pairs, order_pairs, quality_seed, candidate_sample_queries;
  std::vector<int> exact_efs, pq_ms;
  std::vector<double> exact_targets;
  double primary_target_delta, primary_min_recall, primary_min_delta;
  double primary_max_delta, no_signal_discovery, no_signal_recovery;
  double emerging_discovery, emerging_recovery;
};

struct LoadedIndex {
  std::unique_ptr<faiss::Index> owner;
  faiss::IndexHNSW *graph = nullptr;
};

void require(bool condition, const std::string &message) {
  if (!condition) throw std::runtime_error(message);
}

std::vector<std::string> split(const std::string &value, char delimiter) {
  std::stringstream input(value);
  std::vector<std::string> output;
  std::string item;
  while (std::getline(input, item, delimiter)) {
    require(!item.empty(), "empty config-list item");
    output.push_back(item);
  }
  return output;
}

std::map<std::string, std::string> read_values(const std::string &path) {
  std::ifstream input(path);
  require(input.good(), "cannot open config: " + path);
  std::map<std::string, std::string> values;
  std::string line;
  while (std::getline(input, line)) {
    const auto comment = line.find('#');
    if (comment != std::string::npos) line.erase(comment);
    if (line.empty()) continue;
    const auto equals = line.find('=');
    require(equals != std::string::npos, "config line lacks '=': " + line);
    require(values.emplace(line.substr(0, equals), line.substr(equals + 1)).second,
            "duplicate config key");
  }
  return values;
}

Config read_config(const std::string &path) {
  const auto values = read_values(path);
  const auto get = [&](const std::string &key) -> const std::string & {
    const auto iterator = values.find(key);
    require(iterator != values.end() && !iterator->second.empty(),
            "missing config key: " + key);
    return iterator->second;
  };
  const auto integer = [&](const std::string &key) {
    return std::stoll(get(key));
  };
  Config c{get("base_path"), get("query_path"), get("ground_truth_path"),
           get("dataset_source_url"), get("dataset_revision"),
           get("base_sha256"), get("query_sha256"),
           get("ground_truth_sha256"), get("cache_directory"),
           integer("base_vectors"), integer("queries"), integer("k"),
           static_cast<int>(integer("dimension")),
           static_cast<int>(integer("ground_truth_width")),
           static_cast<int>(integer("hnsw_m")),
           static_cast<int>(integer("ef_construction")),
           static_cast<int>(integer("graph_seed")),
           static_cast<int>(integer("construction_threads")),
           static_cast<int>(integer("search_threads")),
           static_cast<int>(integer("pq_nbits")),
           static_cast<int>(integer("pq_seed")),
           static_cast<int>(integer("gt_validation_queries")),
           static_cast<int>(integer("gt_validation_seed")),
           static_cast<int>(integer("distance_sample_queries")),
           static_cast<int>(integer("distance_pairs_per_query")),
           static_cast<int>(integer("order_pairs_per_query")),
           static_cast<int>(integer("quality_sample_seed")),
           static_cast<int>(integer("candidate_set_sample_queries")),
           {}, {}, {}, std::stod(get("pq_primary_target_delta")),
           std::stod(get("pq_primary_min_recall")),
           std::stod(get("pq_primary_min_delta")),
           std::stod(get("pq_primary_max_delta")),
           std::stod(get("no_signal_max_abs_mean_delta_discovery")),
           std::stod(get("no_signal_min_rerank_recovery")),
           std::stod(get("emerging_min_mean_delta_discovery")),
           std::stod(get("emerging_max_rerank_recovery"))};
  for (const auto &item : split(get("exact_ef_values"), ','))
    c.exact_efs.push_back(std::stoi(item));
  for (const auto &item : split(get("exact_recall_targets"), ','))
    c.exact_targets.push_back(std::stod(item));
  for (const auto &item : split(get("pq_m_values"), ','))
    c.pq_ms.push_back(std::stoi(item));
  require(c.base_count == 1000000 && c.query_count == 10000 &&
              c.dimension == 128 && c.gt_width == 100 && c.k == 10,
          "SIFT1M dimensions differ from pre-registration");
  require(c.hnsw_m == 16 && c.ef_construction == 80 && c.pq_nbits == 8,
          "Phase 3A index parameters differ from pre-registration");
  require(c.exact_efs == std::vector<int>({16,32,48,64,96,128,192,256,384,512}) &&
              c.exact_targets == std::vector<double>({.90,.95,.98}) &&
              c.pq_ms == std::vector<int>({16,32,64}),
          "Phase 3A calibration grid differs from pre-registration");
  require(c.no_signal_discovery == .01 && c.no_signal_recovery == .90 &&
              c.emerging_discovery == .03 && c.emerging_recovery == .80,
          "Phase 3A gatekeeper thresholds differ from pre-registration");
  require(c.base_sha256 != "PENDING_DOWNLOAD_VERIFICATION" &&
              c.query_sha256 != "PENDING_DOWNLOAD_VERIFICATION" &&
              c.ground_truth_sha256 != "PENDING_DOWNLOAD_VERIFICATION",
          "dataset checksums must be fixed before execution");
  return c;
}

std::string sha256_file(const std::string &path) {
  std::ifstream input(path, std::ios::binary);
  require(input.good(), "cannot hash file: " + path);
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(
      EVP_MD_CTX_new(), EVP_MD_CTX_free);
  require(context && EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) == 1,
          "cannot initialize SHA-256");
  std::array<char, 1 << 20> buffer{};
  while (input) {
    input.read(buffer.data(), buffer.size());
    if (input.gcount() > 0)
      require(EVP_DigestUpdate(context.get(), buffer.data(), input.gcount()) == 1,
              "cannot update SHA-256");
  }
  std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
  unsigned int size = 0;
  require(EVP_DigestFinal_ex(context.get(), digest.data(), &size) == 1,
          "cannot finalize SHA-256");
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (unsigned int i = 0; i < size; ++i)
    output << std::setw(2) << static_cast<unsigned int>(digest[i]);
  return output.str();
}

template <typename Container> std::string bytes_hash(const Container &values) {
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(
      EVP_MD_CTX_new(), EVP_MD_CTX_free);
  require(context && EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) == 1 &&
              EVP_DigestUpdate(context.get(), values.data(),
                               values.size() * sizeof(*values.data())) == 1,
          "cannot hash memory");
  std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
  unsigned int size = 0;
  require(EVP_DigestFinal_ex(context.get(), digest.data(), &size) == 1,
          "cannot finalize memory hash");
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (unsigned int i = 0; i < size; ++i)
    output << std::setw(2) << static_cast<unsigned int>(digest[i]);
  return output.str();
}

quant_hardness::XvecDataset load_dataset(const Config &c) {
  require(sha256_file(c.base_path) == c.base_sha256,
          "base SHA-256 differs from config");
  require(sha256_file(c.query_path) == c.query_sha256,
          "query SHA-256 differs from config");
  require(sha256_file(c.ground_truth_path) == c.ground_truth_sha256,
          "ground-truth SHA-256 differs from config");
  auto dataset = quant_hardness::load_xvec_dataset(
      c.base_path, c.query_path, c.ground_truth_path);
  require(dataset.base_count == static_cast<std::size_t>(c.base_count) &&
              dataset.query_count == static_cast<std::size_t>(c.query_count) &&
              dataset.dimension == c.dimension &&
              dataset.ground_truth_width == static_cast<std::size_t>(c.gt_width),
          "loaded dataset shape differs from config");
  return dataset;
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

float squared_l2(const float *a, const float *b, int dimension) {
  float total = 0;
  for (int i = 0; i < dimension; ++i) {
    const float difference = a[i] - b[i];
    total += difference * difference;
  }
  return total;
}

double norm(const float *vector, int dimension) {
  double squared = 0.0;
  for (int i = 0; i < dimension; ++i)
    squared += static_cast<double>(vector[i]) * vector[i];
  return std::sqrt(squared);
}

template <typename T>
void write_array(std::ostream &output, std::span<const T> values) {
  output << '[';
  for (std::size_t i = 0; i < values.size(); ++i)
    output << (i ? "," : "") << values[i];
  output << ']';
}

double mean_recall(const quant_hardness::SearchResults &results,
                   std::span<const faiss::idx_t> truth,
                   faiss::idx_t query_count, faiss::idx_t k) {
  double total = 0;
  for (faiss::idx_t query = 0; query < query_count; ++query)
    total += quant_hardness::recall_at_k(
        std::span(results.ids).subspan(query * k, k),
        truth.subspan(query * k, k), k);
  return total / query_count;
}

faiss::SearchParametersHNSW parameters(int ef) {
  faiss::SearchParametersHNSW output;
  output.efSearch = ef;
  output.bounded_queue = true;
  output.check_relative_distance = true;
  return output;
}

LoadedIndex load_graph(const std::filesystem::path &path) {
  LoadedIndex result;
  result.owner.reset(faiss::read_index(path.c_str()));
  result.graph = dynamic_cast<faiss::IndexHNSW *>(result.owner.get());
  require(result.graph && result.graph->storage,
          "cached graph is not an IndexHNSW with storage");
  return result;
}

std::unique_ptr<faiss::IndexPQ> load_pq(const std::filesystem::path &path) {
  std::unique_ptr<faiss::Index> owner(faiss::read_index(path.c_str()));
  auto *pq = dynamic_cast<faiss::IndexPQ *>(owner.get());
  require(pq != nullptr, "cached PQ index has wrong type");
  owner.release();
  return std::unique_ptr<faiss::IndexPQ>(pq);
}

void validate_ground_truth(const Config &c,
                           const quant_hardness::XvecDataset &dataset,
                           const std::filesystem::path &output_path) {
  std::mt19937 generator(c.gt_validation_seed);
  std::vector<faiss::idx_t> ids(c.query_count);
  std::iota(ids.begin(), ids.end(), 0);
  std::shuffle(ids.begin(), ids.end(), generator);
  ids.resize(c.gt_validation_queries);
  std::sort(ids.begin(), ids.end());
  std::vector<float> queries(ids.size() * c.dimension);
  for (std::size_t i = 0; i < ids.size(); ++i)
    std::copy_n(dataset.queries.data() + ids[i] * c.dimension, c.dimension,
                queries.data() + i * c.dimension);
  const auto exact = quant_hardness::exhaustive_l2_top_k(
      dataset.base.data(), c.base_count, queries.data(), ids.size(), c.dimension,
      c.k);
  std::ofstream output(output_path);
  require(output.good(), "cannot create ground-truth validation output");
  output << "query_id,provided_exact_set_recall\n";
  for (std::size_t i = 0; i < ids.size(); ++i) {
    const std::size_t source_offset = ids[i] * dataset.ground_truth_width;
    const auto provided = std::span(dataset.ground_truth_ids)
                              .subspan(source_offset, c.k);
    const auto computed = std::span(exact.ids).subspan(i * c.k, c.k);
    const double recall = quant_hardness::recall_at_k(computed, provided, c.k);
    require(recall == 1.0, "provided GT disagrees with exhaustive top-10");
    output << ids[i] << ',' << recall << '\n';
  }
}

std::filesystem::path pq_path(const Config &c, int m) {
  return std::filesystem::path(c.cache_directory) /
         ("pq_m" + std::to_string(m) + "_8bit.index");
}

void prepare(const Config &c, const std::string &config_path,
             const std::filesystem::path &run_root) {
  require(!std::filesystem::exists(run_root),
          "refusing to overwrite run directory");
  require(!std::filesystem::exists(c.cache_directory),
          "refusing to overwrite index cache");
  std::filesystem::create_directories(run_root);
  std::filesystem::create_directories(c.cache_directory);
  std::ifstream config_input(config_path);
  std::ofstream resolved(run_root / "resolved_config.conf");
  require(config_input.good() && resolved.good(),
          "cannot copy resolved configuration");
  resolved << config_input.rdbuf();
  std::ofstream timing(run_root / "preparation_timing.csv");
  timing << "stage,seconds\n";
  const auto timed = [&](const std::string &stage, const auto &operation) {
    std::cout << "stage=" << stage << " status=START\n" << std::flush;
    const auto start = std::chrono::steady_clock::now();
    operation();
    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    timing << stage << ',' << std::setprecision(17) << seconds << '\n';
    timing.flush();
    std::cout << "stage=" << stage << " seconds=" << seconds
              << " status=PASS\n" << std::flush;
  };
  quant_hardness::XvecDataset dataset;
  timed("load_and_checksum_dataset", [&] { dataset = load_dataset(c); });
  timed("validate_provided_ground_truth", [&] {
    validate_ground_truth(c, dataset,
                          run_root / "ground_truth_validation.csv");
  });

  omp_set_num_threads(c.construction_threads);
  faiss::IndexFlatL2 exact_storage(c.dimension);
  faiss::IndexHNSW graph(&exact_storage, c.hnsw_m);
  graph.hnsw.efConstruction = c.ef_construction;
  graph.hnsw.rng = faiss::RandomGenerator(c.graph_seed);
  timed("build_fp32_hnsw", [&] { graph.add(c.base_count, dataset.base.data()); });
  const std::string fingerprint =
      quant_hardness::graph_fingerprint(graph.hnsw);
  const auto graph_path =
      std::filesystem::path(c.cache_directory) / "hnsw_fp32.index";
  faiss::write_index(&graph, graph_path.c_str());

  std::ofstream pq_state(run_root / "pq_index_state.csv");
  pq_state << "pq_m,pq_nbits,code_size,codebook_sha256,codes_sha256,id_alignment_checked\n";
  for (const int m : c.pq_ms) {
    faiss::IndexPQ pq(c.dimension, m, c.pq_nbits, faiss::METRIC_L2);
    pq.pq.cp.seed = c.pq_seed;
    timed("train_and_encode_pq_m" + std::to_string(m), [&] {
      pq.train(c.base_count, dataset.base.data());
      pq.add(c.base_count, dataset.base.data());
    });
    require(pq.ntotal == c.base_count, "PQ ntotal differs from database");
    const std::array<faiss::idx_t, 3> check_ids{0, c.base_count / 2,
                                               c.base_count - 1};
    std::vector<std::uint8_t> encoded(pq.code_size);
    for (const faiss::idx_t id : check_ids) {
      pq.sa_encode(1, dataset.base.data() + id * c.dimension, encoded.data());
      require(std::equal(encoded.begin(), encoded.end(),
                         pq.codes.begin() + id * pq.code_size),
              "PQ code/ID alignment validation failed");
    }
    faiss::write_index(&pq, pq_path(c, m).c_str());
    pq_state << m << ',' << c.pq_nbits << ',' << pq.code_size << ','
             << bytes_hash(pq.pq.centroids) << ',' << bytes_hash(pq.codes)
             << ",true\n";
  }

  std::ofstream state(run_root / "prepared_state.conf");
  state << "base_sha256=" << c.base_sha256 << '\n'
        << "query_sha256=" << c.query_sha256 << '\n'
        << "ground_truth_sha256=" << c.ground_truth_sha256 << '\n'
        << "graph_fingerprint=" << fingerprint << '\n'
        << "graph_index_path=" << graph_path.string() << '\n';
  std::ofstream manifest(run_root / "manifest.json");
  manifest << "{\n  \"schema_version\":1,\n  \"run_id\":\"phase3a_sift1m_v1\","
           << "\n  \"git_commit\":\"" << QH_GIT_COMMIT
           << "\",\n  \"dirty_worktree\":" << QH_GIT_DIRTY
           << ",\n  \"faiss_commit\":\"" << QH_FAISS_COMMIT
           << "\",\n  \"dataset\":{\"base_path\":\"" << c.base_path
           << "\",\"query_path\":\"" << c.query_path
           << "\",\"ground_truth_path\":\"" << c.ground_truth_path
           << "\",\"source_url\":\"" << c.dataset_source_url
           << "\",\"revision\":\"" << c.dataset_revision
           << "\",\"base_sha256\":\"" << c.base_sha256
           << "\",\"query_sha256\":\"" << c.query_sha256
           << "\",\"ground_truth_sha256\":\"" << c.ground_truth_sha256
           << "\",\"base_vectors\":" << c.base_count
           << ",\"queries\":" << c.query_count
           << ",\"dimension\":" << c.dimension
           << ",\"provided_ground_truth_width\":" << c.gt_width
           << "},\n  \"graph\":{\"M\":" << c.hnsw_m
           << ",\"efConstruction\":" << c.ef_construction
           << ",\"seed\":" << c.graph_seed
           << ",\"fingerprint\":\"" << fingerprint
           << "\"},\n  \"quantizers\":{\"pq_m_values\":[16,32,64],"
              "\"nbits\":8,\"seed\":" << c.pq_seed
           << ",\"training_input_vectors\":" << c.base_count
           << "},\n  \"search\":{\"k\":" << c.k
           << ",\"threads\":" << c.search_threads
           << ",\"bounded_queue\":true,\"check_relative_distance\":true},"
           << "\n  \"candidate_semantics\":\"corrected L0-only\","
           << "\n  \"machine\":{\"hostname\":\"" << hostname()
           << "\",\"kernel\":\"" << kernel()
           << "\",\"hardware_concurrency\":"
           << std::thread::hardware_concurrency() << ",\"compiler\":\""
           << __VERSION__ << "\"}\n}\n";
  std::cout << "graph_fingerprint=" << fingerprint << "\nstatus=PASS\n";
}

std::vector<std::vector<faiss::idx_t>> sample_ids(
    const Config &c, bool pairs, std::mt19937 &generator) {
  const int per_query = pairs ? 2 * c.order_pairs : c.distance_pairs;
  std::uniform_int_distribution<faiss::idx_t> distribution(0,
                                                            c.base_count - 1);
  std::vector<std::vector<faiss::idx_t>> output(
      c.quality_queries, std::vector<faiss::idx_t>(per_query));
  for (auto &row : output)
    for (auto &id : row) id = distribution(generator);
  return output;
}

void write_quality(const Config &c,
                   const quant_hardness::XvecDataset &dataset,
                   faiss::IndexPQ &pq, int m,
                   const std::filesystem::path &run_root) {
  std::mt19937 generator(c.quality_seed);
  const auto distances_ids = sample_ids(c, false, generator);
  const auto order_ids = sample_ids(c, true, generator);
  std::ofstream distances(run_root / ("pq_m" + std::to_string(m) +
                                      "_distance_samples.jsonl"));
  std::ofstream orders(run_root / ("pq_m" + std::to_string(m) +
                                   "_order_samples.jsonl"));
  require(distances.good() && orders.good(), "cannot create quality rows");
  distances << std::setprecision(17);
  orders << std::setprecision(17);
  std::unique_ptr<faiss::DistanceComputer> computer(pq.get_distance_computer());
  for (int query_id = 0; query_id < c.quality_queries; ++query_id) {
    const float *query = dataset.queries.data() + query_id * c.dimension;
    computer->set_query(query);
    for (int sample = 0; sample < c.distance_pairs; ++sample) {
      const faiss::idx_t id = distances_ids[query_id][sample];
      const float exact = squared_l2(
          query, dataset.base.data() + id * c.dimension, c.dimension);
      const float approximate = (*computer)(id);
      distances << "{\"pq_m\":" << m << ",\"query_id\":" << query_id
                << ",\"sample_id\":" << sample << ",\"database_id\":"
                << id << ",\"exact_distance\":" << exact
                << ",\"pq_distance\":" << approximate
                << ",\"absolute_error\":" << std::abs(approximate - exact)
                << ",\"absolute_relative_error\":"
                << (exact > 1e-12F ? std::abs(approximate - exact) / exact : 0)
                << "}\n";
    }
    for (int sample = 0; sample < c.order_pairs; ++sample) {
      faiss::idx_t first = order_ids[query_id][2 * sample];
      faiss::idx_t second = order_ids[query_id][2 * sample + 1];
      if (second == first) second = (second + 1) % c.base_count;
      const float exact_first = squared_l2(
          query, dataset.base.data() + first * c.dimension, c.dimension);
      const float exact_second = squared_l2(
          query, dataset.base.data() + second * c.dimension, c.dimension);
      const float pq_first = (*computer)(first), pq_second = (*computer)(second);
      const bool inversion =
          (exact_first < exact_second && pq_first > pq_second) ||
          (exact_first > exact_second && pq_first < pq_second);
      orders << "{\"pq_m\":" << m << ",\"query_id\":" << query_id
             << ",\"sample_id\":" << sample << ",\"first_database_id\":"
             << first << ",\"second_database_id\":" << second
             << ",\"exact_first\":" << exact_first
             << ",\"exact_second\":" << exact_second
             << ",\"pq_first\":" << pq_first << ",\"pq_second\":"
             << pq_second << ",\"strict_inversion\":"
             << (inversion ? "true" : "false") << ",\"pq_tie\":"
             << (pq_first == pq_second ? "true" : "false") << "}\n";
    }
  }
}

void calibrate(const Config &c, const std::filesystem::path &run_root) {
  require(std::filesystem::exists(run_root / "prepared_state.conf"),
          "prepare stage has not completed");
  auto dataset = load_dataset(c);
  const auto truth = quant_hardness::first_ground_truth_neighbors(dataset, c.k);
  auto loaded = load_graph(std::filesystem::path(c.cache_directory) /
                           "hnsw_fp32.index");
  const auto state = read_values((run_root / "prepared_state.conf").string());
  const std::string fingerprint =
      quant_hardness::graph_fingerprint(loaded.graph->hnsw);
  require(state.at("graph_fingerprint") == fingerprint,
          "loaded graph fingerprint differs from preparation");
  omp_set_num_threads(c.search_threads);

  std::vector<std::pair<int, double>> exact_rows;
  std::ofstream exact_output(run_root / "exact_calibration.csv");
  exact_output << "ef_search,mean_exact_recall,graph_fingerprint\n";
  for (const int ef : c.exact_efs) {
    const auto result = quant_hardness::search_with_storage(
        *loaded.graph, *loaded.graph->storage, dataset.queries.data(),
        c.query_count, c.k, parameters(ef));
    const double recall = mean_recall(result, truth, c.query_count, c.k);
    exact_rows.emplace_back(ef, recall);
    exact_output << ef << ',' << std::setprecision(17) << recall << ','
                 << fingerprint << '\n';
    std::cout << "exact ef=" << ef << " recall=" << recall << '\n';
  }
  std::set<int> used;
  std::vector<int> selected;
  for (const double target : c.exact_targets) {
    auto candidates = exact_rows;
    std::sort(candidates.begin(), candidates.end(), [&](const auto &a,
                                                        const auto &b) {
      const auto ka = std::pair(std::abs(a.second - target), a.first);
      const auto kb = std::pair(std::abs(b.second - target), b.first);
      return ka < kb;
    });
    const auto choice = std::find_if(candidates.begin(), candidates.end(),
                                     [&](const auto &row) {
                                       return !used.contains(row.first);
                                     });
    require(choice != candidates.end(), "cannot choose unique exact point");
    selected.push_back(choice->first);
    used.insert(choice->first);
  }
  const int central_ef = selected[1];
  const double central_exact = std::find_if(
      exact_rows.begin(), exact_rows.end(), [&](const auto &row) {
        return row.first == central_ef;
      })->second;

  struct PqSummary { int m; double recall; double delta; };
  std::vector<PqSummary> pq_rows;
  std::ofstream pq_output(run_root / "pq_calibration.csv");
  pq_output << "pq_m,pq_nbits,ef_search,mean_exact_recall,mean_pq_recall,mean_delta\n";
  for (const int m : c.pq_ms) {
    auto pq = load_pq(pq_path(c, m));
    const auto result = quant_hardness::search_with_storage(
        *loaded.graph, *pq, dataset.queries.data(), c.query_count, c.k,
        parameters(central_ef));
    const double recall = mean_recall(result, truth, c.query_count, c.k);
    pq_rows.push_back({m, recall, central_exact - recall});
    pq_output << m << ',' << c.pq_nbits << ',' << central_ef << ','
              << std::setprecision(17) << central_exact << ',' << recall << ','
              << central_exact - recall << '\n';
    std::ofstream queries(run_root / ("pq_calibration_queries_m" +
                                      std::to_string(m) + ".jsonl"));
    for (faiss::idx_t query = 0; query < c.query_count; ++query) {
      const auto ids = std::span(result.ids).subspan(query * c.k, c.k);
      const double row_recall = quant_hardness::recall_at_k(
          ids, std::span(truth).subspan(query * c.k, c.k), c.k);
      queries << "{\"query_id\":" << query << ",\"pq_m\":" << m
              << ",\"ef_search\":" << central_ef
              << ",\"recall_pq_native\":" << row_recall
              << ",\"pq_result_ids\":";
      write_array(queries, ids);
      queries << "}\n";
    }
    write_quality(c, dataset, *pq, m, run_root);
  }
  std::vector<PqSummary> eligible;
  std::copy_if(pq_rows.begin(), pq_rows.end(), std::back_inserter(eligible),
               [&](const auto &row) {
                 return row.recall >= c.primary_min_recall &&
                        row.delta >= c.primary_min_delta &&
                        row.delta <= c.primary_max_delta;
               });
  require(!eligible.empty(),
          "no PQ configuration satisfies pre-registered operating regime");
  std::sort(eligible.begin(), eligible.end(), [&](const auto &a, const auto &b) {
    return std::pair(std::abs(a.delta - c.primary_target_delta), a.m) <
           std::pair(std::abs(b.delta - c.primary_target_delta), b.m);
  });
  std::ofstream selection(run_root / "selection.conf");
  selection << "low_ef=" << selected[0] << '\n'
            << "central_ef=" << selected[1] << '\n'
            << "very_high_ef=" << selected[2] << '\n'
            << "primary_pq_m=" << eligible.front().m << '\n'
            << "primary_pq_nbits=" << c.pq_nbits << '\n'
            << "graph_fingerprint=" << fingerprint << '\n';
  std::cout << "selected_efs=" << selected[0] << ',' << selected[1] << ','
            << selected[2] << " primary_pq_m=" << eligible.front().m
            << "\nstatus=PASS\n";
}

void write_decomposition_rows(
    const Config &c, const quant_hardness::XvecDataset &dataset,
    std::span<const faiss::idx_t> truth,
    const std::vector<quant_hardness::QueryDecompositionL0> &rows, int ef,
    int pq_m, const std::string &fingerprint,
    const std::filesystem::path &path) {
  std::ofstream output(path);
  require(output.good(), "cannot create decomposition rows");
  output << std::setprecision(17);
  for (faiss::idx_t query_id = 0; query_id < c.query_count; ++query_id) {
    const auto &row = rows[query_id];
    if (row.delta_exact_control != 0.0) {
      std::ostringstream message;
      message << "SIFT1M exact L0 control is nonzero at query " << query_id
              << ": exact_native=" << row.recall_exact_native
              << " exact_oracle=" << row.recall_exact_l0_oracle
              << " truth=";
      for (const auto id : row.ground_truth_ids) message << id << ',';
      message << " exact_native_ids=";
      for (const auto id : row.exact_native_ids) message << id << ',';
      message << " exact_oracle_ids=";
      for (const auto id : row.exact_l0_oracle_ids) message << id << ',';
      throw std::runtime_error(message.str());
    }
    const float *query = dataset.queries.data() + query_id * c.dimension;
    const faiss::idx_t first_id = truth[query_id * c.k];
    const faiss::idx_t tenth_id = truth[query_id * c.k + c.k - 1];
    const double first = std::sqrt(squared_l2(
        query, dataset.base.data() + first_id * c.dimension, c.dimension));
    const double tenth = std::sqrt(squared_l2(
        query, dataset.base.data() + tenth_id * c.dimension, c.dimension));
    const double denominator = row.delta_total;
    output << "{\"query_id\":" << query_id << ",\"query_order\":"
           << query_id << ",\"ef_search\":" << ef << ",\"pq_m\":"
           << pq_m << ",\"pq_nbits\":" << c.pq_nbits
           << ",\"graph_fingerprint\":\"" << fingerprint
           << "\",\"query_l2_norm\":" << norm(query, c.dimension)
           << ",\"exact_1nn_distance\":" << first
           << ",\"exact_10nn_distance\":" << tenth
           << ",\"nn_margin\":" << tenth - first
           << ",\"normalized_nn_margin\":"
           << (tenth > 1e-12 ? (tenth - first) / tenth : 0)
           << ",\"local_distance_concentration\":"
           << (first > 1e-12 ? (tenth - first) / first : 0)
           << ",\"recall_exact_native\":" << row.recall_exact_native
           << ",\"recall_pq_native\":" << row.recall_pq_native
           << ",\"recall_exact_L0_oracle\":"
           << row.recall_exact_l0_oracle
           << ",\"recall_pq_L0_oracle\":" << row.recall_pq_l0_oracle
           << ",\"delta_total\":" << row.delta_total
           << ",\"delta_discovery\":" << row.delta_discovery
           << ",\"delta_ranking\":" << row.delta_ranking
           << ",\"delta_exact_control\":" << row.delta_exact_control
           << ",\"exact_oracle_coverage_tie_mismatch\":"
           << (row.exact_oracle_coverage_tie_mismatch ? "true" : "false")
           << ",\"pq_oracle_coverage_tie_mismatch\":"
           << (row.pq_oracle_coverage_tie_mismatch ? "true" : "false")
           << ",\"exact_L0_candidate_coverage\":" << row.coverage_exact_l0
           << ",\"pq_L0_candidate_coverage\":" << row.coverage_pq_l0
           << ",\"exact_L0_distance_evaluations\":"
           << row.exact_l0_evaluated_ids.size()
           << ",\"pq_L0_distance_evaluations\":"
           << row.pq_l0_evaluated_ids.size()
           << ",\"exact_upper_only_distance_evaluations\":"
           << row.exact_upper_only_evaluated_ids.size()
           << ",\"pq_upper_only_distance_evaluations\":"
           << row.pq_upper_only_evaluated_ids.size()
           << ",\"evaluated_L0_intersection_size\":"
           << row.evaluated_l0_intersection_size
           << ",\"evaluated_L0_jaccard\":" << row.evaluated_l0_jaccard
           << ",\"rerank_recovery\":";
    if (std::abs(denominator) > 1e-12)
      output << row.delta_ranking / denominator;
    else
      output << "null";
    output << ",\"ground_truth_ids\":"; write_array(output, std::span(row.ground_truth_ids));
    output << ",\"exact_native_result_ids\":"; write_array(output, std::span(row.exact_native_ids));
    output << ",\"pq_native_result_ids\":"; write_array(output, std::span(row.pq_native_ids));
    output << ",\"exact_L0_oracle_result_ids\":"; write_array(output, std::span(row.exact_l0_oracle_ids));
    output << ",\"pq_L0_oracle_result_ids\":"; write_array(output, std::span(row.pq_l0_oracle_ids));
    if (query_id < c.candidate_sample_queries) {
      output << ",\"exact_L0_evaluated_ids\":"; write_array(output, std::span(row.exact_l0_evaluated_ids));
      output << ",\"pq_L0_evaluated_ids\":"; write_array(output, std::span(row.pq_l0_evaluated_ids));
      output << ",\"exact_upper_only_evaluated_ids\":"; write_array(output, std::span(row.exact_upper_only_evaluated_ids));
      output << ",\"pq_upper_only_evaluated_ids\":"; write_array(output, std::span(row.pq_upper_only_evaluated_ids));
    } else {
      output << ",\"exact_L0_evaluated_ids\":null,\"pq_L0_evaluated_ids\":null,"
                "\"exact_upper_only_evaluated_ids\":null,"
                "\"pq_upper_only_evaluated_ids\":null";
    }
    output << "}\n";
  }
}

void decompose(const Config &c, const std::filesystem::path &run_root) {
  const auto selected = read_values((run_root / "selection.conf").string());
  const std::vector<int> efs{std::stoi(selected.at("low_ef")),
                             std::stoi(selected.at("central_ef")),
                             std::stoi(selected.at("very_high_ef"))};
  const int pq_m = std::stoi(selected.at("primary_pq_m"));
  auto dataset = load_dataset(c);
  const auto truth = quant_hardness::first_ground_truth_neighbors(dataset, c.k);
  auto loaded = load_graph(std::filesystem::path(c.cache_directory) /
                           "hnsw_fp32.index");
  auto pq = load_pq(pq_path(c, pq_m));
  const std::string fingerprint =
      quant_hardness::graph_fingerprint(loaded.graph->hnsw);
  require(fingerprint == selected.at("graph_fingerprint"),
          "decomposition graph fingerprint differs from calibration");
  omp_set_num_threads(c.search_threads);
  for (const int ef : efs) {
    const auto rows = quant_hardness::measure_paired_decomposition_l0(
        *loaded.graph, *loaded.graph->storage, *pq, dataset.base.data(),
        c.base_count, dataset.queries.data(), c.query_count, c.dimension, c.k,
        truth, parameters(ef), true, true, true);
    write_decomposition_rows(
        c, dataset, truth, rows, ef, pq_m, fingerprint,
        run_root / ("decomposition_ef" + std::to_string(ef) + ".jsonl"));
    require(quant_hardness::graph_fingerprint(loaded.graph->hnsw) == fingerprint,
            "decomposition changed graph fingerprint");
    std::cout << "decomposition ef=" << ef << " status=PASS\n";
  }
  std::cout << "status=PASS\n";
}

// Export observations only: all candidate-level analysis is independent of
// FAISS in scripts/analyze_phase3b_fixed_candidate_ranking.py.
void export_ranking(const std::string &config_path,
                    const std::filesystem::path &run_root, int ef) {
  const auto settings = read_values(config_path);
  const Config c = read_config(settings.at("phase3a_config"));
  require(ef == std::stoi(settings.at("primary_ef")) ||
              ef == std::stoi(settings.at("replication_ef")), "unregistered ef");
  require(std::stoi(settings.at("pq_m")) == 64 &&
              std::stoi(settings.at("pq_nbits")) == 8 &&
              std::stoi(settings.at("k")) == c.k, "wrong ranking configuration");
  require(std::endian::native == std::endian::little,
          "candidate binary schema requires little endian");
  const auto destination = run_root / ("ef" + std::to_string(ef));
  require(!std::filesystem::exists(destination), "refusing to overwrite export");
  const int chunk_size = std::stoi(settings.at("queries_per_chunk"));
  require(chunk_size > 0, "invalid chunk size");
  auto dataset = load_dataset(c);
  auto loaded = load_graph(std::filesystem::path(c.cache_directory) /
                           "hnsw_fp32.index");
  auto pq = load_pq(pq_path(c, 64));
  const auto check_state = [&]() {
    require(quant_hardness::graph_fingerprint(loaded.graph->hnsw) ==
                settings.at("graph_fingerprint"), "graph identity changed");
    require(bytes_hash(pq->pq.centroids) == settings.at("codebook_sha256") &&
                bytes_hash(pq->codes) == settings.at("codes_sha256"),
            "PQ model/codes changed");
    require(pq->ntotal == c.base_count && pq->pq.M == 64 && pq->pq.nbits == 8,
            "PQ shape differs from frozen index");
  };
  check_state();
  omp_set_num_threads(c.search_threads);
  const auto recorded = quant_hardness::search_with_level0_recording(
      *loaded.graph, *pq, dataset.queries.data(), c.query_count, c.k,
      parameters(ef), true);
  check_state();
  std::filesystem::create_directories(destination);
  std::filesystem::copy_file(config_path, destination / "resolved_config.conf");
  std::filesystem::copy_file(settings.at("phase3a_config"),
                             destination / "phase3a_config.conf");
  std::ofstream metadata(destination / "queries.jsonl");
  metadata << std::setprecision(17);
  std::ofstream candidates;
  std::uint64_t offset = 0, total_candidates = 0;
  std::unique_ptr<faiss::DistanceComputer> computer(pq->get_distance_computer());
  for (faiss::idx_t q = 0; q < c.query_count; ++q) {
    const int chunk = static_cast<int>(q / chunk_size);
    const std::string filename = "candidate_scores_" + std::to_string(chunk) + ".bin";
    if (q % chunk_size == 0) {
      if (candidates.is_open()) candidates.close();
      candidates.open(destination / filename, std::ios::binary);
      require(candidates.good(), "cannot open candidate chunk");
      offset = 0;
    }
    const float *query = dataset.queries.data() + q * c.dimension;
    computer->set_query(query);
    const auto &ids = recorded.level0_evaluation_order_ids[q];
    require(std::set<faiss::idx_t>(ids.begin(), ids.end()).size() == ids.size(),
            "candidate IDs not unique");
    std::vector<float> scores;
    for (const faiss::idx_t id : ids) {
      const std::int32_t stored_id = static_cast<std::int32_t>(id);
      const float exact = squared_l2(query, dataset.base.data() + id * c.dimension,
                                      c.dimension);
      const float approximate = (*computer)(id);
      scores.push_back(approximate);
      candidates.write(reinterpret_cast<const char *>(&stored_id), 4);
      candidates.write(reinterpret_cast<const char *>(&exact), 4);
      candidates.write(reinterpret_cast<const char *>(&approximate), 4);
    }
    // Native HNSW uses scalar and batch-4 ADC. Confirm post-hoc scalar scores
    // equal batch scores for every candidate (not just returned neighbors).
    for (std::size_t i = 0; i + 3 < ids.size(); i += 4) {
      float a, b, d, e;
      computer->distances_batch_4(ids[i], ids[i+1], ids[i+2], ids[i+3], a,b,d,e);
      require(a == scores[i] && b == scores[i+1] && d == scores[i+2] &&
                  e == scores[i+3], "scalar/batch ADC differ");
    }
    for (faiss::idx_t j = 0; j < c.k; ++j)
      require((*computer)(recorded.native.ids[q*c.k+j]) ==
                  recorded.native.distances[q*c.k+j], "native ADC mismatch");
    metadata << "{\"query_id\":" << q << ",\"ef_search\":" << ef
             << ",\"candidate_file\":\"" << filename
             << "\",\"byte_offset\":" << offset << ",\"candidate_count\":"
             << ids.size() << ",\"native_ids\":";
    write_array(metadata, std::span(recorded.native.ids).subspan(q*c.k,c.k));
    metadata << ",\"native_pq_distances\":";
    write_array(metadata, std::span(recorded.native.distances).subspan(q*c.k,c.k));
    metadata << ",\"ground_truth_ids\":";
    write_array(metadata, std::span<const faiss::idx_t>(dataset.ground_truth_ids).subspan(
                              q*dataset.ground_truth_width,c.k));
    metadata << "}\n";
    offset += ids.size() * 12;
    total_candidates += ids.size();
    require(candidates.good() && metadata.good(), "failed to write raw scores");
  }
  candidates.close();
  metadata.close();
  check_state();
  std::ofstream manifest(destination / "manifest.json");
  manifest << std::boolalpha << "{\"git_commit\":\"" << QH_GIT_COMMIT
           << "\",\"dirty_worktree\":" << QH_GIT_DIRTY
           << ",\"faiss_commit\":\"" << QH_FAISS_COMMIT
           << "\",\"hostname\":\"" << hostname() << "\",\"kernel\":\""
           << kernel() << "\",\"compiler\":\"" << __VERSION__
           << "\",\"search_threads\":" << c.search_threads
           << ",\"graph_fingerprint\":\"" << settings.at("graph_fingerprint")
           << "\",\"codebook_sha256\":\"" << settings.at("codebook_sha256")
           << "\",\"codes_sha256\":\"" << settings.at("codes_sha256")
           << "\",\"binary_sha256\":\"" << sha256_file("/proc/self/exe")
           << "\",\"export_source_sha256\":\""
           << sha256_file("src/experiments/faiss_sift1m.cpp")
           << "\",\"config_sha256\":\"" << sha256_file(config_path)
           << "\",\"total_candidates\":" << total_candidates
           << ",\"schema\":\"packed little-endian int32 ID, float32 exact squared L2, float32 PQ ADC; unique first L0 evaluation order\""
           << ",\"native_replay_identity\":true,\"scalar_batch_adc_identity\":true"
           << ",\"frozen_state_before_after\":true}\n";
  std::cout << "ranking_export ef=" << ef << " candidates=" << total_candidates
            << " status=PASS\n";
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc == 5 && std::string(argv[1]) == "ranking-export") {
      export_ranking(argv[2], argv[3], std::stoi(argv[4]));
      return 0;
    }
    require(argc == 4,
            "usage: faiss_sift1m prepare|calibrate|decompose CONFIG RUN_DIR");
    const Config config = read_config(argv[2]);
    const std::filesystem::path run_root(argv[3]);
    const std::string mode(argv[1]);
    if (mode == "prepare") prepare(config, argv[2], run_root);
    else if (mode == "calibrate") calibrate(config, run_root);
    else if (mode == "decompose") decompose(config, run_root);
    else throw std::runtime_error("unknown mode: " + mode);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "status=FAIL\nerror=" << error.what() << '\n';
    return 1;
  }
}
