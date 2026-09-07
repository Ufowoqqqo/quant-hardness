#include "graph/faiss_shared_hnsw.h"
#include "instrumentation/paired_decomposition_l0.h"
#include "metrics/ground_truth.h"
#include "metrics/recall.h"

#include <faiss/IndexFlat.h>
#include <faiss/IndexPQ.h>

#include <omp.h>
#include <openssl/evp.h>
#include <sys/utsname.h>
#include <unistd.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

struct Seeds {
  std::uint32_t database_z, query_z, centers, database_labels, query_labels;
  std::uint32_t graph, pq, samples;
};

struct Config {
  std::vector<Seeds> seeds;
  faiss::idx_t base_vectors, queries, k;
  int dimension, clusters, hnsw_m, ef_construction, fixed_ef_search;
  std::vector<double> rhos;
  std::vector<int> ef_calibration;
  int pq_m, pq_nbits, sample_queries, distance_pairs, order_pairs;
  double exact_trigger, matched_target, no_evidence_discovery;
  double no_evidence_recovery, emerging_discovery, emerging_recovery;
  int emerging_replicates, candidate_sample_queries;
};

struct Samples {
  std::vector<std::vector<faiss::idx_t>> distance_ids;
  std::vector<std::vector<std::pair<faiss::idx_t, faiss::idx_t>>> order_ids;
};

void require(bool condition, const std::string &message) {
  if (!condition) throw std::runtime_error(message);
}

std::vector<std::string> split(const std::string &value, char delimiter) {
  std::stringstream stream(value);
  std::vector<std::string> output;
  std::string item;
  while (std::getline(stream, item, delimiter)) {
    require(!item.empty(), "empty config list item");
    output.push_back(item);
  }
  return output;
}

Config read_config(const std::string &path) {
  std::ifstream input(path);
  require(input.good(), "cannot open config: " + path);
  std::map<std::string, std::string> values;
  std::string line;
  while (std::getline(input, line)) {
    const auto comment = line.find('#');
    if (comment != std::string::npos) line.erase(comment);
    const auto equals = line.find('=');
    if (equals != std::string::npos) {
      require(values.emplace(line.substr(0, equals), line.substr(equals + 1)).second,
              "duplicate config key");
    }
  }
  const auto get = [&](const std::string &key) -> const std::string & {
    const auto iterator = values.find(key);
    require(iterator != values.end() && !iterator->second.empty(),
            "missing config key: " + key);
    return iterator->second;
  };
  const auto integer = [&](const std::string &key) { return std::stoll(get(key)); };
  Config config{{},
                integer("base_vectors"), integer("queries"), integer("k"),
                static_cast<int>(integer("dimension")),
                static_cast<int>(integer("clusters")),
                static_cast<int>(integer("hnsw_m")),
                static_cast<int>(integer("ef_construction")),
                static_cast<int>(integer("fixed_ef_search")), {}, {},
                static_cast<int>(integer("pq_m")),
                static_cast<int>(integer("pq_nbits")),
                static_cast<int>(integer("distance_sample_queries")),
                static_cast<int>(integer("distance_pairs_per_query")),
                static_cast<int>(integer("order_pairs_per_query")),
                std::stod(get("exact_recall_calibration_trigger")),
                std::stod(get("matched_exact_recall_target")),
                std::stod(get("no_evidence_max_abs_mean_delta_discovery")),
                std::stod(get("no_evidence_min_rerank_recovery")),
                std::stod(get("emerging_min_mean_delta_discovery")),
                std::stod(get("emerging_max_rerank_recovery")),
                static_cast<int>(integer("emerging_required_replicates")),
                static_cast<int>(integer("candidate_set_sample_queries"))};
  for (const auto &tuple : split(get("seed_tuples"), ',')) {
    const auto fields = split(tuple, ':');
    require(fields.size() == 8, "seed tuple must have eight fields");
    config.seeds.push_back({
        static_cast<std::uint32_t>(std::stoul(fields[0])),
        static_cast<std::uint32_t>(std::stoul(fields[1])),
        static_cast<std::uint32_t>(std::stoul(fields[2])),
        static_cast<std::uint32_t>(std::stoul(fields[3])),
        static_cast<std::uint32_t>(std::stoul(fields[4])),
        static_cast<std::uint32_t>(std::stoul(fields[5])),
        static_cast<std::uint32_t>(std::stoul(fields[6])),
        static_cast<std::uint32_t>(std::stoul(fields[7]))});
  }
  for (const auto &item : split(get("rho_values"), ','))
    config.rhos.push_back(std::stod(item));
  for (const auto &item : split(get("ef_calibration_values"), ','))
    config.ef_calibration.push_back(std::stoi(item));
  require(values.size() == 24, "config contains unknown keys");
  require(config.seeds.size() == 4 && config.base_vectors == 20000 &&
              config.queries == 500 && config.dimension == 64 &&
              config.k == 10 && config.clusters == 16,
          "Phase 2B dataset parameters differ from pre-registration");
  require(config.rhos ==
              std::vector<double>({0.0, 0.10, 0.25, 0.50, 0.75, 0.90}),
          "Phase 2B rho grid differs from pre-registration");
  require(config.hnsw_m == 16 && config.ef_construction == 80 &&
              config.fixed_ef_search == 256 && config.pq_m == 32 &&
              config.pq_nbits == 8,
          "Phase 2B graph/search/PQ parameters differ");
  require(config.ef_calibration ==
              std::vector<int>({256, 384, 512, 768, 1024}),
          "Phase 2B efSearch grid differs");
  require(config.sample_queries == 100 && config.distance_pairs == 100 &&
              config.order_pairs == 100,
          "Phase 2B PQ-quality sampling differs from Phase 1");
  require(config.exact_trigger == .90 && config.matched_target == .95 &&
              config.no_evidence_discovery == .01 &&
              config.no_evidence_recovery == .90 &&
              config.emerging_discovery == .03 &&
              config.emerging_recovery == .80 &&
              config.emerging_replicates == 3,
          "Phase 2B interpretation thresholds differ from pre-registration");
  return config;
}

std::vector<float> normal_values(std::uint32_t seed, std::size_t count) {
  std::mt19937 generator(seed);
  std::normal_distribution<float> distribution(0.0F, 1.0F);
  std::vector<float> output(count);
  for (float &value : output) value = distribution(generator);
  return output;
}

std::vector<int> labels(std::uint32_t seed, std::size_t count, int clusters) {
  std::mt19937 generator(seed);
  std::uniform_int_distribution<int> distribution(0, clusters - 1);
  std::vector<int> output(count);
  for (int &value : output) value = distribution(generator);
  return output;
}

std::vector<float> unit_centers(std::uint32_t seed, int clusters, int dimension) {
  auto output = normal_values(seed, static_cast<std::size_t>(clusters * dimension));
  for (int cluster = 0; cluster < clusters; ++cluster) {
    double squared = 0.0;
    for (int component = 0; component < dimension; ++component) {
      const float value = output[cluster * dimension + component];
      squared += static_cast<double>(value) * value;
    }
    const double norm = std::sqrt(squared);
    for (int component = 0; component < dimension; ++component)
      output[cluster * dimension + component] = static_cast<float>(
          output[cluster * dimension + component] / norm);
  }
  return output;
}

std::vector<float> mixture_vectors(const std::vector<float> &z,
                                   const std::vector<int> &point_labels,
                                   const std::vector<float> &centers,
                                   double rho, int dimension) {
  require(z.size() == point_labels.size() * static_cast<std::size_t>(dimension),
          "mixture latent/label size mismatch");
  std::vector<float> output(z.size());
  const double noise_scale = std::sqrt(1.0 - rho);
  const double center_scale = std::sqrt(rho * dimension);
  for (std::size_t point = 0; point < point_labels.size(); ++point) {
    for (int component = 0; component < dimension; ++component) {
      output[point * dimension + component] = static_cast<float>(
          noise_scale * z[point * dimension + component] +
          center_scale * centers[point_labels[point] * dimension + component]);
    }
  }
  return output;
}

Samples make_samples(const Config &config, std::uint32_t seed) {
  std::mt19937 generator(seed);
  std::uniform_int_distribution<faiss::idx_t> ids(0, config.base_vectors - 1);
  Samples output;
  output.distance_ids.resize(config.sample_queries);
  output.order_ids.resize(config.sample_queries);
  for (int query = 0; query < config.sample_queries; ++query) {
    for (int sample = 0; sample < config.distance_pairs; ++sample)
      output.distance_ids[query].push_back(ids(generator));
    for (int sample = 0; sample < config.order_pairs; ++sample) {
      faiss::idx_t first = ids(generator), second = ids(generator);
      while (second == first) second = ids(generator);
      output.order_ids[query].emplace_back(first, second);
    }
  }
  return output;
}

float exact_squared_l2(const float *first, const float *second, int dimension) {
  float total = 0.0F;
  for (int component = 0; component < dimension; ++component) {
    const float difference = first[component] - second[component];
    total += difference * difference;
  }
  return total;
}

std::string sha256(std::span<const std::uint8_t> bytes) {
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(
      EVP_MD_CTX_new(), EVP_MD_CTX_free);
  require(context && EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) == 1 &&
              EVP_DigestUpdate(context.get(), bytes.data(), bytes.size()) == 1,
          "cannot initialize SHA-256");
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
  return sha256(std::span<const std::uint8_t>(
      reinterpret_cast<const std::uint8_t *>(values.data()),
      values.size() * sizeof(*values.data())));
}

std::string token(double value) {
  std::ostringstream output;
  output << std::fixed << std::setprecision(2) << value;
  std::string result = output.str();
  for (char &character : result) if (character == '.') character = 'p';
  return result;
}

template <typename T>
void write_array(std::ostream &output, const std::vector<T> &values) {
  output << '[';
  for (std::size_t i = 0; i < values.size(); ++i)
    output << (i ? "," : "") << values[i];
  output << ']';
}

double vector_norm(const float *vector, int dimension) {
  double squared = 0.0;
  for (int i = 0; i < dimension; ++i)
    squared += static_cast<double>(vector[i]) * vector[i];
  return std::sqrt(squared);
}

std::pair<double, double> nearest_center_distances(
    const float *query, const std::vector<float> &centers, double rho,
    int clusters, int dimension) {
  const double scale = std::sqrt(rho * dimension);
  double first = INFINITY, second = INFINITY;
  for (int cluster = 0; cluster < clusters; ++cluster) {
    double squared = 0.0;
    for (int component = 0; component < dimension; ++component) {
      const double difference =
          query[component] - scale * centers[cluster * dimension + component];
      squared += difference * difference;
    }
    const double distance = std::sqrt(squared);
    if (distance < first) { second = first; first = distance; }
    else if (distance < second) second = distance;
  }
  return {first, second};
}

double generating_center_distance(const float *query, int label,
                                  const std::vector<float> &centers,
                                  double rho, int dimension) {
  const double scale = std::sqrt(rho * dimension);
  double squared = 0.0;
  for (int component = 0; component < dimension; ++component) {
    const double difference =
        query[component] - scale * centers[label * dimension + component];
    squared += difference * difference;
  }
  return std::sqrt(squared);
}

double mean_recall(const quant_hardness::SearchResults &results,
                   const std::vector<faiss::idx_t> &truth,
                   faiss::idx_t query_count, faiss::idx_t k) {
  double total = 0.0;
  for (faiss::idx_t query = 0; query < query_count; ++query) {
    total += quant_hardness::recall_at_k(
        std::span(results.ids).subspan(query * k, k),
        std::span(truth).subspan(query * k, k), k);
  }
  return total / query_count;
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

void write_query_rows(
    const std::filesystem::path &path, int replicate_id, double rho,
    const std::string &phase, int ef_search, const Config &config,
    const std::vector<int> &base_labels,
    const std::vector<float> &queries, const std::vector<int> &query_labels,
    const std::vector<float> &centers,
    const quant_hardness::GroundTruthResults &truth,
    const std::vector<quant_hardness::QueryDecompositionL0> &rows,
    const std::string &fingerprint) {
  std::ofstream output(path);
  require(output.good(), "cannot create per-query output");
  output << std::setprecision(17);
  for (faiss::idx_t query_id = 0; query_id < config.queries; ++query_id) {
    const auto &row = rows[query_id];
    require(row.delta_exact_control == 0.0,
            "corrected L0 exact control must be zero");
    const float *query = queries.data() + query_id * config.dimension;
    const std::size_t truth_offset = query_id * config.k;
    const double nearest = truth.distances[truth_offset];
    const double tenth = truth.distances[truth_offset + config.k - 1];
    const double nn_margin = tenth - nearest;
    const auto [first_center, second_center] = nearest_center_distances(
        query, centers, rho, config.clusters, config.dimension);
    const double center_margin = second_center - first_center;
    int component_count = 0;
    std::array<bool, 16> seen{};
    for (faiss::idx_t rank = 0; rank < config.k; ++rank) {
      const int label = base_labels[truth.ids[truth_offset + rank]];
      if (!seen[label]) { seen[label] = true; ++component_count; }
    }
    output << "{\"replicate_id\":" << replicate_id
           << ",\"query_id\":" << query_id << ",\"query_order\":" << query_id
           << ",\"rho\":" << rho << ",\"phase\":\"" << phase
           << "\",\"ef_search\":" << ef_search
           << ",\"graph_fingerprint\":\"" << fingerprint
           << "\",\"generating_cluster_id\":" << query_labels[query_id]
           << ",\"distance_to_generating_center\":"
           << generating_center_distance(query, query_labels[query_id], centers,
                                         rho, config.dimension)
           << ",\"distance_to_nearest_center\":" << first_center
           << ",\"distance_to_second_center\":" << second_center
           << ",\"cluster_center_margin\":" << center_margin
           << ",\"normalized_cluster_center_margin\":"
           << (second_center > 1e-12 ? center_margin / second_center : 0.0)
           << ",\"exact_nn_distance_squared\":" << nearest
           << ",\"exact_10th_distance_squared\":" << tenth
           << ",\"nn_margin_squared\":" << nn_margin
           << ",\"normalized_nn_margin_by_10th\":"
           << (tenth > 1e-12 ? nn_margin / tenth : 0.0)
           << ",\"query_l2_norm\":" << vector_norm(query, config.dimension)
           << ",\"ground_truth_component_count\":" << component_count
           << ",\"ground_truth_spans_multiple_components\":"
           << (component_count > 1 ? "true" : "false")
           << ",\"recall_exact_native\":" << row.recall_exact_native
           << ",\"recall_pq_native\":" << row.recall_pq_native
           << ",\"recall_exact_L0_oracle\":" << row.recall_exact_l0_oracle
           << ",\"recall_pq_L0_oracle\":" << row.recall_pq_l0_oracle
           << ",\"coverage_exact_L0\":" << row.coverage_exact_l0
           << ",\"coverage_pq_L0\":" << row.coverage_pq_l0
           << ",\"delta_total\":" << row.delta_total
           << ",\"delta_discovery\":" << row.delta_discovery
           << ",\"delta_ranking\":" << row.delta_ranking
           << ",\"delta_exact_control\":" << row.delta_exact_control
           << ",\"exact_L0_unique_distance_evaluations\":"
           << row.exact_l0_evaluated_ids.size()
           << ",\"pq_L0_unique_distance_evaluations\":"
           << row.pq_l0_evaluated_ids.size()
           << ",\"exact_upper_only_unique_distance_evaluations\":"
           << row.exact_upper_only_evaluated_ids.size()
           << ",\"pq_upper_only_unique_distance_evaluations\":"
           << row.pq_upper_only_evaluated_ids.size()
           << ",\"evaluated_L0_intersection_size\":"
           << row.evaluated_l0_intersection_size
           << ",\"evaluated_L0_jaccard\":" << row.evaluated_l0_jaccard
           << ",\"ground_truth_ids\":";
    write_array(output, row.ground_truth_ids);
    output << ",\"exact_native_result_ids\":"; write_array(output, row.exact_native_ids);
    output << ",\"pq_native_result_ids\":"; write_array(output, row.pq_native_ids);
    output << ",\"exact_L0_oracle_result_ids\":"; write_array(output, row.exact_l0_oracle_ids);
    output << ",\"pq_L0_oracle_result_ids\":"; write_array(output, row.pq_l0_oracle_ids);
    if (query_id < config.candidate_sample_queries) {
      output << ",\"exact_L0_evaluated_ids\":"; write_array(output, row.exact_l0_evaluated_ids);
      output << ",\"pq_L0_evaluated_ids\":"; write_array(output, row.pq_l0_evaluated_ids);
      output << ",\"exact_upper_only_evaluated_ids\":"; write_array(output, row.exact_upper_only_evaluated_ids);
      output << ",\"pq_upper_only_evaluated_ids\":"; write_array(output, row.pq_upper_only_evaluated_ids);
    } else {
      output << ",\"exact_L0_evaluated_ids\":null,\"pq_L0_evaluated_ids\":null"
                ",\"exact_upper_only_evaluated_ids\":null"
                ",\"pq_upper_only_evaluated_ids\":null";
    }
    output << "}\n";
  }
  require(output.good(), "failed while writing per-query rows");
}

void write_quality(const std::filesystem::path &directory, int replicate_id,
                   double rho, const Config &config,
                   const std::vector<float> &base,
                   const std::vector<float> &queries, faiss::IndexPQ &pq,
                   const Samples &samples) {
  std::ofstream distances(directory / "distance_samples.jsonl");
  std::ofstream orders(directory / "order_samples.jsonl");
  require(distances.good() && orders.good(), "cannot create PQ-quality output");
  distances << std::setprecision(17);
  orders << std::setprecision(17);
  std::unique_ptr<faiss::DistanceComputer> computer(pq.get_distance_computer());
  for (int query_id = 0; query_id < config.sample_queries; ++query_id) {
    const float *query = queries.data() + query_id * config.dimension;
    computer->set_query(query);
    for (int sample_id = 0; sample_id < config.distance_pairs; ++sample_id) {
      const faiss::idx_t database_id = samples.distance_ids[query_id][sample_id];
      const float exact = exact_squared_l2(
          query, base.data() + database_id * config.dimension, config.dimension);
      const float approximate = (*computer)(database_id);
      distances << "{\"replicate_id\":" << replicate_id << ",\"rho\":" << rho
                << ",\"query_id\":" << query_id << ",\"sample_id\":" << sample_id
                << ",\"database_id\":" << database_id
                << ",\"exact_distance\":" << exact
                << ",\"pq_distance\":" << approximate
                << ",\"absolute_error\":" << std::abs(approximate - exact)
                << ",\"absolute_relative_error\":"
                << (exact > 1e-12F ? std::abs(approximate - exact) / exact : 0.0F)
                << "}\n";
    }
    for (int sample_id = 0; sample_id < config.order_pairs; ++sample_id) {
      const auto [first_id, second_id] = samples.order_ids[query_id][sample_id];
      const float exact_first = exact_squared_l2(
          query, base.data() + first_id * config.dimension, config.dimension);
      const float exact_second = exact_squared_l2(
          query, base.data() + second_id * config.dimension, config.dimension);
      const float pq_first = (*computer)(first_id);
      const float pq_second = (*computer)(second_id);
      const bool inversion =
          (exact_first < exact_second && pq_first > pq_second) ||
          (exact_first > exact_second && pq_first < pq_second);
      orders << "{\"replicate_id\":" << replicate_id << ",\"rho\":" << rho
             << ",\"query_id\":" << query_id << ",\"sample_id\":" << sample_id
             << ",\"first_database_id\":" << first_id
             << ",\"second_database_id\":" << second_id
             << ",\"exact_first\":" << exact_first
             << ",\"exact_second\":" << exact_second
             << ",\"pq_first\":" << pq_first << ",\"pq_second\":" << pq_second
             << ",\"strict_inversion\":" << (inversion ? "true" : "false")
             << ",\"pq_tie\":" << (pq_first == pq_second ? "true" : "false")
             << "}\n";
    }
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    require(argc == 3,
            "usage: faiss_clustered_geometry CONFIG NEW_OUTPUT_DIRECTORY");
    const Config config = read_config(argv[1]);
    const std::filesystem::path root(argv[2]);
    require(!std::filesystem::exists(root), "refusing to overwrite output");
    std::filesystem::create_directories(root);
    std::ifstream source_config(argv[1]);
    std::ofstream resolved(root / "resolved_config.conf");
    resolved << source_config.rdbuf();
    std::ofstream calibration(root / "exact_recall_calibration.csv");
    calibration << "replicate_id,rho,ef_search,mean_exact_recall,selected_matched_ef\n";
    std::ofstream manifest(root / "manifest.json");
    manifest << "{\n  \"schema_version\":1,\n  \"run_id\":\"phase2b_clustered_geometry_v1\","
             << "\n  \"git_commit\":\"" << QH_GIT_COMMIT
             << "\",\n  \"dirty_worktree\":" << QH_GIT_DIRTY
             << ",\n  \"faiss_commit\":\"" << QH_FAISS_COMMIT
             << "\",\n  \"replicate_count\":4,\n  \"condition_count_per_replicate\":6,"
             << "\n  \"distribution\":\"balanced 16-component Gaussian mixture; "
                "query and database IID conditional on shared centers\","
             << "\n  \"generator\":\"sqrt(1-rho)*z + sqrt(rho*d)*c_j; "
                "unit c_j; uniform j\","
             << "\n  \"fixed_parameters\":{\"base_vectors\":20000,\"queries\":500,"
                "\"dimension\":64,\"clusters\":16,\"hnsw_m\":16,"
                "\"ef_construction\":80,\"fixed_ef_search\":256,"
                "\"pq_m\":32,\"pq_nbits\":8,\"k\":10},"
             << "\n  \"candidate_semantics\":\"L0-only including the re-evaluated "
                "level-0 seed; upper-only sets recorded separately\","
             << "\n  \"pq_quality_sampling\":{\"queries\":100,"
                "\"distance_pairs_per_query\":100,\"order_pairs_per_query\":100,"
                "\"relative_error\":\"absolute error / exact distance\","
                "\"inversion\":\"strict reversal; ties are not inversions\"},"
             << "\n  \"pre_registered_thresholds\":{"
                "\"exact_recall_calibration_trigger\":0.90,"
                "\"matched_exact_recall_target\":0.95,"
                "\"no_evidence_max_abs_mean_delta_discovery\":0.01,"
                "\"no_evidence_min_rerank_recovery\":0.90,"
                "\"emerging_min_mean_delta_discovery\":0.03,"
                "\"emerging_max_rerank_recovery\":0.80,"
                "\"emerging_required_replicates\":3},"
             << "\n  \"machine\":{\"hostname\":\"" << hostname()
             << "\",\"kernel\":\"" << kernel()
             << "\",\"hardware_concurrency\":" << std::thread::hardware_concurrency()
             << ",\"compiler\":\"" << __VERSION__ << "\"}\n}\n";
    omp_set_num_threads(1);

    for (std::size_t replicate_id = 0; replicate_id < config.seeds.size();
         ++replicate_id) {
      const Seeds seeds = config.seeds[replicate_id];
      const auto database_z = normal_values(
          seeds.database_z, config.base_vectors * config.dimension);
      const auto query_z =
          normal_values(seeds.query_z, config.queries * config.dimension);
      const auto centers = unit_centers(seeds.centers, config.clusters,
                                        config.dimension);
      const auto database_labels = labels(
          seeds.database_labels, config.base_vectors, config.clusters);
      const auto query_labels =
          labels(seeds.query_labels, config.queries, config.clusters);
      const Samples samples = make_samples(config, seeds.samples);
      const auto replicate_directory =
          root / ("replicate_" + (replicate_id < 10 ? std::string("0") : "") +
                  std::to_string(replicate_id));
      std::filesystem::create_directories(replicate_directory);

      for (const double rho : config.rhos) {
        const auto condition_directory =
            replicate_directory / ("rho_" + token(rho));
        std::filesystem::create_directories(condition_directory);
        const auto base = mixture_vectors(database_z, database_labels, centers,
                                          rho, config.dimension);
        const auto queries = mixture_vectors(query_z, query_labels, centers,
                                             rho, config.dimension);
        const auto truth = quant_hardness::exhaustive_l2_top_k(
            base.data(), config.base_vectors, queries.data(), config.queries,
            config.dimension, config.k);

        faiss::IndexFlatL2 exact_storage(config.dimension);
        faiss::IndexHNSW graph(&exact_storage, config.hnsw_m);
        graph.hnsw.efConstruction = config.ef_construction;
        graph.hnsw.rng = faiss::RandomGenerator(seeds.graph);
        graph.add(config.base_vectors, base.data());
        const std::string graph_hash =
            quant_hardness::graph_fingerprint(graph.hnsw);
        faiss::IndexPQ pq(config.dimension, config.pq_m, config.pq_nbits,
                          faiss::METRIC_L2);
        pq.pq.cp.seed = static_cast<int>(seeds.pq);
        pq.train(config.base_vectors, base.data());
        pq.add(config.base_vectors, base.data());
        const std::string codebook_hash = bytes_hash(pq.pq.centroids);
        const std::string codes_hash = bytes_hash(pq.codes);

        std::ofstream condition_manifest(condition_directory / "manifest.json");
        condition_manifest << "{\n  \"schema_version\":1,\n  \"replicate_id\":"
                           << replicate_id << ",\n  \"rho\":" << rho
                           << ",\n  \"seeds\":{\"database_z\":" << seeds.database_z
                           << ",\"query_z\":" << seeds.query_z
                           << ",\"centers\":" << seeds.centers
                           << ",\"database_labels\":" << seeds.database_labels
                           << ",\"query_labels\":" << seeds.query_labels
                           << ",\"graph\":" << seeds.graph
                           << ",\"pq\":" << seeds.pq
                           << ",\"samples\":" << seeds.samples
                           << "},\n  \"graph_fingerprint\":\"" << graph_hash
                           << "\",\n  \"pq_codebook_sha256\":\"" << codebook_hash
                           << "\",\n  \"pq_codes_sha256\":\"" << codes_hash
                           << "\",\n  \"database_vectors_sha256\":\""
                           << bytes_hash(base) << "\",\n  \"query_vectors_sha256\":\""
                           << bytes_hash(queries) << "\",\n  \"cluster_centers_sha256\":\""
                           << bytes_hash(centers) << "\"\n}\n";

        faiss::SearchParametersHNSW fixed;
        fixed.efSearch = config.fixed_ef_search;
        fixed.bounded_queue = true;
        fixed.check_relative_distance = true;
        const auto fixed_rows = quant_hardness::measure_paired_decomposition_l0(
            graph, exact_storage, pq, base.data(), config.base_vectors,
            queries.data(), config.queries, config.dimension, config.k,
            truth.ids, fixed, true);
        write_query_rows(condition_directory / "fixed.jsonl", replicate_id,
                         rho, "fixed", config.fixed_ef_search, config,
                         database_labels, queries, query_labels, centers, truth,
                         fixed_rows, graph_hash);
        write_quality(condition_directory, replicate_id, rho, config, base,
                      queries, pq, samples);
        double exact_mean = 0.0;
        for (const auto &row : fixed_rows) exact_mean += row.recall_exact_native;
        exact_mean /= fixed_rows.size();
        if (exact_mean < config.exact_trigger) {
          int selected = 0;
          std::vector<std::pair<int, double>> sweep;
          for (const int ef : config.ef_calibration) {
            faiss::SearchParametersHNSW parameters;
            parameters.efSearch = ef;
            parameters.bounded_queue = true;
            parameters.check_relative_distance = true;
            const auto exact = quant_hardness::search_with_storage(
                graph, exact_storage, queries.data(), config.queries,
                config.k, parameters);
            const double recall =
                mean_recall(exact, truth.ids, config.queries, config.k);
            sweep.emplace_back(ef, recall);
            if (selected == 0 && recall >= config.matched_target) selected = ef;
          }
          for (const auto &[ef, recall] : sweep)
            calibration << replicate_id << ',' << rho << ',' << ef << ','
                        << std::setprecision(17) << recall << ',' << selected << '\n';
          if (selected != 0) {
            faiss::SearchParametersHNSW matched;
            matched.efSearch = selected;
            matched.bounded_queue = true;
            matched.check_relative_distance = true;
            const auto matched_rows =
                quant_hardness::measure_paired_decomposition_l0(
                    graph, exact_storage, pq, base.data(), config.base_vectors,
                    queries.data(), config.queries, config.dimension, config.k,
                    truth.ids, matched, true);
            write_query_rows(condition_directory / "matched.jsonl", replicate_id,
                             rho, "matched", selected, config,
                             database_labels, queries, query_labels, centers,
                             truth, matched_rows, graph_hash);
          }
        }
        require(quant_hardness::graph_fingerprint(graph.hnsw) == graph_hash,
                "condition changed graph fingerprint");
        require(bytes_hash(pq.pq.centroids) == codebook_hash &&
                    bytes_hash(pq.codes) == codes_hash,
                "condition changed PQ model or codes");
        std::cout << "replicate=" << replicate_id << " rho=" << rho
                  << " exact_recall=" << exact_mean << " status=PASS\n";
      }
    }
    require(resolved.good() && calibration.good() && manifest.good(),
            "failed while writing run metadata");
    std::cout << "run_directory=" << root.string() << "\nstatus=PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "status=FAIL\nerror=" << error.what() << '\n';
    return 1;
  }
}
