#include "graph/faiss_shared_hnsw.h"
#include "instrumentation/paired_decomposition.h"
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
#include <vector>

namespace {

struct Seeds {
  std::uint32_t database;
  std::uint32_t query_latent;
  std::uint32_t graph;
  std::uint32_t pq;
  std::uint32_t direction;
};

struct Config {
  std::vector<Seeds> seeds;
  faiss::idx_t base_vectors;
  faiss::idx_t queries;
  int dimension;
  faiss::idx_t k;
  int hnsw_m;
  int ef_construction;
  int fixed_ef_search;
  std::vector<int> ef_calibration_values;
  int pq_m;
  int pq_nbits;
  std::vector<double> alphas;
  std::vector<double> sigmas;
  double exact_recall_trigger;
  double matched_exact_target;
  double no_evidence_discovery;
  double no_evidence_recovery;
  double emerging_discovery;
  double emerging_recovery;
  int emerging_replicates;
  int candidate_sample_queries;
};

struct Condition {
  std::string id;
  std::string family;
  double severity;
  std::vector<float> queries;
};

void require(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

std::vector<std::string> split(const std::string &value, char delimiter) {
  std::stringstream stream(value);
  std::vector<std::string> output;
  std::string item;
  while (std::getline(stream, item, delimiter)) {
    require(!item.empty(), "empty list item");
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
  const auto integer = [&](const std::string &key) {
    return std::stoll(get(key));
  };
  Config config{{},
                integer("base_vectors"),
                integer("queries"),
                static_cast<int>(integer("dimension")),
                integer("k"),
                static_cast<int>(integer("hnsw_m")),
                static_cast<int>(integer("ef_construction")),
                static_cast<int>(integer("fixed_ef_search")),
                {},
                static_cast<int>(integer("pq_m")),
                static_cast<int>(integer("pq_nbits")),
                {},
                {},
                std::stod(get("exact_recall_calibration_trigger")),
                std::stod(get("matched_exact_recall_target")),
                std::stod(get("no_evidence_max_abs_mean_delta_discovery")),
                std::stod(get("no_evidence_min_rerank_recovery")),
                std::stod(get("emerging_min_mean_delta_discovery")),
                std::stod(get("emerging_max_rerank_recovery")),
                static_cast<int>(integer("emerging_required_replicates")),
                static_cast<int>(integer("candidate_set_sample_queries"))};
  for (const std::string &tuple : split(get("seed_tuples"), ',')) {
    const auto fields = split(tuple, ':');
    require(fields.size() == 5, "seed tuple must have five fields");
    config.seeds.push_back({static_cast<std::uint32_t>(std::stoul(fields[0])),
                            static_cast<std::uint32_t>(std::stoul(fields[1])),
                            static_cast<std::uint32_t>(std::stoul(fields[2])),
                            static_cast<std::uint32_t>(std::stoul(fields[3])),
                            static_cast<std::uint32_t>(std::stoul(fields[4]))});
  }
  for (const std::string &item : split(get("ef_calibration_values"), ',')) {
    config.ef_calibration_values.push_back(std::stoi(item));
  }
  for (const std::string &item : split(get("mean_shift_alphas"), ',')) {
    config.alphas.push_back(std::stod(item));
  }
  for (const std::string &item : split(get("radial_shift_sigmas"), ',')) {
    config.sigmas.push_back(std::stod(item));
  }
  require(values.size() == 21, "config contains unknown keys");
  require(config.seeds.size() == 4, "Phase 2A requires four replicates");
  require(config.base_vectors == 20000 && config.queries == 500 &&
              config.dimension == 64 && config.k == 10,
          "Phase 2A data dimensions are fixed");
  require(config.hnsw_m == 16 && config.ef_construction == 80 &&
              config.fixed_ef_search == 256,
          "Phase 2A HNSW parameters are fixed");
  require(config.pq_m == 32 && config.pq_nbits == 8,
          "Phase 2A requires PQ32x8");
  require(config.alphas == std::vector<double>({0.0, 0.25, 0.5, 0.75, 1.0}) &&
              config.sigmas == std::vector<double>({1.0, 1.25, 1.5, 2.0}),
          "Phase 2A shift grid differs from pre-registration");
  require(config.ef_calibration_values ==
              std::vector<int>({256, 384, 512, 768, 1024}),
          "Phase 2A efSearch calibration grid differs");
  require(config.exact_recall_trigger == 0.90 &&
              config.matched_exact_target == 0.95 &&
              config.no_evidence_discovery == 0.01 &&
              config.no_evidence_recovery == 0.90 &&
              config.emerging_discovery == 0.03 &&
              config.emerging_recovery == 0.80 &&
              config.emerging_replicates == 3,
          "Phase 2A interpretation thresholds differ from pre-registration");
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

std::vector<float> unit_direction(std::uint32_t seed, int dimension) {
  std::vector<float> direction = normal_vectors(seed, 1, dimension);
  double squared_norm = 0.0;
  for (const float value : direction) {
    squared_norm += static_cast<double>(value) * value;
  }
  const double norm = std::sqrt(squared_norm);
  for (float &value : direction) {
    value = static_cast<float>(value / norm);
  }
  return direction;
}

std::string severity_token(double value) {
  std::ostringstream output;
  output << std::fixed << std::setprecision(2) << value;
  std::string token = output.str();
  for (char &character : token) {
    if (character == '.') {
      character = 'p';
    }
  }
  return token;
}

std::vector<Condition> make_conditions(const Config &config,
                                       const std::vector<float> &latent,
                                       const std::vector<float> &direction) {
  std::vector<Condition> conditions;
  conditions.push_back({"iid", "iid", 0.0, latent});
  const double shift_scale = std::sqrt(config.dimension);
  for (const double alpha : config.alphas) {
    if (alpha == 0.0) {
      continue;
    }
    std::vector<float> queries = latent;
    for (faiss::idx_t query = 0; query < config.queries; ++query) {
      for (int component = 0; component < config.dimension; ++component) {
        queries[query * config.dimension + component] +=
            static_cast<float>(alpha * shift_scale * direction[component]);
      }
    }
    conditions.push_back({"mean_alpha_" + severity_token(alpha), "mean", alpha,
                          std::move(queries)});
  }
  for (const double sigma : config.sigmas) {
    if (sigma == 1.0) {
      continue;
    }
    std::vector<float> queries = latent;
    for (float &value : queries) {
      value = static_cast<float>(sigma * value);
    }
    conditions.push_back({"radial_sigma_" + severity_token(sigma), "radial",
                          sigma, std::move(queries)});
  }
  return conditions;
}

std::string sha256(std::span<const std::uint8_t> bytes) {
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(
      EVP_MD_CTX_new(), EVP_MD_CTX_free);
  require(context &&
              EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) == 1 &&
              EVP_DigestUpdate(context.get(), bytes.data(), bytes.size()) == 1,
          "cannot initialize SHA-256");
  std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
  unsigned int size = 0;
  require(EVP_DigestFinal_ex(context.get(), digest.data(), &size) == 1,
          "cannot finalize SHA-256");
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (unsigned int i = 0; i < size; ++i) {
    output << std::setw(2) << static_cast<unsigned int>(digest[i]);
  }
  return output.str();
}

std::string float_vector_hash(const std::vector<float> &values) {
  return sha256(std::span(reinterpret_cast<const std::uint8_t *>(values.data()),
                          values.size() * sizeof(float)));
}

template <typename ByteContainer>
std::string byte_vector_hash(const ByteContainer &values) {
  return sha256(std::span<const std::uint8_t>(values.data(), values.size()));
}

std::vector<double> centroid(const std::vector<float> &base, faiss::idx_t count,
                             int dimension) {
  std::vector<double> result(dimension, 0.0);
  for (faiss::idx_t vector = 0; vector < count; ++vector) {
    for (int component = 0; component < dimension; ++component) {
      result[component] += base[vector * dimension + component];
    }
  }
  for (double &value : result) {
    value /= count;
  }
  return result;
}

double vector_norm(const float *vector, int dimension) {
  double squared = 0.0;
  for (int component = 0; component < dimension; ++component) {
    squared += static_cast<double>(vector[component]) * vector[component];
  }
  return std::sqrt(squared);
}

double centroid_distance(const float *query,
                         const std::vector<double> &database_centroid,
                         int dimension) {
  double squared = 0.0;
  for (int component = 0; component < dimension; ++component) {
    const double difference = query[component] - database_centroid[component];
    squared += difference * difference;
  }
  return std::sqrt(squared);
}

double projection(const float *query, const std::vector<float> &direction,
                  int dimension) {
  double result = 0.0;
  for (int component = 0; component < dimension; ++component) {
    result += static_cast<double>(query[component]) * direction[component];
  }
  return result;
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
void write_array(std::ostream &output, const std::vector<Integer> &values) {
  output << '[';
  for (std::size_t i = 0; i < values.size(); ++i) {
    output << (i == 0 ? "" : ",") << values[i];
  }
  output << ']';
}

double mean_recall(const quant_hardness::SearchResults &results,
                   const std::vector<faiss::idx_t> &truth,
                   faiss::idx_t query_count, faiss::idx_t k) {
  double total = 0.0;
  for (faiss::idx_t query = 0; query < query_count; ++query) {
    const std::size_t offset = query * k;
    total +=
        quant_hardness::recall_at_k(std::span(results.ids).subspan(offset, k),
                                    std::span(truth).subspan(offset, k), k);
  }
  return total / query_count;
}

void write_rows(const std::filesystem::path &path, int replicate_id,
                const Condition &condition, const std::string &phase,
                int ef_search, const Config &config,
                const std::vector<double> &database_centroid,
                const std::vector<float> &direction,
                const quant_hardness::GroundTruthResults &truth,
                const std::vector<quant_hardness::QueryDecomposition> &rows) {
  std::ofstream output(path);
  require(output.good(), "cannot create per-condition raw output");
  output << std::setprecision(17);
  const double root_dimension = std::sqrt(config.dimension);
  for (faiss::idx_t query_id = 0; query_id < config.queries; ++query_id) {
    const auto &row = rows[query_id];
    const float *query = condition.queries.data() + query_id * config.dimension;
    const double norm = vector_norm(query, config.dimension);
    const double center_distance =
        centroid_distance(query, database_centroid, config.dimension);
    const std::size_t truth_offset = query_id * config.k;
    const double nearest = truth.distances[truth_offset];
    const double tenth = truth.distances[truth_offset + config.k - 1];
    const double margin = tenth - nearest;
    output << "{\"replicate_id\":" << replicate_id
           << ",\"query_id\":" << query_id << ",\"query_order\":" << query_id
           << ",\"phase\":\"" << phase << "\",\"condition_id\":\""
           << condition.id << "\",\"shift_family\":\"" << condition.family
           << "\",\"shift_severity\":" << condition.severity
           << ",\"ef_search\":" << ef_search << ",\"query_l2_norm\":" << norm
           << ",\"query_norm_over_sqrt_dimension\":" << norm / root_dimension
           << ",\"distance_to_database_centroid\":" << center_distance
           << ",\"centroid_distance_over_sqrt_dimension\":"
           << center_distance / root_dimension
           << ",\"exact_nn_distance_squared\":" << nearest
           << ",\"exact_10th_distance_squared\":" << tenth
           << ",\"nn_margin_squared\":" << margin
           << ",\"normalized_nn_margin_by_10th\":"
           << (tenth > 1e-12 ? margin / tenth : 0.0)
           << ",\"nearest_to_10th_distance_ratio\":"
           << (tenth > 1e-12 ? nearest / tenth : 0.0)
           << ",\"projection_onto_mean_shift_direction\":";
    if (condition.family == "mean" || condition.family == "iid") {
      output << projection(query, direction, config.dimension);
    } else {
      output << "null";
    }
    output << ",\"recall_exact_native\":" << row.recall_exact_native
           << ",\"recall_pq_native\":" << row.recall_pq_native
           << ",\"recall_exact_oracle\":" << row.recall_exact_oracle
           << ",\"recall_pq_oracle\":" << row.recall_pq_oracle
           << ",\"coverage_exact\":" << row.coverage_exact
           << ",\"coverage_pq\":" << row.coverage_pq
           << ",\"delta_total\":" << row.delta_total
           << ",\"delta_discovery\":" << row.delta_discovery
           << ",\"delta_ranking\":" << row.delta_ranking
           << ",\"delta_exact_control\":" << row.delta_exact_control
           << ",\"exact_unique_distance_evaluations\":"
           << row.exact_evaluated_ids.size()
           << ",\"pq_unique_distance_evaluations\":"
           << row.pq_evaluated_ids.size() << ",\"evaluated_intersection_size\":"
           << row.evaluated_intersection_size
           << ",\"evaluated_jaccard\":" << row.evaluated_jaccard
           << ",\"ground_truth_ids\":";
    write_array(output, row.ground_truth_ids);
    output << ",\"exact_native_result_ids\":";
    write_array(output, row.exact_native_ids);
    output << ",\"pq_native_result_ids\":";
    write_array(output, row.pq_native_ids);
    output << ",\"exact_oracle_result_ids\":";
    write_array(output, row.exact_oracle_ids);
    output << ",\"pq_oracle_result_ids\":";
    write_array(output, row.pq_oracle_ids);
    if (query_id < config.candidate_sample_queries) {
      output << ",\"exact_evaluated_ids\":";
      write_array(output, row.exact_evaluated_ids);
      output << ",\"pq_evaluated_ids\":";
      write_array(output, row.pq_evaluated_ids);
    } else {
      output << ",\"exact_evaluated_ids\":null,\"pq_evaluated_ids\":null";
    }
    output << "}\n";
  }
  require(output.good(), "failed while writing per-condition raw output");
}

} // namespace

int main(int argc, char **argv) {
  try {
    require(argc == 3, "usage: faiss_query_shift CONFIG NEW_OUTPUT_DIRECTORY");
    const Config config = read_config(argv[1]);
    const std::filesystem::path root(argv[2]);
    require(!std::filesystem::exists(root),
            "refusing to overwrite Phase 2A output directory");
    std::filesystem::create_directories(root);
    std::ifstream source_config(argv[1]);
    std::ofstream resolved(root / "resolved_config.conf");
    resolved << source_config.rdbuf();
    std::ofstream calibration(root / "exact_recall_calibration.csv");
    calibration << "replicate_id,condition_id,ef_search,mean_exact_recall,"
                   "selected_matched_ef\n";
    std::ofstream root_manifest(root / "manifest.json");
    root_manifest
        << "{\n  \"schema_version\":1,\n  "
           "\"run_id\":\"phase2a_query_shift_v1\","
           "\n  \"git_commit\":\""
        << QH_GIT_COMMIT << "\",\n  \"dirty_worktree\":" << QH_GIT_DIRTY
        << ",\n  \"faiss_commit\":\"" << QH_FAISS_COMMIT
        << "\",\n  \"replicate_count\":4,\n  \"database_distribution\":"
           "\"N(0,I_64), unchanged across query conditions within replicate\","
           "\n  \"query_coupling\":\"one latent z per query reused across all "
           "alpha and sigma severities; IID is shared alpha=0/sigma=1 "
           "control\","
           "\n  \"fixed_parameters\":{\"base_vectors\":20000,\"queries\":500,"
           "\"dimension\":64,\"hnsw_m\":16,\"ef_construction\":80,"
           "\"fixed_ef_search\":256,\"pq_m\":32,\"pq_nbits\":8,"
           "\"k\":10},\n  \"graph_parameters\":{\"distance\":"
           "\"FP32 squared L2\",\"M\":16,\"efConstruction\":80},"
           "\n  \"search_parameters\":{\"k\":10,\"fixed_efSearch\":256,"
           "\"bounded_queue\":true,\"check_relative_distance\":true,"
           "\"threads\":1,\"query_order\":\"ascending query_id\"},"
           "\n  \"quantizer_parameters\":{\"type\":\"FAISS PQ ADC\","
           "\"M\":32,\"nbits\":8,\"training_vectors\":20000,"
           "\"encoded_vectors\":20000},"
           "\n  \"ground_truth\":\"exhaustive FP32 squared-L2 top-10, "
           "recomputed per query condition\","
           "\n  \"pre_registered_thresholds\":{"
           "\"exact_recall_calibration_trigger\":0.90,"
           "\"matched_exact_recall_target\":0.95,"
           "\"no_evidence_max_abs_mean_delta_discovery\":0.01,"
           "\"no_evidence_min_rerank_recovery\":0.90,"
           "\"emerging_min_mean_delta_discovery\":0.03,"
           "\"emerging_max_rerank_recovery\":0.80,"
           "\"emerging_required_replicates\":3},"
           "\n  \"candidate_set_storage\":\"all scalar metrics for every "
           "query; sorted V_E/V_P only for query IDs 0 through 24 per "
           "condition\",\n  \"machine\":{\"hostname\":\""
        << hostname() << "\",\"kernel\":\"" << kernel()
        << "\",\"hardware_concurrency\":" << std::thread::hardware_concurrency()
        << ",\"compiler\":\"" << __VERSION__ << "\"}\n}\n";
    omp_set_num_threads(1);

    for (std::size_t replicate_id = 0; replicate_id < config.seeds.size();
         ++replicate_id) {
      const Seeds seeds = config.seeds[replicate_id];
      const std::filesystem::path replicate_directory =
          root / ("replicate_" +
                  (replicate_id < 10 ? std::string("0") : std::string()) +
                  std::to_string(replicate_id));
      std::filesystem::create_directories(replicate_directory / "fixed");
      std::filesystem::create_directories(replicate_directory / "matched");

      const std::vector<float> base =
          normal_vectors(seeds.database, config.base_vectors, config.dimension);
      const std::vector<float> latent =
          normal_vectors(seeds.query_latent, config.queries, config.dimension);
      const std::vector<float> direction =
          unit_direction(seeds.direction, config.dimension);
      const std::vector<double> database_centroid =
          centroid(base, config.base_vectors, config.dimension);
      const auto conditions = make_conditions(config, latent, direction);

      faiss::IndexFlatL2 exact_storage(config.dimension);
      faiss::IndexHNSW graph(&exact_storage, config.hnsw_m);
      graph.hnsw.efConstruction = config.ef_construction;
      graph.hnsw.rng = faiss::RandomGenerator(seeds.graph);
      graph.add(config.base_vectors, base.data());
      const std::string graph_hash =
          quant_hardness::graph_fingerprint(graph.hnsw);

      faiss::IndexPQ pq_storage(config.dimension, config.pq_m, config.pq_nbits,
                                faiss::METRIC_L2);
      pq_storage.pq.cp.seed = static_cast<int>(seeds.pq);
      pq_storage.train(config.base_vectors, base.data());
      pq_storage.add(config.base_vectors, base.data());
      const std::string codebook_hash =
          float_vector_hash(pq_storage.pq.centroids);
      const std::string codes_hash = byte_vector_hash(pq_storage.codes);

      std::ofstream replicate_manifest(replicate_directory / "manifest.json");
      replicate_manifest << "{\n  \"schema_version\":1,\n  \"replicate_id\":"
                         << replicate_id << ",\n  \"git_commit\":\""
                         << QH_GIT_COMMIT
                         << "\",\n  \"dirty_worktree\":" << QH_GIT_DIRTY
                         << ",\n  \"faiss_commit\":\"" << QH_FAISS_COMMIT
                         << "\",\n  \"seeds\":{\"database\":" << seeds.database
                         << ",\"query_latent\":" << seeds.query_latent
                         << ",\"graph\":" << seeds.graph
                         << ",\"pq\":" << seeds.pq
                         << ",\"direction\":" << seeds.direction
                         << "},\n  \"graph_fingerprint\":\"" << graph_hash
                         << "\",\n  \"pq_codebook_sha256\":\"" << codebook_hash
                         << "\",\n  \"pq_codes_sha256\":\"" << codes_hash
                         << "\",\n  \"mean_shift_unit_direction\":";
      write_array(replicate_manifest, direction);
      replicate_manifest
          << ",\n  \"candidate_sets_preserved_for_query_ids\":\"0 through "
          << config.candidate_sample_queries - 1 << ", inclusive\"\n}\n";

      for (const Condition &condition : conditions) {
        const auto truth = quant_hardness::exhaustive_l2_top_k(
            base.data(), config.base_vectors, condition.queries.data(),
            config.queries, config.dimension, config.k);
        faiss::SearchParametersHNSW fixed_parameters;
        fixed_parameters.efSearch = config.fixed_ef_search;
        fixed_parameters.bounded_queue = true;
        fixed_parameters.check_relative_distance = true;
        const auto fixed_rows = quant_hardness::measure_paired_decomposition(
            graph, exact_storage, pq_storage, base.data(), config.base_vectors,
            condition.queries.data(), config.queries, config.dimension,
            config.k, truth.ids, fixed_parameters, true);
        write_rows(replicate_directory / "fixed" / (condition.id + ".jsonl"),
                   replicate_id, condition, "fixed", config.fixed_ef_search,
                   config, database_centroid, direction, truth, fixed_rows);
        double exact_mean = 0.0;
        for (const auto &row : fixed_rows) {
          exact_mean += row.recall_exact_native;
        }
        exact_mean /= fixed_rows.size();

        if (exact_mean < config.exact_recall_trigger) {
          int selected_ef = 0;
          std::vector<std::pair<int, double>> sweep;
          for (const int ef_search : config.ef_calibration_values) {
            faiss::SearchParametersHNSW parameters;
            parameters.efSearch = ef_search;
            parameters.bounded_queue = true;
            parameters.check_relative_distance = true;
            const auto exact = quant_hardness::search_with_storage(
                graph, exact_storage, condition.queries.data(), config.queries,
                config.k, parameters);
            const double recall =
                mean_recall(exact, truth.ids, config.queries, config.k);
            sweep.emplace_back(ef_search, recall);
            if (selected_ef == 0 && recall >= config.matched_exact_target) {
              selected_ef = ef_search;
            }
          }
          for (const auto &[ef_search, recall] : sweep) {
            calibration << replicate_id << ',' << condition.id << ','
                        << ef_search << ',' << std::setprecision(17) << recall
                        << ',' << selected_ef << '\n';
          }
          if (selected_ef != 0) {
            faiss::SearchParametersHNSW matched_parameters;
            matched_parameters.efSearch = selected_ef;
            matched_parameters.bounded_queue = true;
            matched_parameters.check_relative_distance = true;
            const auto matched_rows =
                quant_hardness::measure_paired_decomposition(
                    graph, exact_storage, pq_storage, base.data(),
                    config.base_vectors, condition.queries.data(),
                    config.queries, config.dimension, config.k, truth.ids,
                    matched_parameters, true);
            write_rows(replicate_directory / "matched" /
                           (condition.id + ".jsonl"),
                       replicate_id, condition, "matched", selected_ef, config,
                       database_centroid, direction, truth, matched_rows);
          }
        }
        require(quant_hardness::graph_fingerprint(graph.hnsw) == graph_hash,
                "query condition changed graph fingerprint");
        require(float_vector_hash(pq_storage.pq.centroids) == codebook_hash &&
                    byte_vector_hash(pq_storage.codes) == codes_hash,
                "query condition changed PQ codebooks or codes");
      }
      std::cout << "replicate=" << replicate_id
                << " graph_fingerprint=" << graph_hash << " status=PASS\n";
    }
    require(calibration.good() && resolved.good() && root_manifest.good(),
            "failed while writing Phase 2A metadata");
    std::cout << "run_directory=" << root.string() << "\nstatus=PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "status=FAIL\nerror=" << error.what() << '\n';
    return 1;
  }
}
