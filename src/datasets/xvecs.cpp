#include "datasets/xvecs.h"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace quant_hardness {
namespace {

template <typename Value>
struct Matrix {
  std::size_t rows;
  int dimension;
  std::vector<Value> values;
};

template <typename Value>
Matrix<Value> read_xvecs(const std::string &path, bool floating_point) {
  static_assert(sizeof(Value) == sizeof(std::uint32_t));
  std::ifstream input(path, std::ios::binary);
  if (!input.good()) throw std::runtime_error("cannot open xvec file: " + path);
  std::int32_t dimension = 0;
  input.read(reinterpret_cast<char *>(&dimension), sizeof(dimension));
  if (!input || dimension <= 0)
    throw std::runtime_error("invalid first xvec dimension: " + path);
  const std::uintmax_t record_bytes =
      sizeof(std::int32_t) + static_cast<std::uintmax_t>(dimension) * sizeof(Value);
  const std::uintmax_t file_bytes = std::filesystem::file_size(path);
  if (file_bytes % record_bytes != 0)
    throw std::runtime_error("xvec file size is not a whole number of records");
  const std::size_t rows = static_cast<std::size_t>(file_bytes / record_bytes);
  input.seekg(0);
  Matrix<Value> output{rows, dimension, {}};
  output.values.resize(rows * static_cast<std::size_t>(dimension));
  for (std::size_t row = 0; row < rows; ++row) {
    std::int32_t row_dimension = 0;
    input.read(reinterpret_cast<char *>(&row_dimension), sizeof(row_dimension));
    if (!input || row_dimension != dimension)
      throw std::runtime_error("inconsistent xvec record dimension");
    input.read(reinterpret_cast<char *>(output.values.data() + row * dimension),
               static_cast<std::streamsize>(dimension * sizeof(Value)));
    if (!input) throw std::runtime_error("truncated xvec record");
  }
  if (!floating_point) {
    for (const Value value : output.values) {
      std::int32_t signed_value = 0;
      std::memcpy(&signed_value, &value, sizeof(value));
      if (signed_value < 0)
        throw std::runtime_error("negative ivecs ID");
    }
  }
  return output;
}

} // namespace

XvecDataset load_xvec_dataset(const std::string &base_path,
                              const std::string &query_path,
                              const std::string &ground_truth_path) {
  const auto base = read_xvecs<float>(base_path, true);
  const auto queries = read_xvecs<float>(query_path, true);
  const auto source_truth = read_xvecs<std::int32_t>(ground_truth_path, false);
  if (base.dimension != queries.dimension)
    throw std::runtime_error("base/query dimensions differ");
  if (queries.rows != source_truth.rows)
    throw std::runtime_error("query/ground-truth row counts differ");
  XvecDataset output{base.rows, queries.rows, base.dimension,
                     static_cast<std::size_t>(source_truth.dimension),
                     std::move(base.values), std::move(queries.values), {}};
  output.ground_truth_ids.reserve(source_truth.values.size());
  for (const std::int32_t id : source_truth.values) {
    if (id < 0 || static_cast<std::size_t>(id) >= output.base_count)
      throw std::runtime_error("ground-truth ID outside base range");
    output.ground_truth_ids.push_back(static_cast<faiss::idx_t>(id));
  }
  return output;
}

std::vector<faiss::idx_t> first_ground_truth_neighbors(
    const XvecDataset &dataset, std::size_t k) {
  if (k == 0 || k > dataset.ground_truth_width)
    throw std::invalid_argument("invalid requested ground-truth width");
  std::vector<faiss::idx_t> output;
  output.reserve(dataset.query_count * k);
  for (std::size_t query = 0; query < dataset.query_count; ++query) {
    const auto begin = dataset.ground_truth_ids.begin() +
                       query * dataset.ground_truth_width;
    output.insert(output.end(), begin, begin + k);
  }
  return output;
}

} // namespace quant_hardness
