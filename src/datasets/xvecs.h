#pragma once

#include <faiss/Index.h>

#include <cstddef>
#include <string>
#include <vector>

namespace quant_hardness {

struct XvecDataset {
  std::size_t base_count;
  std::size_t query_count;
  int dimension;
  std::size_t ground_truth_width;
  std::vector<float> base;
  std::vector<float> queries;
  std::vector<faiss::idx_t> ground_truth_ids;
};

// Strictly loads the standard little-endian fvecs/ivecs record format. Every
// record repeats its dimension; inconsistent or truncated files are rejected.
XvecDataset load_xvec_dataset(const std::string &base_path,
                              const std::string &query_path,
                              const std::string &ground_truth_path);

std::vector<faiss::idx_t> first_ground_truth_neighbors(
    const XvecDataset &dataset, std::size_t k);

} // namespace quant_hardness
