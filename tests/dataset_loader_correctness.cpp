#include "datasets/xvecs.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}

template <typename Value>
void write_xvecs(const std::filesystem::path &path,
                 const std::vector<std::vector<Value>> &rows) {
  std::ofstream output(path, std::ios::binary);
  const std::int32_t dimension = static_cast<std::int32_t>(rows.at(0).size());
  for (const auto &row : rows) {
    require(static_cast<std::int32_t>(row.size()) == dimension,
            "test row dimensions differ");
    output.write(reinterpret_cast<const char *>(&dimension), sizeof(dimension));
    output.write(reinterpret_cast<const char *>(row.data()),
                 static_cast<std::streamsize>(row.size() * sizeof(Value)));
  }
}

} // namespace

int main() {
  const auto root = std::filesystem::temp_directory_path() /
                    "quant_hardness_xvec_loader_test";
  try {
    std::filesystem::create_directory(root);
    write_xvecs<float>(root / "base.fvecs", {{0, 1}, {2, 3}, {4, 5}});
    write_xvecs<float>(root / "query.fvecs", {{6, 7}, {8, 9}});
    write_xvecs<std::int32_t>(root / "truth.ivecs", {{2, 1, 0}, {0, 1, 2}});
    const auto loaded = quant_hardness::load_xvec_dataset(
        (root / "base.fvecs").string(), (root / "query.fvecs").string(),
        (root / "truth.ivecs").string());
    require(loaded.base_count == 3 && loaded.query_count == 2 &&
                loaded.dimension == 2 && loaded.ground_truth_width == 3,
            "loaded dimensions are wrong");
    require(loaded.base == std::vector<float>({0, 1, 2, 3, 4, 5}) &&
                loaded.queries == std::vector<float>({6, 7, 8, 9}),
            "loaded vectors are wrong");
    require(quant_hardness::first_ground_truth_neighbors(loaded, 2) ==
                std::vector<faiss::idx_t>({2, 1, 0, 1}),
            "ground-truth slicing is wrong");
    std::filesystem::remove_all(root);
    std::cout << "status=PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::filesystem::remove_all(root);
    std::cerr << "status=FAIL\nerror=" << error.what() << '\n';
    return 1;
  }
}
