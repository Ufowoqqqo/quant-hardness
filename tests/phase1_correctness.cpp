#include "graph/faiss_shared_hnsw.h"
#include "metrics/ground_truth.h"
#include "metrics/recall.h"

#include <faiss/IndexFlat.h>
#include <omp.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void test_exhaustive_ground_truth() {
  const std::vector<float> base{0.0F, 3.0F, 10.0F, 20.0F};
  const std::vector<float> queries{2.0F};
  const auto truth = quant_hardness::exhaustive_l2_top_k(
      base.data(), 4, queries.data(), 1, 1, 3);
  require(truth.ids == std::vector<faiss::idx_t>({1, 0, 2}),
          "exhaustive ground-truth IDs are wrong");
  require(truth.distances == std::vector<float>({1.0F, 4.0F, 64.0F}),
          "exhaustive ground-truth distances are wrong");
}

void test_recall_at_10() {
  const std::vector<faiss::idx_t> truth{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  const std::vector<faiss::idx_t> perfect{9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
  const std::vector<faiss::idx_t> partial{0, 1, 2, 3, 4, 10, 11, 12, 13, 14};
  const std::vector<faiss::idx_t> disjoint{10, 11, 12, 13, 14,
                                           15, 16, 17, 18, 19};
  require(quant_hardness::recall_at_k(perfect, truth, 10) == 1.0,
          "perfect Recall@10 is wrong");
  require(quant_hardness::recall_at_k(partial, truth, 10) == 0.5,
          "partial Recall@10 is wrong");
  require(quant_hardness::recall_at_k(disjoint, truth, 10) == 0.0,
          "zero Recall@10 is wrong");

  bool rejected_duplicate = false;
  try {
    std::vector<faiss::idx_t> duplicate = perfect;
    duplicate[1] = duplicate[0];
    (void)quant_hardness::recall_at_k(duplicate, truth, 10);
  } catch (const std::invalid_argument &) {
    rejected_duplicate = true;
  }
  require(rejected_duplicate, "duplicate result IDs were not rejected");
}

void test_delta_sign() {
  require(std::abs(quant_hardness::delta_recall(0.8, 0.5) - 0.3) < 1e-12,
          "positive delta sign is wrong");
  require(std::abs(quant_hardness::delta_recall(0.4, 0.7) + 0.3) < 1e-12,
          "negative delta sign is wrong");
}

void test_query_order() {
  const std::vector<std::uint64_t> exact{0, 1, 2, 3};
  const std::vector<std::uint64_t> same{0, 1, 2, 3};
  const std::vector<std::uint64_t> changed{0, 2, 1, 3};
  quant_hardness::require_identical_query_order(exact, same);

  bool rejected_change = false;
  try {
    quant_hardness::require_identical_query_order(exact, changed);
  } catch (const std::invalid_argument &) {
    rejected_change = true;
  }
  require(rejected_change, "changed paired query order was not rejected");
}

void test_identical_graph_fingerprint() {
  constexpr int dimension = 4;
  const std::vector<float> base{0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
                                0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1};
  faiss::IndexFlatL2 storage(dimension);
  faiss::IndexHNSW graph(&storage, 2);
  graph.hnsw.rng = faiss::RandomGenerator(7);
  graph.train(5, base.data());
  graph.add(5, base.data());
  const std::string first = quant_hardness::graph_fingerprint(graph.hnsw);
  const std::string second = quant_hardness::graph_fingerprint(graph.hnsw);
  require(first == second, "unchanged graph fingerprints differ");
  const auto saved = graph.hnsw.entry_point;
  graph.hnsw.entry_point = saved + 1;
  require(quant_hardness::graph_fingerprint(graph.hnsw) != first,
          "structural mutation did not change fingerprint");
  graph.hnsw.entry_point = saved;
  require(quant_hardness::graph_fingerprint(graph.hnsw) == first,
          "restored graph fingerprint differs");
}

} // namespace

int main() {
  try {
    omp_set_num_threads(1);
    test_exhaustive_ground_truth();
    test_recall_at_10();
    test_delta_sign();
    test_query_order();
    test_identical_graph_fingerprint();
    std::cout << "status=PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "status=FAIL\nerror=" << error.what() << '\n';
    return 1;
  }
}
