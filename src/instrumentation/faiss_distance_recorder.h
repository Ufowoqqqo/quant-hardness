#pragma once

#include <faiss/Index.h>

#include <cstddef>
#include <cstdint>
#include <unordered_set>
#include <vector>

namespace quant_hardness {

// Index facade that delegates every distance calculation to wrapped_storage
// while recording the unique database IDs passed to query-to-node distance
// calls. It is research-only and restricted to one search thread.
class RecordingIndex final : public faiss::Index {
public:
  RecordingIndex(faiss::Index &wrapped_storage, const float *query_base,
                 faiss::idx_t query_count);

  void add(faiss::idx_t count, const float *vectors) override;
  void
  search(faiss::idx_t query_count, const float *queries, faiss::idx_t k,
         float *distances, faiss::idx_t *labels,
         const faiss::SearchParameters *parameters = nullptr) const override;
  void reset() override;
  faiss::DistanceComputer *get_distance_computer() const override;

  std::vector<faiss::idx_t> sorted_evaluated_ids(faiss::idx_t query_id) const;
  std::size_t unique_evaluation_count(faiss::idx_t query_id) const;

  void record(const float *query, faiss::idx_t database_id) const;

private:
  faiss::Index &wrapped_storage_;
  const float *query_base_;
  faiss::idx_t query_count_;
  mutable std::vector<std::unordered_set<faiss::idx_t>> evaluated_ids_;
};

} // namespace quant_hardness
