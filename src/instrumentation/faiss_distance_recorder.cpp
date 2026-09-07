#include "instrumentation/faiss_distance_recorder.h"

#include <faiss/impl/DistanceComputer.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>

namespace quant_hardness {
namespace {

class RecordingDistanceComputer final : public faiss::DistanceComputer {
public:
  RecordingDistanceComputer(faiss::DistanceComputer *wrapped,
                            const RecordingIndex &recorder)
      : wrapped_(wrapped), recorder_(recorder) {
    if (wrapped == nullptr) {
      throw std::invalid_argument("wrapped distance computer is null");
    }
  }

  void set_query(const float *query) override {
    query_ = query;
    wrapped_->set_query(query);
  }

  float operator()(faiss::idx_t database_id) override {
    require_query();
    recorder_.record(query_, database_id);
    return (*wrapped_)(database_id);
  }

  void distances_batch_4(const faiss::idx_t first, const faiss::idx_t second,
                         const faiss::idx_t third, const faiss::idx_t fourth,
                         float &first_distance, float &second_distance,
                         float &third_distance,
                         float &fourth_distance) override {
    require_query();
    recorder_.record(query_, first);
    recorder_.record(query_, second);
    recorder_.record(query_, third);
    recorder_.record(query_, fourth);
    wrapped_->distances_batch_4(first, second, third, fourth, first_distance,
                                second_distance, third_distance,
                                fourth_distance);
  }

  float symmetric_dis(faiss::idx_t first, faiss::idx_t second) override {
    // This is database-to-database distance, not a query distance evaluation.
    return wrapped_->symmetric_dis(first, second);
  }

private:
  void require_query() const {
    if (query_ == nullptr) {
      throw std::logic_error("distance evaluated before set_query");
    }
  }

  std::unique_ptr<faiss::DistanceComputer> wrapped_;
  const RecordingIndex &recorder_;
  const float *query_ = nullptr;
};

} // namespace

RecordingIndex::RecordingIndex(faiss::Index &wrapped_storage,
                               const float *query_base,
                               faiss::idx_t query_count)
    : faiss::Index(wrapped_storage.d, wrapped_storage.metric_type),
      wrapped_storage_(wrapped_storage), query_base_(query_base),
      query_count_(query_count),
      evaluated_ids_(static_cast<std::size_t>(query_count)) {
  if (query_base == nullptr || query_count <= 0) {
    throw std::invalid_argument("invalid query array for recording index");
  }
  ntotal = wrapped_storage.ntotal;
  is_trained = wrapped_storage.is_trained;
  metric_arg = wrapped_storage.metric_arg;
}

void RecordingIndex::add(faiss::idx_t, const float *) {
  throw std::logic_error("RecordingIndex cannot add vectors");
}

void RecordingIndex::search(faiss::idx_t, const float *, faiss::idx_t, float *,
                            faiss::idx_t *,
                            const faiss::SearchParameters *) const {
  throw std::logic_error("RecordingIndex is only a distance-computer facade");
}

void RecordingIndex::reset() {
  throw std::logic_error("RecordingIndex cannot reset storage");
}

faiss::DistanceComputer *RecordingIndex::get_distance_computer() const {
  return new RecordingDistanceComputer(wrapped_storage_.get_distance_computer(),
                                       *this);
}

void RecordingIndex::record(const float *query,
                            faiss::idx_t database_id) const {
  if (database_id < 0 || database_id >= ntotal) {
    throw std::out_of_range("recorded database ID is outside storage");
  }
  const std::uintptr_t base = reinterpret_cast<std::uintptr_t>(query_base_);
  const std::uintptr_t current = reinterpret_cast<std::uintptr_t>(query);
  const std::size_t query_bytes = static_cast<std::size_t>(d) * sizeof(float);
  if (current < base || (current - base) % query_bytes != 0) {
    throw std::invalid_argument("set_query pointer is not query-array aligned");
  }
  const std::size_t query_id = (current - base) / query_bytes;
  if (query_id >= static_cast<std::size_t>(query_count_)) {
    throw std::out_of_range("set_query pointer is outside query array");
  }
  evaluated_ids_[query_id].insert(database_id);
}

std::vector<faiss::idx_t>
RecordingIndex::sorted_evaluated_ids(faiss::idx_t query_id) const {
  if (query_id < 0 || query_id >= query_count_) {
    throw std::out_of_range("query ID is outside recorder");
  }
  const auto &ids = evaluated_ids_[static_cast<std::size_t>(query_id)];
  std::vector<faiss::idx_t> sorted(ids.begin(), ids.end());
  std::sort(sorted.begin(), sorted.end());
  return sorted;
}

std::size_t
RecordingIndex::unique_evaluation_count(faiss::idx_t query_id) const {
  if (query_id < 0 || query_id >= query_count_) {
    throw std::out_of_range("query ID is outside recorder");
  }
  return evaluated_ids_[static_cast<std::size_t>(query_id)].size();
}

} // namespace quant_hardness
