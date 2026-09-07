#include "instrumentation/faiss_level0_recorder.h"

#include "instrumentation/faiss_distance_recorder.h"

#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/HNSW.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>

namespace quant_hardness {
namespace {

class StorageGuard {
public:
  StorageGuard(faiss::IndexHNSW &graph, faiss::Index &storage)
      : graph_(graph), original_(graph.storage) {
    if (original_ == nullptr || storage.d != graph.d ||
        storage.ntotal != graph.ntotal ||
        storage.metric_type != graph.metric_type) {
      throw std::invalid_argument("invalid level-0 traversal storage");
    }
    graph_.storage = &storage;
  }
  StorageGuard(const StorageGuard &) = delete;
  StorageGuard &operator=(const StorageGuard &) = delete;
  ~StorageGuard() { graph_.storage = original_; }

private:
  faiss::IndexHNSW &graph_;
  faiss::Index *original_;
};

std::vector<faiss::idx_t>
set_difference(const std::vector<faiss::idx_t> &first,
               const std::vector<faiss::idx_t> &second) {
  std::vector<faiss::idx_t> output;
  std::set_difference(first.begin(), first.end(), second.begin(), second.end(),
                      std::back_inserter(output));
  return output;
}

} // namespace

PhaseSeparatedSearchResults search_with_level0_recording(
    faiss::IndexHNSW &graph, faiss::Index &traversal_storage,
    const float *queries, faiss::idx_t query_count, faiss::idx_t k,
    const faiss::SearchParametersHNSW &parameters,
    bool validate_native_identity) {
  if (queries == nullptr || query_count <= 0 || k <= 0 ||
      graph.hnsw.entry_point < 0 || graph.hnsw.is_similarity ||
      graph.metric_type != faiss::METRIC_L2 || graph.hnsw.is_panorama) {
    throw std::invalid_argument(
        "level-0 recorder requires non-empty vanilla squared-L2 HNSW");
  }
  const std::string fingerprint = graph_fingerprint(graph.hnsw);
  SearchResults reference;
  if (validate_native_identity) {
    reference = search_with_storage(graph, traversal_storage, queries,
                                    query_count, k, parameters);
  }

  RecordingIndex upper_recording(traversal_storage, queries, query_count);
  std::unique_ptr<faiss::DistanceComputer> upper_distance(
      upper_recording.get_distance_computer());
  std::vector<faiss::HNSW::storage_idx_t> level0_entries(query_count);
  std::vector<float> level0_entry_distances(query_count);
  for (faiss::idx_t query_id = 0; query_id < query_count; ++query_id) {
    upper_distance->set_query(queries + query_id * graph.d);
    faiss::HNSW::storage_idx_t nearest = graph.hnsw.entry_point;
    float nearest_distance = (*upper_distance)(nearest);
    for (int level = graph.hnsw.max_level; level >= 1; --level) {
      faiss::hnsw_detail::greedy_update_nearest(
          graph.hnsw, *upper_distance, level, nearest, nearest_distance);
    }
    level0_entries[query_id] = nearest;
    level0_entry_distances[query_id] = nearest_distance;
  }

  RecordingIndex level0_recording(traversal_storage, queries, query_count);
  std::unique_ptr<faiss::DistanceComputer> seed_distance(
      level0_recording.get_distance_computer());
  for (faiss::idx_t query_id = 0; query_id < query_count; ++query_id) {
    seed_distance->set_query(queries + query_id * graph.d);
    const float repeated = (*seed_distance)(level0_entries[query_id]);
    if (repeated != level0_entry_distances[query_id]) {
      throw std::runtime_error(
          "level-0 entry distance changed during phase attribution");
    }
  }

  PhaseSeparatedSearchResults output;
  output.native.distances.resize(static_cast<std::size_t>(query_count * k));
  output.native.ids.resize(static_cast<std::size_t>(query_count * k));
  faiss::Index *const original_storage = graph.storage;
  {
    StorageGuard guard(graph, level0_recording);
    graph.search_level_0(
        query_count, queries, k, level0_entries.data(),
        level0_entry_distances.data(), output.native.distances.data(),
        output.native.ids.data(), 1, 1, &parameters);
  }
  if (graph.storage != original_storage ||
      graph_fingerprint(graph.hnsw) != fingerprint) {
    throw std::runtime_error("phase-separated search changed HNSW state");
  }
  if (validate_native_identity &&
      (reference.ids != output.native.ids ||
       reference.distances != output.native.distances)) {
    throw std::runtime_error(
        "phase-separated replay changed native search results");
  }

  output.upper_only_evaluated_ids.reserve(query_count);
  output.level0_evaluated_ids.reserve(query_count);
  for (faiss::idx_t query_id = 0; query_id < query_count; ++query_id) {
    std::vector<faiss::idx_t> level0 =
        level0_recording.sorted_evaluated_ids(query_id);
    const std::vector<faiss::idx_t> upper =
        upper_recording.sorted_evaluated_ids(query_id);
    output.upper_only_evaluated_ids.push_back(set_difference(upper, level0));
    output.level0_evaluated_ids.push_back(std::move(level0));
  }
  return output;
}

} // namespace quant_hardness
