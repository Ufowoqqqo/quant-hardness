#include "graph/faiss_shared_hnsw.h"

#include <openssl/evp.h>

#include <array>
#include <cstdint>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <vector>

namespace quant_hardness {
namespace {

class CanonicalBytes {
public:
  void text(std::string_view value) {
    unsigned_integer(static_cast<std::uint64_t>(value.size()));
    bytes_.insert(bytes_.end(), value.begin(), value.end());
  }

  template <typename Integer> void integer(Integer value) {
    static_assert(std::is_integral_v<Integer>);
    if constexpr (std::is_signed_v<Integer>) {
      unsigned_integer(
          static_cast<std::uint64_t>(static_cast<std::int64_t>(value)));
    } else {
      unsigned_integer(static_cast<std::uint64_t>(value));
    }
  }

  template <typename Container>
  void integer_vector(std::string_view name, const Container &values) {
    text(name);
    integer(values.size());
    for (const auto value : values) {
      integer(value);
    }
  }

  template <typename Value>
  void integer_buffer(std::string_view name, const Value *values,
                      std::size_t size) {
    text(name);
    integer(size);
    for (std::size_t i = 0; i < size; ++i) {
      integer(values[i]);
    }
  }

  const std::vector<std::uint8_t> &bytes() const { return bytes_; }

private:
  void unsigned_integer(std::uint64_t value) {
    for (unsigned int shift = 0; shift < 64; shift += 8) {
      bytes_.push_back(static_cast<std::uint8_t>(value >> shift));
    }
  }

  std::vector<std::uint8_t> bytes_;
};

class TraversalStorageGuard {
public:
  TraversalStorageGuard(faiss::IndexHNSW &graph_index,
                        faiss::Index &traversal_storage)
      : graph_index_(graph_index), original_(graph_index.storage) {
    if (original_ == nullptr) {
      throw std::invalid_argument("HNSW graph has no storage");
    }
    if (!traversal_storage.is_trained) {
      throw std::invalid_argument("traversal storage is not trained");
    }
    if (graph_index_.metric_type != faiss::METRIC_L2 ||
        graph_index_.hnsw.is_similarity || graph_index_.hnsw.is_panorama) {
      throw std::invalid_argument(
          "Phase 1 adapter supports only vanilla squared-L2 HNSW");
    }
    if (traversal_storage.d != graph_index_.d ||
        traversal_storage.metric_type != graph_index_.metric_type ||
        traversal_storage.ntotal != graph_index_.ntotal) {
      throw std::invalid_argument(
          "traversal storage must match graph dimension, metric, and ntotal");
    }
    graph_index_.storage = &traversal_storage;
  }

  TraversalStorageGuard(const TraversalStorageGuard &) = delete;
  TraversalStorageGuard &operator=(const TraversalStorageGuard &) = delete;

  ~TraversalStorageGuard() { graph_index_.storage = original_; }

private:
  faiss::IndexHNSW &graph_index_;
  faiss::Index *original_;
};

std::string sha256_hex(const std::vector<std::uint8_t> &bytes) {
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(
      EVP_MD_CTX_new(), EVP_MD_CTX_free);
  if (!context ||
      EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1 ||
      EVP_DigestUpdate(context.get(), bytes.data(), bytes.size()) != 1) {
    throw std::runtime_error("failed to initialize graph SHA-256");
  }

  std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
  unsigned int digest_size = 0;
  if (EVP_DigestFinal_ex(context.get(), digest.data(), &digest_size) != 1) {
    throw std::runtime_error("failed to finalize graph SHA-256");
  }

  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (unsigned int i = 0; i < digest_size; ++i) {
    output << std::setw(2) << static_cast<unsigned int>(digest[i]);
  }
  return output.str();
}

} // namespace

std::string graph_fingerprint(const faiss::HNSW &hnsw) {
  CanonicalBytes encoded;
  encoded.text("faiss-hnsw-struct-v1");
  encoded.text("entry_point");
  encoded.integer(hnsw.entry_point);
  encoded.text("max_level");
  encoded.integer(hnsw.max_level);
  encoded.integer_vector("levels", hnsw.levels);
  encoded.integer_vector("offsets", hnsw.offsets);
  encoded.integer_buffer("neighbors", hnsw.neighbors.data(),
                         hnsw.neighbors.size());
  encoded.integer_vector("cum_nneighbor_per_level",
                         hnsw.cum_nneighbor_per_level);
  return sha256_hex(encoded.bytes());
}

SearchResults search_with_storage(faiss::IndexHNSW &graph_index,
                                  faiss::Index &traversal_storage,
                                  const float *queries,
                                  faiss::idx_t query_count, faiss::idx_t k,
                                  const faiss::SearchParametersHNSW &params) {
  if (queries == nullptr || query_count <= 0 || k <= 0) {
    throw std::invalid_argument("queries, query_count, and k must be valid");
  }

  const std::string before = graph_fingerprint(graph_index.hnsw);
  faiss::Index *const original_storage = graph_index.storage;

  SearchResults results;
  results.distances.resize(static_cast<std::size_t>(query_count * k));
  results.ids.resize(static_cast<std::size_t>(query_count * k));
  {
    TraversalStorageGuard guard(graph_index, traversal_storage);
    if (graph_fingerprint(graph_index.hnsw) != before) {
      throw std::runtime_error(
          "selecting traversal storage changed HNSW topology");
    }
    graph_index.search(query_count, queries, k, results.distances.data(),
                       results.ids.data(), &params);
    if (graph_fingerprint(graph_index.hnsw) != before) {
      throw std::runtime_error("HNSW topology changed during search");
    }
  }

  if (graph_index.storage != original_storage ||
      graph_fingerprint(graph_index.hnsw) != before) {
    throw std::runtime_error("HNSW state was not restored after search");
  }
  return results;
}

} // namespace quant_hardness
