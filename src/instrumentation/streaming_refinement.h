#pragma once
#include <faiss/IndexHNSW.h>
#include <faiss/IndexPQ.h>
#include <array>
#include <cstdint>
#include <vector>

namespace quant_hardness {
enum class StreamPolicy { Native=0, Gap=1, Random=2, All=3 };
struct StreamOutput {
  std::array<faiss::idx_t,10> ids{},native_ids{};
  std::array<float,10> native_scores{};
  std::array<faiss::idx_t,16> top_ids{};
  std::array<uint32_t,16> top_order{};
  std::array<float,16> exact{};
  double gap=0,total_us=0,search_us=0,gap_us=0,refine_us=0,final_us=0;
  bool refined=false;
};

// Fixed-k10, ef64 adapter. No graph algorithm implementation is copied.
class StreamingRefinement {
 public:
  StreamingRefinement(faiss::IndexHNSW&,faiss::IndexPQ&,const float*,double,int);
  StreamOutput run(const float*,StreamPolicy,bool random_refine,bool components,bool defer=false);
  static void final_rerank(StreamOutput&);
  std::vector<int32_t> candidate_ids;
  std::vector<float> candidate_scores;
  size_t scratch_bytes() const;
 private:
  faiss::IndexHNSW& graph_;
  faiss::IndexPQ& pq_;
  const float* base_;
  double threshold_;
  int ef_;
  std::vector<uint32_t> tags_,order_;
  uint32_t epoch_=0;
};
}
