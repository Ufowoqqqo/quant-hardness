#pragma once
#include "instrumentation/streaming_refinement.h"
#include <algorithm>
#include <stdexcept>

namespace quant_hardness {
struct RetainedCandidate {
    float score;
    int32_t id;
    uint32_t order;
};
inline bool candidate_better(const RetainedCandidate& a,const RetainedCandidate& b) {
    return a.score < b.score || (a.score == b.score && a.order < b.order);
}
// Deterministic per-ID scores are required. Non-finite ADC is outside this
// frozen L2/PQ experiment. No full seen-ID set or per-insertion allocations.
class BoundedTop16 {
 public:
    void reset() {size_=0;calls_=0;}
    void add(int32_t id,float score) {
        RetainedCandidate x{score,id,calls_++};
        if(size_==16 && !candidate_better(x,heap_[0])) return;
        for(size_t i=0;i<size_;++i) if(heap_[i].id==id) return;
        if(size_<16) {
            heap_[size_++]=x;
            std::push_heap(heap_.begin(),heap_.begin()+size_,candidate_better);
        } else {
            std::pop_heap(heap_.begin(),heap_.end(),candidate_better);
            heap_[15]=x;
            std::push_heap(heap_.begin(),heap_.end(),candidate_better);
        }
    }
    std::array<RetainedCandidate,16> sorted() const {
        if(size_!=16) throw std::runtime_error("frozen L16 requires16 unique candidates");
        auto result=heap_;
        std::sort(result.begin(),result.end(),candidate_better);
        return result;
    }
    size_t size() const {return size_;}
    uint32_t calls() const {return calls_;}
 private:
    std::array<RetainedCandidate,16> heap_{};
    size_t size_=0;
    uint32_t calls_=0;
};

enum class AccessMode {Native=0,Full=1,Bounded=2,FullAll=3,BoundedAll=4,BoundedGap=5};
inline constexpr const char* access_names[]={"NATIVE","FULL","BOUNDED","FULL_ALL16","BOUNDED_ALL16","GAP_BOUNDED"};
struct AccessOutput {
    StreamOutput result;
    std::array<float,16> top_scores{};
    size_t candidates=0;
    double retention_us=0;
};
class CandidateAccess {
 public:
    CandidateAccess(faiss::IndexHNSW&,faiss::IndexPQ&,const float*,double,int,bool full_storage);
    // clock_level0: outer total only;1: stage clocks;2: intrusive retention
    // callback clocks, for diagnostics only.
    AccessOutput run(const float*,AccessMode,int clock_level);
    size_t full_scratch_bytes() const;
    const std::vector<int32_t>& ids() const {return ids_;}
    const std::vector<float>& scores() const {return scores_;}
 private:
    template<bool Detail> AccessOutput execute(const float*,AccessMode,int);
    faiss::IndexHNSW& graph_;
    faiss::IndexPQ& pq_;
    const float* base_;
    double threshold_;
    int ef_;
    std::vector<uint32_t> tags_,order_;
    std::vector<int32_t> ids_;
    std::vector<float> scores_;
    uint32_t epoch_=0;
};
}
