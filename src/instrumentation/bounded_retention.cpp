#include "instrumentation/bounded_retention.h"
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/impl/VisitedTable.h>
#include <faiss/utils/distances.h>
#include <chrono>
#include <memory>
#include <numeric>
#include <limits>

namespace quant_hardness {
namespace {
using Clock=std::chrono::steady_clock;
double micros(Clock::time_point a,Clock::time_point b) {return std::chrono::duration<double,std::micro>(b-a).count();}
struct FullCollector {
    std::vector<uint32_t>& tags; uint32_t epoch;
    std::vector<int32_t>& ids; std::vector<float>& scores;
    void add(int32_t id,float s) {if(tags[id]!=epoch){tags[id]=epoch;ids.push_back(id);scores.push_back(s);}}
};
template<class Collector,bool Detail> class Capture final:public faiss::DistanceComputer {
 public:
    Capture(faiss::DistanceComputer& d,Collector& c,double& t):dc(d),collector(c),retention(t){}
    void set_query(const float* q) override {dc.set_query(q);}
    void seed(int32_t id,float s) {
        if constexpr(Detail) {auto start=Clock::now();collector.add(id,s);retention+=micros(start,Clock::now());}
        else collector.add(id,s);
    }
    float operator()(faiss::idx_t id) override {float s=dc(id);seed(id,s);return s;}
    void distances_batch_4(faiss::idx_t a,faiss::idx_t b,faiss::idx_t c,faiss::idx_t d,float& x,float& y,float& z,float& w) override {
        dc.distances_batch_4(a,b,c,d,x,y,z,w);
        if constexpr(Detail) {auto start=Clock::now();collector.add(a,x);collector.add(b,y);collector.add(c,z);collector.add(d,w);retention+=micros(start,Clock::now());}
        else {collector.add(a,x);collector.add(b,y);collector.add(c,z);collector.add(d,w);}
    }
    float symmetric_dis(faiss::idx_t a,faiss::idx_t b) override {return dc.symmetric_dis(a,b);}
 private:
    faiss::DistanceComputer& dc;Collector& collector;double& retention;
};
}
CandidateAccess::CandidateAccess(faiss::IndexHNSW& g,faiss::IndexPQ& p,const float* b,double t,int ef,bool full):graph_(g),pq_(p),base_(b),threshold_(t),ef_(ef) {
    if(g.metric_type!=faiss::METRIC_L2 || g.hnsw.is_panorama || g.hnsw.is_similarity || g.ntotal!=p.ntotal || g.d!=p.d || g.ntotal>INT32_MAX || (ef!=32 && ef!=64 && ef!=128)) throw std::runtime_error("unsupported frozen search");
    if(full) {tags_.resize(g.ntotal);ids_.reserve(2048);scores_.reserve(2048);order_.reserve(2048);}
}
size_t CandidateAccess::full_scratch_bytes() const {return 4*(tags_.capacity()+ids_.capacity()+scores_.capacity()+order_.capacity());}
AccessOutput CandidateAccess::run(const float* q,AccessMode m,int clock) {
    return clock==2?execute<true>(q,m,clock):execute<false>(q,m,clock);
}
template<bool Detail> AccessOutput CandidateAccess::execute(const float* q,AccessMode mode,int clock) {
    auto start=Clock::now(),t1=start,t2=start,t3=start;AccessOutput output;
    auto& out=output.result;out.gap=std::numeric_limits<double>::quiet_NaN();
    {
        bool full=mode==AccessMode::Full||mode==AccessMode::FullAll;
        bool retain=mode!=AccessMode::Native;
        BoundedTop16 bounded;
        if(full) {
            if(tags_.empty())throw std::runtime_error("full storage not allocated");
            ids_.clear();scores_.clear();
            if(++epoch_==0){std::fill(tags_.begin(),tags_.end(),0);++epoch_;}
        }
        std::unique_ptr<faiss::DistanceComputer> dc(pq_.get_distance_computer());
        auto& vt=faiss::VisitedTable::get_reusable(graph_.ntotal,graph_.hnsw.use_visited_hashset);
        faiss::SearchParametersHNSW params;params.efSearch=ef_;
        using RH=faiss::HeapBlockResultHandler<faiss::HNSW::C_distance>;
        RH block(1,out.native_scores.data(),out.native_ids.data(),10);RH::SingleResultHandler res(block);res.begin(0);dc->set_query(q);
        if(!retain) graph_.hnsw.search(*dc,&graph_,res,vt,&params);
        else {
            auto nearest=graph_.hnsw.entry_point;float distance=(*dc)(nearest);
            for(int level=graph_.hnsw.max_level;level>=1;--level)faiss::hnsw_detail::greedy_update_nearest(graph_.hnsw,*dc,level,nearest,distance);
            faiss::HNSWStats stats;
            if(full) {
                FullCollector col{tags_,epoch_,ids_,scores_};Capture<FullCollector,Detail> cap(*dc,col,output.retention_us);cap.seed(nearest,distance);
                graph_.hnsw.search_level_0(cap,res,1,&nearest,&distance,1,stats,vt,&params);
            } else {
                Capture<BoundedTop16,Detail> cap(*dc,bounded,output.retention_us);cap.seed(nearest,distance);
                graph_.hnsw.search_level_0(cap,res,1,&nearest,&distance,1,stats,vt,&params);
            }
            vt.advance();
        }
        vt.advance();
        if(clock)t1=Clock::now();
        if(retain) {
            std::array<RetainedCandidate,16> top;
            if(full) {
                if(ids_.size()<16)throw std::runtime_error("short full pool");
                order_.resize(ids_.size());std::iota(order_.begin(),order_.end(),0);
                std::partial_sort(order_.begin(),order_.begin()+16,order_.end(),[&](uint32_t a,uint32_t b){return scores_[a]<scores_[b]||(scores_[a]==scores_[b]&&a<b);});
                for(int i=0;i<16;++i){auto j=order_[i];top[i]={scores_[j],ids_[j],j};}
                output.candidates=ids_.size();
            } else {top=bounded.sorted();output.candidates=bounded.calls();}
            for(int i=0;i<16;++i){out.top_ids[i]=top[i].id;out.top_order[i]=top[i].order;output.top_scores[i]=top[i].score;}
            if(mode==AccessMode::BoundedGap) {out.gap=double(top[11].score)-double(top[8].score);out.refined=out.gap<=threshold_;}
            else out.refined=mode==AccessMode::FullAll||mode==AccessMode::BoundedAll;
        }
        if(clock)t2=Clock::now();
        if(out.refined)for(int j=0;j<16;++j)out.exact[j]=faiss::fvec_L2sqr(q,base_+out.top_ids[j]*graph_.d,graph_.d);
        if(clock)t3=Clock::now();
        res.end();out.ids=out.native_ids;if(out.refined)StreamingRefinement::final_rerank(out);
    }
    auto end=Clock::now();out.total_us=micros(start,end);
    if(clock){out.search_us=micros(start,t1);out.gap_us=micros(t1,t2);out.refine_us=micros(t2,t3);out.final_us=micros(t3,end);}
    return output;
}
}
