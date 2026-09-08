#include "instrumentation/streaming_refinement.h"
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/impl/VisitedTable.h>
#include <faiss/utils/distances.h>
#include <algorithm>
#include <chrono>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>

namespace quant_hardness {
namespace {
using Clock=std::chrono::steady_clock;
double us(Clock::time_point a,Clock::time_point b){return std::chrono::duration<double,std::micro>(b-a).count();}
class Capture final: public faiss::DistanceComputer {
 public:
  Capture(faiss::DistanceComputer& d,std::vector<uint32_t>& t,uint32_t e,std::vector<int32_t>& i,std::vector<float>& s):dc(d),tags(t),epoch(e),ids(i),scores(s){}
  void set_query(const float* q) override{dc.set_query(q);}
  void add(faiss::idx_t i,float s){if(tags[i]!=epoch){tags[i]=epoch;ids.push_back(i);scores.push_back(s);}}
  float operator()(faiss::idx_t i) override{float s=dc(i);add(i,s);return s;}
  void distances_batch_4(faiss::idx_t a,faiss::idx_t b,faiss::idx_t c,faiss::idx_t d,float& x,float& y,float& z,float& w) override{
    dc.distances_batch_4(a,b,c,d,x,y,z,w);add(a,x);add(b,y);add(c,z);add(d,w);
  }
  float symmetric_dis(faiss::idx_t a,faiss::idx_t b) override{return dc.symmetric_dis(a,b);}
 private:
  faiss::DistanceComputer& dc;std::vector<uint32_t>& tags;uint32_t epoch;
  std::vector<int32_t>& ids;std::vector<float>& scores;
};
}
StreamingRefinement::StreamingRefinement(faiss::IndexHNSW& g,faiss::IndexPQ& p,const float* b,double t,int ef):graph_(g),pq_(p),base_(b),threshold_(t),ef_(ef),tags_(g.ntotal,0){
  if(g.metric_type!=faiss::METRIC_L2||g.hnsw.is_panorama||g.hnsw.is_similarity||g.ntotal!=p.ntotal||g.d!=p.d||g.ntotal>INT32_MAX)throw std::runtime_error("unsupported streaming index");
  candidate_ids.reserve(2048);candidate_scores.reserve(2048);order_.reserve(2048);
}
size_t StreamingRefinement::scratch_bytes() const{return 4*(tags_.capacity()+candidate_ids.capacity()+candidate_scores.capacity()+order_.capacity());}
void StreamingRefinement::final_rerank(StreamOutput& out){
  std::array<int,16> ord;std::iota(ord.begin(),ord.end(),0);
  std::sort(ord.begin(),ord.end(),[&](int a,int b){return out.exact[a]<out.exact[b]||(out.exact[a]==out.exact[b]&&out.top_order[a]<out.top_order[b]);});
  for(int j=0;j<10;++j)out.ids[j]=out.top_ids[ord[j]];
}
StreamOutput StreamingRefinement::run(const float* q,StreamPolicy policy,bool random_refine,bool components,bool defer){
  const auto start=Clock::now();StreamOutput out;out.gap=std::numeric_limits<double>::quiet_NaN();
  auto t1=start,t2=start,t3=start;
  {
    const bool retain=policy==StreamPolicy::Gap||policy==StreamPolicy::All||(policy==StreamPolicy::Random&&random_refine);
    candidate_ids.clear();candidate_scores.clear();
    if(retain&&++epoch_==0){std::fill(tags_.begin(),tags_.end(),0);++epoch_;}
    std::unique_ptr<faiss::DistanceComputer> dc(pq_.get_distance_computer());
    auto& vt=faiss::VisitedTable::get_reusable(graph_.ntotal,graph_.hnsw.use_visited_hashset);
    faiss::SearchParametersHNSW params;params.efSearch=ef_;
    using RH=faiss::HeapBlockResultHandler<faiss::HNSW::C_distance>;
    RH block(1,out.native_scores.data(),out.native_ids.data(),10);RH::SingleResultHandler res(block);res.begin(0);dc->set_query(q);
    if(!retain){graph_.hnsw.search(*dc,&graph_,res,vt,&params);}
    else{
      auto nearest=graph_.hnsw.entry_point;float distance=(*dc)(nearest);
      for(int level=graph_.hnsw.max_level;level>=1;--level)faiss::hnsw_detail::greedy_update_nearest(graph_.hnsw,*dc,level,nearest,distance);
      Capture cap(*dc,tags_,epoch_,candidate_ids,candidate_scores);cap.add(nearest,distance);
      faiss::HNSWStats stats;graph_.hnsw.search_level_0(cap,res,1,&nearest,&distance,1,stats,vt,&params);
      vt.advance(); // match HNSW::search's internal advance
    }
    vt.advance(); // match IndexHNSW per-query wrapper
    if(components)t1=Clock::now();
    if(retain){
      if(candidate_ids.size()<16)throw std::runtime_error("short candidate pool: frozen L16 invalid");
      order_.resize(candidate_ids.size());std::iota(order_.begin(),order_.end(),0);
      std::partial_sort(order_.begin(),order_.begin()+16,order_.end(),[&](uint32_t a,uint32_t b){return candidate_scores[a]<candidate_scores[b]||(candidate_scores[a]==candidate_scores[b]&&a<b);});
      if(policy==StreamPolicy::Gap){out.gap=double(candidate_scores[order_[11]])-double(candidate_scores[order_[8]]);out.refined=out.gap<=threshold_;}
      else out.refined=true;
      if(out.refined)for(int j=0;j<16;++j){out.top_ids[j]=candidate_ids[order_[j]];out.top_order[j]=order_[j];}
    }
    if(components)t2=Clock::now();
    if(out.refined&&!defer)for(int j=0;j<16;++j)out.exact[j]=faiss::fvec_L2sqr(q,base_+out.top_ids[j]*graph_.d,graph_.d);
    if(components)t3=Clock::now();
    res.end();out.ids=out.native_ids;
    if(out.refined&&!defer)final_rerank(out);
  } // query-local DistanceComputer cleanup belongs to total/final time
  const auto end=Clock::now();out.total_us=us(start,end);
  if(components){out.search_us=us(start,t1);out.gap_us=us(t1,t2);out.refine_us=us(t2,t3);out.final_us=us(t3,end);}
  return out;
}
}
