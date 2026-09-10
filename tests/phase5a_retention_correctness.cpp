#include "experiments/phase5a_common.h"
using namespace p5;
int main(){try{
 omp_set_num_threads(1);const int n=512,d=1536;std::mt19937 rng(95000101);std::normal_distribution<float> normal;
 std::vector<float> x(size_t(n)*d),q(size_t(12)*d);for(float&v:x)v=normal(rng);for(float&v:q)v=normal(rng);
 faiss::IndexHNSWFlat graph(d,16);graph.hnsw.efConstruction=80;graph.add(n,x.data());
 // Standard FAISS PQ with deterministic fixture centroids, not another
 // trained experimental model. Tests the d1536/M768 distance-computer path.
 faiss::IndexPQ pq(d,768,8);for(float&v:pq.pq.centroids)v=normal(rng);pq.is_trained=true;pq.add(n,x.data());
 auto fp=graph_fingerprint(graph.hnsw);
 for(int ef:{32,64,128}){CandidateAccess full(graph,pq,x.data(),0,ef,true),bound(graph,pq,x.data(),0,ef,false);
  for(int qi=0;qi<12;++qi){auto* query=q.data()+qi*d;auto a=full.run(query,AccessMode::FullAll,0);auto b=bound.run(query,AccessMode::BoundedAll,0);auto native=bound.run(query,AccessMode::Native,0);
   req(a.result.ids==b.result.ids&&a.result.top_ids==b.result.top_ids&&a.top_scores==b.top_scores&&a.result.top_order==b.result.top_order,"full/bounded mismatch");req(a.result.native_ids==native.result.ids,"native mismatch");
   std::vector<std::pair<float,size_t>> sort;for(size_t j=0;j<full.ids().size();++j)sort.emplace_back(full.scores()[j],j);std::sort(sort.begin(),sort.end());std::vector<int32_t> top;
   for(int j=0;j<16;++j){top.push_back(full.ids()[sort[j].second]);req(top.back()==b.result.top_ids[j],"offline full-sort mismatch");}
   auto reference=rerank(x.data(),query,d,top);req(std::equal(reference.begin(),reference.end(),b.result.ids.begin()),"exact refinement reference mismatch");
  }
 }
 req(graph_fingerprint(graph.hnsw)==fp,"graph changed");std::cout<<"PASS d1536/PQ768; ef32/64/128; full-sort, refinement, native identity and graph invariance\n";
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}return 0;}
