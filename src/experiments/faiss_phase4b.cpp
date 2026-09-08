#include "instrumentation/streaming_refinement.h"
#include "graph/faiss_shared_hnsw.h"
#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>
#include <faiss/utils/distances.h>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <climits>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>

using Id=faiss::idx_t;using namespace quant_hardness;
void require(bool b,const std::string& m){if(!b)throw std::runtime_error(m);}
template<class T> std::vector<T> load(const std::string& p){size_t bytes=std::filesystem::file_size(p);require(bytes%sizeof(T)==0,"bad array size");std::vector<T> a(bytes/sizeof(T));std::ifstream f(p,std::ios::binary);f.read(reinterpret_cast<char*>(a.data()),bytes);require(f.good(),"read failed "+p);return a;}
template<class T> void save(const std::string& p,const std::vector<T>& a){require(!std::filesystem::exists(p),"refusing overwrite "+p);std::ofstream f(p,std::ios::binary);f.write(reinterpret_cast<const char*>(a.data()),a.size()*sizeof(T));require(f.good(),"write failed");}
template<class T> void append(std::ofstream& f,const T* data,size_t n){f.write(reinterpret_cast<const char*>(data),n*sizeof(T));require(f.good(),"append failed");}
const char* names[]={"NATIVE","GAP","RANDOM","ALL16"};

int main(int argc,char**argv){try{
  require(argc==3,"usage: faiss_phase4b reference|gt|export|benchmark|batch CONFIG");std::string mode=argv[1];
  std::ifstream cf(argv[2]);std::map<std::string,std::string> c;std::string line;
  while(std::getline(cf,line)){auto pos=line.find('=');if(pos!=std::string::npos)c[line.substr(0,pos)]=line.substr(pos+1);}
  auto num=[&](auto s){return std::stoi(c.at(s));};auto root=c.at("run")+"/";auto old=c.at("phase4a")+"/";
  omp_set_num_threads(num("threads"));
  auto owner=std::unique_ptr<faiss::Index>(faiss::read_index(c.at("graph_path").c_str()));auto* graph=dynamic_cast<faiss::IndexHNSW*>(owner.get());
  require(graph,"not HNSW");auto* flat=dynamic_cast<faiss::IndexFlat*>(graph->storage);require(flat,"not FP32 storage");
  auto pqowner=std::unique_ptr<faiss::Index>(faiss::read_index(c.at("model_path").c_str()));auto* pq=dynamic_cast<faiss::IndexPQ*>(pqowner.get());require(pq,"not PQ");
  const int d=num("dimension"),n=num("query_count"),nw=num("warmup_queries");const float* base=flat->get_xb();
  require(d==128&&num("k")==10&&num("L")==16&&num("ef_search")==64&&pq->pq.M==64&&pq->pq.nbits==8,"frozen architecture changed");
  const auto fingerprint=graph_fingerprint(graph->hnsw);require(fingerprint==c.at("graph_fingerprint"),"graph mismatch");
  auto queries=load<float>(root+"queries.f32"),warm=load<float>(root+"warmup.f32");require(queries.size()==size_t(n)*d&&warm.size()==size_t(nw)*d,"query count mismatch");
  StreamingRefinement engine(*graph,*pq,base,std::stod(c.at("gap_threshold")),num("ef_search"));
  faiss::SearchParametersHNSW params;params.efSearch=num("ef_search");
  auto public_native=[&](const float* q){StreamOutput x;graph->storage=pq;graph->search(1,q,10,x.native_scores.data(),x.native_ids.data(),&params);graph->storage=flat;return x;};
  if(mode=="reference"){
    auto ids=load<Id>(old+"candidate_ids.i64"),offsets=load<int64_t>(old+"candidate_offsets.i64");auto scores=load<float>(old+"candidate_pq.f32");
    std::vector<Id> ix,iy;std::vector<float> scalar;
    for(int q=0;q<nw;++q){auto r=engine.run(warm.data()+size_t(q)*d,StreamPolicy::Gap,false,true);auto native=public_native(warm.data()+size_t(q)*d);
      require(r.native_ids==native.native_ids&&r.native_scores==native.native_scores,"native reference identity");
      require(engine.candidate_ids.size()==size_t(offsets[q+1]-offsets[q]),"old L0 candidate count");
      for(size_t j=0;j<engine.candidate_ids.size();++j)require(engine.candidate_ids[j]==ids[offsets[q]+j]&&engine.candidate_scores[j]==scores[offsets[q]+j],"old L0 first-evaluation IDs/ADC scores");
      auto rn=engine.run(warm.data()+size_t(q)*d,StreamPolicy::Native,false,false);require(rn.ids==native.native_ids,"low-level native mismatch");
      auto all=engine.run(warm.data()+size_t(q)*d,StreamPolicy::All,false,false);
      for(int j=0;j<16;++j){ix.push_back(q);iy.push_back(all.top_ids[j]);scalar.push_back(all.exact[j]);}
    }
    std::vector<float> batched(ix.size());faiss::pairwise_indexed_L2sqr(d,ix.size(),warm.data(),ix.data(),base,iy.data(),batched.data());require(batched==scalar,"batch kernel differs from scalar");
    std::ofstream f(root+"reference_validation.json");f<<"{\"status\":\"PASS\",\"old_queries_checked\":"<<nw<<",\"old_L0_ID_score_order_identity\":true,\"public_native_identity\":true,\"scalar_batch_distances_checked\":"<<ix.size()<<"}\n";
  }else if(mode=="gt"){
    require(std::filesystem::exists(root+"reference_validation.json"),"reference gate required");
    omp_set_num_threads(num("offline_gt_threads"));faiss::distance_compute_blas_threshold=INT_MAX;
    std::vector<Id> ids(size_t(n)*11);std::vector<float> distances(size_t(n)*11);
    flat->search(n,queries.data(),11,distances.data(),ids.data());save(root+"gt_ids.i64",ids);save(root+"gt_distances.f32",distances);
  }else if(mode=="export"){
    require(std::filesystem::exists(root+"reference_validation.json"),"reference gate required");
    require(!std::filesystem::exists(root+"candidate_ids.i32"),"refuse export overwrite");
    std::ofstream idfile(root+"candidate_ids.i32",std::ios::binary),scorefile(root+"candidate_pq.f32",std::ios::binary),exactfile(root+"candidate_exact.f32",std::ios::binary);
    std::vector<int64_t> offsets{0};std::vector<Id> native,all,gapids,oracle;std::vector<double> gaps;std::vector<uint8_t> decisions;size_t count=0,maxpool=0,minpool=SIZE_MAX;
    for(int q=0;q<n;++q){const float* x=queries.data()+size_t(q)*d;auto r=engine.run(x,StreamPolicy::Gap,false,false);auto control=public_native(x);
      require(r.native_ids==control.native_ids&&r.native_scores==control.native_scores,"test native ID/score mismatch");
      const auto ids=engine.candidate_ids;const auto ps=engine.candidate_scores;size_t sz=ids.size();require(sz>=16,"short pool");minpool=std::min(minpool,sz);maxpool=std::max(maxpool,sz);
      auto unique=ids;std::sort(unique.begin(),unique.end());require(std::adjacent_find(unique.begin(),unique.end())==unique.end(),"nonunique pool");
      std::vector<size_t> order(sz);std::iota(order.begin(),order.end(),0);std::stable_sort(order.begin(),order.end(),[&](size_t a,size_t b){return ps[a]<ps[b];});
      require(r.gap==double(ps[order[11]])-double(ps[order[8]]),"gap full-sort reference mismatch");
      for(int j=0;j<10;++j)require(control.native_scores[j]==ps[order[j]],"native/global PQ strict score mismatch");
      std::vector<float> ex(sz);for(size_t j=0;j<sz;++j)ex[j]=faiss::fvec_L2sqr(x,base+size_t(ids[j])*d,d);
      StreamOutput ref;for(int j=0;j<16;++j){ref.top_ids[j]=ids[order[j]];ref.top_order[j]=order[j];ref.exact[j]=ex[order[j]];}StreamingRefinement::final_rerank(ref);
      require(r.ids==(r.refined?ref.ids:control.native_ids),"actual GAP/refinement differs from full-sort reference");
      auto physicalall=engine.run(x,StreamPolicy::All,false,false);require(physicalall.ids==ref.ids,"actual ALL16 mismatch");
      require(engine.candidate_ids==ids&&engine.candidate_scores==ps,"retention altered search");
      std::stable_sort(order.begin(),order.end(),[&](size_t a,size_t b){return ex[a]<ex[b]||(ex[a]==ex[b]&&a<b);});
      for(int j=0;j<10;++j)oracle.push_back(ids[order[j]]);
      append(idfile,ids.data(),sz);append(scorefile,ps.data(),sz);append(exactfile,ex.data(),sz);offsets.push_back(offsets.back()+sz);
      native.insert(native.end(),r.native_ids.begin(),r.native_ids.end());gapids.insert(gapids.end(),r.ids.begin(),r.ids.end());all.insert(all.end(),ref.ids.begin(),ref.ids.end());gaps.push_back(r.gap);decisions.push_back(r.refined);count+=r.refined;
    }
    save(root+"candidate_offsets.i64",offsets);save(root+"native_ids.i64",native);save(root+"all16_ids.i64",all);save(root+"gap_ids.i64",gapids);save(root+"oracle_ids.i64",oracle);save(root+"gaps.f64",gaps);save(root+"gap_decisions.u8",decisions);
    std::ofstream f(root+"export_validation.json");f<<"{\"status\":\"PASS\",\"queries\":"<<n<<",\"actual_gap_count\":"<<count<<",\"min_candidates\":"<<minpool<<",\"max_candidates\":"<<maxpool<<",\"scratch_capacity_bytes\":"<<engine.scratch_bytes()<<",\"graph_fingerprint\":\""<<fingerprint<<"\",\"all_query_native_identity\":true,\"full_sort_reference\":true}\n";
  }else if(mode=="benchmark"){
    require(std::filesystem::exists(root+"validation.json"),"independent validation required before timings");
    std::filesystem::create_directories(root+"timings");auto mask=load<uint8_t>(root+"random_masks.u8"),decisions=load<uint8_t>(root+"gap_decisions.u8");auto gaps=load<double>(root+"gaps.f64");
    auto native=load<Id>(root+"native_ids.i64"),all=load<Id>(root+"all16_ids.i64"),gap=load<Id>(root+"gap_ids.i64");auto order=load<int32_t>(root+"benchmark_order.i32");
    for(size_t s=0;s<order.size();s+=3){int rep=order[s],pi=order[s+1],components=order[s+2];auto policy=StreamPolicy(pi);std::string prefix=root+"timings/"+names[pi]+"_r"+std::to_string(rep)+"_c"+std::to_string(components);
      require(!std::filesystem::exists(prefix+".csv"),"refuse timing overwrite");
      for(int q=0;q<nw;++q)engine.run(warm.data()+size_t(q)*d,policy,q%4==0,components);
      std::vector<StreamOutput> outputs;outputs.reserve(n);
      const auto stream_start=std::chrono::steady_clock::now();
      for(int q=0;q<n;++q)outputs.push_back(engine.run(queries.data()+size_t(q)*d,policy,mask[q]!=0,components));
      const auto stream_end=std::chrono::steady_clock::now();
      // Validate only after the complete timed stream; no label-array traffic
      // between measured searches. File output likewise follows the stream.
      for(int q=0;q<n;++q){const auto& r=outputs[q];
        const auto& expected=pi==0?native:pi==1?gap:pi==3?all:mask[q]?all:native;
        require(std::equal(r.ids.begin(),r.ids.end(),expected.begin()+size_t(q)*10),"timed output changed");
        require(r.refined==(pi==0?false:pi==1?bool(decisions[q]):pi==3?true:bool(mask[q])),"timed decision changed");
        if(pi==1)require(r.gap==gaps[q],"timed gap changed");
      }
      std::ofstream f(prefix+".csv");f<<"query_id,refined,exact_evals,total_us,t_search,t_gap,t_refine,t_final_selection,gap\n"<<std::setprecision(17);
      std::vector<Id> returned;returned.reserve(size_t(n)*10);
      for(int q=0;q<n;++q){auto&r=outputs[q];f<<q<<','<<r.refined<<','<<16*r.refined<<','<<r.total_us<<','<<r.search_us<<','<<r.gap_us<<','<<r.refine_us<<','<<r.final_us<<','<<r.gap<<'\n';returned.insert(returned.end(),r.ids.begin(),r.ids.end());}
      save(prefix+"_ids.i64",returned);std::cout<<"completed "<<names[pi]<<" rep="<<rep<<" components="<<components<<'\n'<<std::flush;
      const double stream_us=std::chrono::duration<double,std::micro>(stream_end-stream_start).count();
      std::ofstream sf(prefix+"_stream.json");sf<<std::setprecision(17)<<"{\"stream_us\":"<<stream_us<<",\"stream_qps\":"<<n*1e6/stream_us<<",\"queries\":"<<n<<",\"scratch_bytes\":"<<engine.scratch_bytes()<<"}\n";
    }
    std::ofstream f(root+"benchmark_complete.json");f<<"{\"status\":\"PASS\",\"runs\":"<<order.size()/3<<",\"all_timed_IDs_decisions_validated\":true}\n";
  }else if(mode=="batch"){
    require(std::filesystem::exists(root+"benchmark_complete.json"),"primary benchmark must finish first");
    std::filesystem::create_directories(root+"batch");auto order=load<int32_t>(root+"batch_order.i32");auto expected=load<Id>(root+"gap_ids.i64");int bs=num("batch_size");
    using Clock=std::chrono::steady_clock;
    for(size_t s=0;s<order.size();s+=2){int rep=order[s],method=order[s+1];std::string file=root+"batch/m"+std::to_string(method)+"_r"+std::to_string(rep)+".csv";require(!std::filesystem::exists(file),"refuse batch overwrite");
      for(int q=0;q<nw;++q)engine.run(warm.data()+size_t(q)*d,StreamPolicy::Gap,false,false);
      std::vector<std::array<double,5>> rows;
      for(int start=0;start<n;start+=bs){int size=std::min(bs,n-start);auto t0=Clock::now();std::vector<StreamOutput> output;output.reserve(size);
        for(int j=0;j<size;++j)output.push_back(engine.run(queries.data()+size_t(start+j)*d,StreamPolicy::Gap,false,false,true));
        auto t1=Clock::now();std::vector<Id> ix,iy;std::vector<float> dist;
        if(method==1){for(int j=0;j<size;++j)if(output[j].refined)for(int k=0;k<16;++k){ix.push_back(start+j);iy.push_back(output[j].top_ids[k]);}dist.resize(ix.size());if(!ix.empty())faiss::pairwise_indexed_L2sqr(d,ix.size(),queries.data(),ix.data(),base,iy.data(),dist.data());}
        size_t pos=0,calls=0;for(int j=0;j<size;++j)if(output[j].refined){for(int k=0;k<16;++k){output[j].exact[k]=method==1?dist[pos++]:faiss::fvec_L2sqr(queries.data()+size_t(start+j)*d,base+output[j].top_ids[k]*d,d);++calls;}StreamingRefinement::final_rerank(output[j]);}
        auto t2=Clock::now();for(int j=0;j<size;++j)require(std::equal(output[j].ids.begin(),output[j].ids.end(),expected.begin()+size_t(start+j)*10),"batch differs from streaming");
        rows.push_back({double(start),double(size),std::chrono::duration<double,std::micro>(t2-t0).count(),std::chrono::duration<double,std::micro>(t2-t1).count(),double(calls)});
      }
      std::ofstream f(file);f<<"start_query,queries,total_batch_us,refine_and_final_us,exact_evals\n"<<std::setprecision(17);for(auto&r:rows)f<<r[0]<<','<<r[1]<<','<<r[2]<<','<<r[3]<<','<<r[4]<<'\n';
      std::cout<<"batch method="<<method<<" rep="<<rep<<" complete\n"<<std::flush;
    }
    std::ofstream f(root+"batch_complete.json");f<<"{\"status\":\"PASS\",\"same_output_IDs_as_streaming\":true}\n";
  }else throw std::runtime_error("unknown mode");
  require(graph_fingerprint(graph->hnsw)==fingerprint,"graph mutated");std::cout<<mode<<" PASS\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}return 0;}
