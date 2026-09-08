// Phase 4A export only. Policies are separate offline replay of PQ-visible data.
#include "graph/faiss_shared_hnsw.h"
#include "instrumentation/faiss_level0_recorder.h"
#include <faiss/IndexFlat.h>
#include <faiss/IndexPQ.h>
#include <faiss/index_io.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/utils/distances.h>
#include <omp.h>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

using Id=faiss::idx_t;
void require(bool ok,const std::string& message){if(!ok)throw std::runtime_error(message);}
template<class T> std::vector<T> read(const std::string& path,size_t count){
  require(std::filesystem::file_size(path)==count*sizeof(T),"wrong file size: "+path);
  std::vector<T> a(count);std::ifstream f(path,std::ios::binary);
  f.read(reinterpret_cast<char*>(a.data()),a.size()*sizeof(T));require(f.good(),"read failed");return a;
}
template<class T> void write(const std::string& path,const std::vector<T>& a){
  require(!std::filesystem::exists(path),"refusing overwrite: "+path);
  std::ofstream f(path,std::ios::binary);f.write(reinterpret_cast<const char*>(a.data()),a.size()*sizeof(T));require(f.good(),"write failed");
}
template<class T> void append(std::ofstream& f,const std::vector<T>& a){
  f.write(reinterpret_cast<const char*>(a.data()),a.size()*sizeof(T));require(f.good(),"append failed");
}

int main(int argc,char**argv){try{
  require(argc==2,"usage faiss_phase4a CONFIG");std::map<std::string,std::string> c;
  std::ifstream cf(argv[1]);std::string line;while(std::getline(cf,line)){auto p=line.find('=');if(p!=std::string::npos)c[line.substr(0,p)]=line.substr(p+1);}
  auto integer=[&](auto key){return std::stoi(c.at(key));};auto root=c.at("run")+"/";
  require(std::filesystem::exists(root+"provenance.json"),"split/provenance required");
  require(!std::filesystem::exists(root+"model.index"),"refusing partial rerun");
  const int d=integer("dimension"),nq=integer("query_count"),nb=integer("base_count"),k=integer("k");
  auto queries=read<float>(root+"queries.f32",size_t(nq)*d);
  auto train=read<float>(root+"training.f32",size_t(integer("training_count"))*d);
  std::unique_ptr<faiss::Index> owner(faiss::read_index(c.at("graph_path").c_str()));
  auto* graph=dynamic_cast<faiss::IndexHNSW*>(owner.get());require(graph&&graph->ntotal==nb&&graph->d==d,"invalid frozen graph");
  auto* flat=dynamic_cast<faiss::IndexFlat*>(graph->storage);require(flat,"graph storage must be FP32 flat");
  auto* base=flat->get_xb();
  // Byte-for-byte graph storage / source ID alignment for the complete database.
  std::ifstream bf(c.at("base_path"),std::ios::binary);std::vector<float> row(d);
  for(int i=0;i<nb;++i){int dimension;bf.read(reinterpret_cast<char*>(&dimension),4);bf.read(reinterpret_cast<char*>(row.data()),d*4);
    require(bf.good()&&dimension==d&&std::equal(row.begin(),row.end(),base+size_t(i)*d),"base ID alignment");}
  const auto fingerprint=quant_hardness::graph_fingerprint(graph->hnsw);
  require(fingerprint==c.at("graph_fingerprint"),"historical graph fingerprint mismatch");
  omp_set_num_threads(integer("training_threads"));
  faiss::IndexPQ pq(d,integer("pq_m"),integer("pq_nbits"),faiss::METRIC_L2);
  pq.pq.cp.seed=integer("pq_seed");pq.pq.cp.niter=integer("pq_iterations");
  std::cout<<"training one frozen PQ model\n"<<std::flush;
  pq.train(integer("training_count"),train.data());pq.add(nb,base);
  faiss::write_index(&pq,(root+"model.index").c_str());
  write(root+"codebooks.f32",pq.pq.centroids);
  std::vector<uint8_t> codes(pq.codes.size());pq.sa_encode(nb,base,codes.data());
  require(std::equal(codes.begin(),codes.end(),pq.codes.data()),"complete PQ code ID alignment failed");codes.clear();codes.shrink_to_fit();
  std::cout<<"exhaustive FP32 ground truth\n"<<std::flush;
  std::vector<float> gd(size_t(nq)*11);std::vector<Id> gi(size_t(nq)*11);
  flat->search(nq,queries.data(),11,gd.data(),gi.data());
  write(root+"gt_distances.f32",gd);write(root+"gt_ids.i64",gi);
  omp_set_num_threads(integer("search_threads"));faiss::SearchParametersHNSW params;params.efSearch=integer("ef_search");
  std::cout<<"PQ L0 search with native identity control\n"<<std::flush;
  auto p=quant_hardness::search_with_level0_recording(*graph,pq,queries.data(),nq,k,params,true);
  require(quant_hardness::graph_fingerprint(graph->hnsw)==fingerprint,"graph changed after PQ");
  write(root+"pq_native_ids.i64",p.native.ids);write(root+"pq_native_scores.f32",p.native.distances);
  std::ofstream idout(root+"candidate_ids.i64",std::ios::binary),deout(root+"candidate_exact.f32",std::ios::binary),dpout(root+"candidate_pq.f32",std::ios::binary);
  std::vector<int64_t> offsets{0},upper;
  std::unique_ptr<faiss::DistanceComputer> dc(pq.get_distance_computer());
  for(int q=0;q<nq;++q){auto& ids=p.level0_evaluation_order_ids[q];require(ids.size()>=64,"short pool: equal-budget protocol must stop");
    auto sorted=ids;std::sort(sorted.begin(),sorted.end());require(std::adjacent_find(sorted.begin(),sorted.end())==sorted.end(),"duplicate candidate IDs");
    std::vector<float> exact(ids.size()),approx(ids.size());dc->set_query(queries.data()+size_t(q)*d);
    for(size_t j=0;j<ids.size();++j){approx[j]=(*dc)(ids[j]);exact[j]=faiss::fvec_L2sqr(queries.data()+size_t(q)*d,base+ids[j]*d,d);}
    append(idout,ids);append(deout,exact);append(dpout,approx);offsets.push_back(offsets.back()+ids.size());upper.push_back(p.upper_only_evaluated_ids[q].size());
  }
  write(root+"candidate_offsets.i64",offsets);write(root+"pq_upper_counts.i64",upper);
  // Separate control only: these candidate IDs are never used by a policy.
  std::cout<<"exact L0 discovery/control measurement\n"<<std::flush;
  auto e=quant_hardness::search_with_level0_recording(*graph,*flat,queries.data(),nq,k,params,true);
  write(root+"exact_native_ids.i64",e.native.ids);write(root+"exact_native_scores.f32",e.native.distances);
  std::vector<Id> eo;std::vector<float> ed;std::vector<int64_t> ec;
  for(int q=0;q<nq;++q){auto ids=e.level0_evaluation_order_ids[q];std::vector<float> ds(ids.size());std::vector<size_t> ix(ids.size());std::iota(ix.begin(),ix.end(),0);
    for(size_t j=0;j<ids.size();++j)ds[j]=faiss::fvec_L2sqr(queries.data()+size_t(q)*d,base+ids[j]*d,d);
    std::stable_sort(ix.begin(),ix.end(),[&](size_t a,size_t b){return ds[a]<ds[b];});
    for(int j=0;j<k;++j){eo.push_back(ids[ix[j]]);ed.push_back(ds[ix[j]]);}ec.push_back(ids.size());
  }
  write(root+"exact_control_oracle_ids.i64",eo);write(root+"exact_control_oracle_scores.f32",ed);write(root+"exact_control_pool_counts.i64",ec);
  require(quant_hardness::graph_fingerprint(graph->hnsw)==fingerprint,"graph changed after exact");
  std::ofstream out(root+"export_validation.json");out<<"{\"status\":\"PASS\",\"graph_fingerprint\":\""<<fingerprint<<"\",\"instrumented_native_identity_all_queries\":true,\"complete_base_storage_alignment\":true,\"complete_pq_code_alignment\":true,\"min_candidates\":64,\"pq_niter\":"<<pq.pq.cp.niter<<",\"pq_max_points_per_centroid\":"<<pq.pq.cp.max_points_per_centroid<<"}\n";
  std::cout<<"export PASS\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}return 0;}
