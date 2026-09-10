#include "experiments/phase5a_common.h"
#include "instrumentation/faiss_level0_recorder.h"
#include <faiss/utils/random.h>
#include <atomic>
#include <queue>
using namespace p5;

int main(int argc,char**argv){try{
 req(argc==3,"usage: faiss_phase5a gt|gt-reference|graph|pq|sanity|recall CONFIG");
 auto c=config(argv[2]);auto root=c.at("run")+"/";auto prep=root+"prepared/";std::string mode=argv[1];
 const int d=std::stoi(c.at("expected_dimension")),n=std::stoi(c.at("base_count")),nq=std::stoi(c.at("query_count"));
 req(d==1536&&n==990000&&nq==10000&&c.at("pq_m")=="768"&&c.at("ef_values")=="32,64,128","frozen config");
 req(std::filesystem::exists(prep+"provenance.json"),"preparation gate");
 omp_set_dynamic(0);omp_set_max_active_levels(1);omp_set_num_threads(24);
 if(mode=="gt"||mode=="gt-reference"){
  Mapped base(prep+"base.f32"),queries(prep+"query.f32");
  std::vector<Id> sample;if(mode=="gt-reference")sample=load<Id>(prep+"gt_validation_query_ids.i64");else{sample.resize(nq);std::iota(sample.begin(),sample.end(),0);}
  const int count=sample.size();std::vector<Id> ids(size_t(count)*11);std::vector<double> distances(size_t(count)*11);
  req(!std::filesystem::exists(root+mode+"_ids.i64"),"GT overwrite");auto start=Clock::now();std::atomic<int> done=0;
  #pragma omp parallel for schedule(dynamic)
  for(int s=0;s<count;++s){const float* q=queries.data()+sample[s]*d;std::priority_queue<std::pair<double,Id>> heap;
   for(int i=0;i<n;++i){const float* x=base.data()+size_t(i)*d;double dist=0;
    if(mode=="gt")dist=faiss::fvec_L2sqr(q,x,d);
    else for(int j=0;j<d;++j){double delta=double(q[j])-double(x[j]);dist+=delta*delta;}
    std::pair<double,Id> a{dist,i};if(heap.size()<11)heap.push(a);else if(a<heap.top()){heap.pop();heap.push(a);}
   }
   for(int j=10;j>=0;--j){ids[size_t(s)*11+j]=heap.top().second;distances[size_t(s)*11+j]=heap.top().first;heap.pop();}
   int v=++done;if(v%100==0||v==count){
    #pragma omp critical
    std::cout<<mode<<" queries "<<v<<'/'<<count<<std::endl;
   }
  }
  save(root+mode+"_ids.i64",ids);save(root+mode+"_distances.f64",distances);
  auto f=output(root+mode+".json");f<<"{\"queries\":"<<count<<",\"seconds\":"<<seconds(start)<<",\"ids_sha256\":\""<<file_hash(root+mode+"_ids.i64")<<"\",\"tie_order\":\"distance then ascending base ID; strict ID recall unchanged\"}\n";
  return 0;
 }
 if(mode=="graph"){
  Mapped base(prep+"base.f32");req(!std::filesystem::exists(root+"graph.index"),"graph overwrite");
  faiss::IndexHNSWFlat graph(d,std::stoi(c.at("hnsw_m")),faiss::METRIC_L2);graph.hnsw.efConstruction=std::stoi(c.at("ef_construction"));graph.hnsw.rng=faiss::RandomGenerator(std::stoi(c.at("graph_seed")));graph.verbose=true;
  omp_set_num_threads(std::stoi(c.at("construction_threads")));auto start=Clock::now();graph.add(n,base.data());double elapsed=seconds(start);
  faiss::write_index(&graph,(root+"graph.index").c_str());auto f=output(root+"graph.json");
  f<<"{\"seconds\":"<<elapsed<<",\"fingerprint\":\""<<graph_fingerprint(graph.hnsw)<<"\",\"file_sha256\":\""<<file_hash(root+"graph.index")<<"\",\"bytes\":"<<std::filesystem::file_size(root+"graph.index")<<",\"M\":16,\"efConstruction\":80,\"seed\":20260907,\"threads\":24,\"metric\":\"normalized FP32 squared L2\",\"add_calls\":1}\n";return 0;
 }
 if(mode=="pq"){
  req(!std::filesystem::exists(root+"pq.index"),"PQ overwrite");Mapped train(prep+"training.f32"),base(prep+"base.f32");
  faiss::IndexPQ pq(d,768,8,faiss::METRIC_L2);pq.pq.cp.seed=std::stoi(c.at("pq_seed"));pq.pq.cp.niter=std::stoi(c.at("pq_niter"));pq.verbose=true;pq.pq.verbose=true;
  omp_set_num_threads(std::stoi(c.at("pq_training_threads")));auto start=Clock::now();pq.train(65536,train.data());double training=seconds(start);start=Clock::now();pq.add(n,base.data());double encoding=seconds(start);
  faiss::write_index(&pq,(root+"pq.index").c_str());auto f=output(root+"pq.json");f<<"{\"training_seconds\":"<<training<<",\"encoding_seconds\":"<<encoding<<",\"code_size\":"<<pq.code_size<<",\"dsub\":"<<pq.pq.dsub<<",\"training_rows\":65536,\"seed\":"<<pq.pq.cp.seed<<",\"niter\":"<<pq.pq.cp.niter<<",\"nredo\":"<<pq.pq.cp.nredo<<",\"max_points_per_centroid\":"<<pq.pq.cp.max_points_per_centroid<<",\"codebook_sha256\":\""<<hash_bytes(pq.pq.centroids.data(),pq.pq.centroids.size()*4)<<"\",\"codes_sha256\":\""<<hash_bytes(pq.codes.data(),pq.codes.size())<<"\",\"file_sha256\":\""<<file_hash(root+"pq.index")<<"\"}\n";return 0;
 }
 auto po=std::unique_ptr<faiss::Index>(faiss::read_index((root+"pq.index").c_str()));auto* pq=dynamic_cast<faiss::IndexPQ*>(po.get());req(pq,"PQ index");
 if(mode=="sanity"){
  Mapped base(prep+"base.f32"),queries(prep+"query.f32");omp_set_num_threads(1);std::mt19937_64 rng(std::stoull(c.at("quality_seed")));
  auto f=output(root+"sanity_pairs.csv");f<<"sample,query_id,base_id,other_id,exact,pq,absolute_relative_error,reconstruction_l2,order_inversion,adc_reference_error\n";
  std::unique_ptr<faiss::DistanceComputer> dc(pq->get_distance_computer());std::vector<float> reconstructed(d);std::vector<uint8_t> code(768);
  for(int s=0;s<std::stoi(c.at("quality_pairs"));++s){Id qi=rng()%nq,i=rng()%n,j=rng()%n;while(j==i)j=rng()%n;auto* q=queries.data()+qi*d;auto* x=base.data()+i*d;dc->set_query(q);float exact=faiss::fvec_L2sqr(q,x,d),approx=(*dc)(i),other=(*dc)(j),eother=faiss::fvec_L2sqr(q,base.data()+j*d,d);
   pq->reconstruct(i,reconstructed.data());float recon=std::sqrt(faiss::fvec_L2sqr(x,reconstructed.data(),d));float ref=faiss::fvec_L2sqr(q,reconstructed.data(),d);
   req(std::isfinite(approx)&&std::abs(approx-ref)<=2e-5f,"ADC/reconstruction reference mismatch");pq->pq.compute_code(x,code.data());req(std::equal(code.begin(),code.end(),pq->codes.begin()+i*768),"PQ ID alignment");
   f<<s<<','<<qi<<','<<i<<','<<j<<','<<exact<<','<<approx<<','<<(exact>1e-12?std::abs(approx-exact)/exact:std::numeric_limits<float>::quiet_NaN())<<','<<recon<<','<<((exact-eother)*(approx-other)<0)<<','<<std::abs(approx-ref)<<'\n';
  }auto pass=output(root+"sanity_complete.json");pass<<"{\"status\":\"PASS\",\"sample_pairs\":10000,\"sample_order_pairs\":10000,\"reference_ADC_and_code_alignment_checked\":true}\n";return 0;
 }
 req(mode=="recall","unknown mode");req(std::filesystem::exists(root+"gt_validation.json")&&std::filesystem::exists(root+"sanity_complete.json"),"GT/sanity gates");
 auto owner=std::unique_ptr<faiss::Index>(faiss::read_index((root+"graph.index").c_str()));auto* graph=dynamic_cast<faiss::IndexHNSW*>(owner.get());req(graph,"graph");auto* flat=dynamic_cast<faiss::IndexFlat*>(graph->storage);req(flat,"flat");architecture(*graph,*pq);
 auto fingerprint=graph_fingerprint(graph->hnsw);auto queries=load<float>(prep+"query.f32");auto gt=load<Id>(root+"gt_ids.i64");omp_set_num_threads(1);
 for(int ef:{32,64,128}){
  std::string dir=root+"recall_ef"+std::to_string(ef)+"/";req(!std::filesystem::exists(dir),"recall overwrite");std::filesystem::create_directory(dir);
  faiss::SearchParametersHNSW params;params.efSearch=ef;
  auto exact=search_with_level0_recording(*graph,*flat,queries.data(),nq,10,params,true);
  auto pq_reference=search_with_storage(*graph,*pq,queries.data(),nq,10,params);
  CandidateAccess full(*graph,*pq,flat->get_xb(),0,ef,true),bounded(*graph,*pq,flat->get_xb(),0,ef,false);
  auto rows=output(dir+"per_query.csv");rows<<"query_id,recall_exact,recall_pq,recall_candidate_oracle,recall_all16,recall_exact_oracle,delta_exact_control,total_loss,discovery_loss,ranking_loss,exact_l0_count,pq_l0_count,coverage_exact,coverage_pq,top16_boundary_tie\n";
  std::vector<Id> native,all,top,oracle,exact_oracle;std::vector<int32_t> candidates;std::vector<int64_t> offsets{0};std::vector<float> scores;
  for(int qi=0;qi<nq;++qi){const float* q=queries.data()+size_t(qi)*d;auto a=full.run(q,AccessMode::FullAll,0),b=bounded.run(q,AccessMode::BoundedAll,0),plain=bounded.run(q,AccessMode::Native,0);
   req(a.result.ids==b.result.ids&&a.result.top_ids==b.result.top_ids&&a.top_scores==b.top_scores&&a.result.top_order==b.result.top_order,"bounded/full top16 or final mismatch");req(a.result.native_ids==plain.result.ids&&b.result.native_ids==plain.result.ids,"native instrumentation mismatch");
   req(std::equal(plain.result.ids.begin(),plain.result.ids.end(),pq_reference.ids.begin()+qi*10),"ordinary FAISS PQ-native ID mismatch");
   req(std::equal(plain.result.native_scores.begin(),plain.result.native_scores.end(),pq_reference.distances.begin()+qi*10),"ordinary FAISS PQ-native score mismatch");
   auto ids=full.ids();req(std::set<int32_t>(ids.begin(),ids.end()).size()==ids.size(),"nonunique L0");auto pq_or=rerank(flat->get_xb(),q,d,ids);auto eids=exact.level0_evaluation_order_ids[qi];std::vector<int32_t> ei(eids.begin(),eids.end());auto eor=rerank(flat->get_xb(),q,d,ei);
   const Id* truth=gt.data()+size_t(qi)*11;double re=recall(exact.native.ids.data()+qi*10,truth),rp=recall(plain.result.ids.data(),truth),ro=recall(pq_or.data(),truth),ra=recall(a.result.ids.data(),truth),reo=recall(eor.data(),truth);
   auto coverage=[&](const auto& v){int hits=0;for(int k=0;k<10;++k)if(std::find(v.begin(),v.end(),truth[k])!=v.end())++hits;return hits/10.;};
   bool tie=false;for(size_t j=0;j<ids.size();++j)if(full.scores()[j]==a.top_scores[15]&&std::find(a.result.top_ids.begin(),a.result.top_ids.end(),ids[j])==a.result.top_ids.end())tie=true;
   rows<<qi<<','<<re<<','<<rp<<','<<ro<<','<<ra<<','<<reo<<','<<reo-re<<','<<re-rp<<','<<re-ro<<','<<ro-rp<<','<<ei.size()<<','<<ids.size()<<','<<coverage(ei)<<','<<coverage(ids)<<','<<tie<<'\n';
   native.insert(native.end(),plain.result.ids.begin(),plain.result.ids.end());all.insert(all.end(),a.result.ids.begin(),a.result.ids.end());top.insert(top.end(),a.result.top_ids.begin(),a.result.top_ids.end());oracle.insert(oracle.end(),pq_or.begin(),pq_or.end());exact_oracle.insert(exact_oracle.end(),eor.begin(),eor.end());
   candidates.insert(candidates.end(),ids.begin(),ids.end());scores.insert(scores.end(),full.scores().begin(),full.scores().end());offsets.push_back(candidates.size());
   if(qi%1000==0)std::cout<<"recall ef"<<ef<<" query "<<qi<<std::endl;
  }
  save(dir+"native_ids.i64",native);save(dir+"all16_ids.i64",all);save(dir+"top16_ids.i64",top);save(dir+"oracle_ids.i64",oracle);save(dir+"exact_ids.i64",exact.native.ids);save(dir+"exact_oracle_ids.i64",exact_oracle);save(dir+"pq_candidates.i32",candidates);save(dir+"pq_scores.f32",scores);save(dir+"pq_offsets.i64",offsets);
  req(graph_fingerprint(graph->hnsw)==fingerprint,"graph changed");auto f=output(dir+"complete.json");f<<"{\"status\":\"PASS\",\"graph_fingerprint\":\""<<fingerprint<<"\",\"full_bounded_native_identity_all_queries\":true}\n";
 }
 }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}return 0;}
