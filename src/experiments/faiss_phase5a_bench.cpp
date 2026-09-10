#include "experiments/phase5a_common.h"
#include <faiss/impl/VisitedTable.h>
#include <barrier>
#include <thread>
using namespace p5;
struct Sample{int qi,pass;AccessOutput out;};
struct Worker{std::vector<Sample> samples;std::exception_ptr error;Clock::time_point end;uint64_t tls=0;};
void snapshot(const std::string& path){auto f=output(path);for(auto p:{"/proc/loadavg","/proc/stat","/proc/meminfo"}){std::ifstream in(p);f<<p<<'\n'<<in.rdbuf()<<'\n';}}
int main(int argc,char**argv){try{
 req(argc==3,"usage: faiss_phase5a_bench single|concurrent CONFIG");auto c=config(argv[2]);auto root=c.at("run")+"/";std::string stage=argv[1];req(stage=="single"||stage=="concurrent","stage");
 req(std::filesystem::exists(root+"recall_audit.json"),"recall correctness gate");if(stage=="concurrent")req(std::filesystem::exists(root+"single_analysis_complete.json"),"primary single-thread analysis gate");
 const int nq=10000,d=1536,passes=std::stoi(c.at("passes_per_repetition")),warmups=std::stoi(c.at("warmup_queries_per_worker"));
 omp_set_num_threads(1);omp_set_dynamic(0);omp_set_max_active_levels(1);pin(std::stoi(c.at("coordinator_cpu")));
 auto owner=std::unique_ptr<faiss::Index>(faiss::read_index((root+"graph.index").c_str()));auto* graph=dynamic_cast<faiss::IndexHNSW*>(owner.get());req(graph,"graph");auto* flat=dynamic_cast<faiss::IndexFlat*>(graph->storage);req(flat,"flat");
 auto po=std::unique_ptr<faiss::Index>(faiss::read_index((root+"pq.index").c_str()));auto* pq=dynamic_cast<faiss::IndexPQ*>(po.get());req(pq,"pq");architecture(*graph,*pq);auto fp=graph_fingerprint(graph->hnsw);
 auto cb=hash_bytes(pq->pq.centroids.data(),pq->pq.centroids.size()*4),codes=hash_bytes(pq->codes.data(),pq->codes.size());
 auto query=load<float>(root+"prepared/query.f32"),warm=load<float>(root+"prepared/warmup.f32");req(query.size()==size_t(nq)*d&&warm.size()==size_t(warmups)*d,"query shape");
 std::map<int,std::vector<Id>> refs_native,refs_all,refs_top;
 for(int ef:{32,64,128}){auto dir=root+"recall_ef"+std::to_string(ef)+"/";refs_native[ef]=load<Id>(dir+"native_ids.i64");refs_all[ef]=load<Id>(dir+"all16_ids.i64");refs_top[ef]=load<Id>(dir+"top16_ids.i64");}
 auto dir=root+"benchmark_"+stage+"/";req(!std::filesystem::exists(dir),"benchmark overwrite");std::filesystem::create_directory(dir);
 std::vector<std::array<int,3>> order;std::mt19937_64 rng(std::stoull(c.at("order_seed"))+(stage=="concurrent"));
 for(int rep=0;rep<5;++rep){std::vector<std::array<int,3>> cells;if(stage=="single")for(int ef:{32,64,128})for(int policy:{0,1})cells.push_back({ef,1,policy});else for(int t:{1,8})for(int policy:{0,1})cells.push_back({64,t,policy});std::shuffle(cells.begin(),cells.end(),rng);order.insert(order.end(),cells.begin(),cells.end());}
 auto schedule=output(dir+"schedule.csv");schedule<<"order,rep,ef,workers,policy\n";int cell_count=stage=="single"?6:4;for(size_t z=0;z<order.size();++z)schedule<<z<<','<<z/cell_count<<','<<order[z][0]<<','<<order[z][1]<<','<<order[z][2]<<'\n';schedule.close();
 for(size_t z=0;z<order.size();++z){int ef=order[z][0],T=order[z][1],policy=order[z][2],rep=z/cell_count;auto prefix=dir+"ef"+std::to_string(ef)+"_T"+std::to_string(T)+"_p"+std::to_string(policy)+"_r"+std::to_string(rep);snapshot(prefix+"_before.txt");
  std::vector<Worker> workers(T);std::vector<std::thread> threads;Clock::time_point start;std::barrier ready(T+1);std::barrier launch(T+1,[&]() noexcept{start=Clock::now();});
  for(int w=0;w<T;++w)threads.emplace_back([&,w]{auto& worker=workers[w];std::unique_ptr<CandidateAccess> engine;auto mode=policy?AccessMode::BoundedAll:AccessMode::Native;
   try{pin(w+2);omp_set_num_threads(1);omp_set_dynamic(0);omp_set_max_active_levels(1);req(omp_get_max_threads()==1,"internal threads");engine=std::make_unique<CandidateAccess>(*graph,*pq,flat->get_xb(),0,ef,false);worker.tls=reinterpret_cast<uint64_t>(&faiss::VisitedTable::get_reusable(graph->ntotal,graph->hnsw.use_visited_hashset));
    int per=(nq-1-w)/T+1;worker.samples.resize(passes*per);for(int j=0;j<passes*per;++j){worker.samples[j].qi=w+(j%per)*T;worker.samples[j].pass=j/per;}
    for(int j=0;j<warmups;++j)engine->run(warm.data()+size_t((j+31*w)%warmups)*d,mode,0);
   }catch(...){worker.error=std::current_exception();}
   ready.arrive_and_wait();launch.arrive_and_wait();try{if(!worker.error)for(auto&s:worker.samples)s.out=engine->run(query.data()+size_t(s.qi)*d,mode,0);}catch(...){worker.error=std::current_exception();}worker.end=Clock::now();
  });
  ready.arrive_and_wait();auto live=std::distance(std::filesystem::directory_iterator("/proc/self/task"),std::filesystem::directory_iterator());launch.arrive_and_wait();for(auto&t:threads)t.join();snapshot(prefix+"_after.txt");req(live==T+1,"nested process threads");
  auto end=start;std::set<uint64_t> tls;for(auto&w:workers){if(w.error)std::rethrow_exception(w.error);end=std::max(end,w.end);tls.insert(w.tls);}req(tls.size()==size_t(T),"shared visited TLS");
  auto rows=output(prefix+".csv");rows<<"worker,query_id,pass,latency_us\n";std::vector<Id> ids;
  for(int w=0;w<T;++w)for(auto&s:workers[w].samples){auto& out=s.out.result;auto& ref=policy?refs_all[ef]:refs_native[ef];req(std::equal(out.ids.begin(),out.ids.end(),ref.begin()+s.qi*10),"concurrent final IDs mismatch");req(std::equal(out.native_ids.begin(),out.native_ids.end(),refs_native[ef].begin()+s.qi*10),"concurrent native mismatch");if(policy)req(std::equal(out.top_ids.begin(),out.top_ids.end(),refs_top[ef].begin()+s.qi*16),"concurrent top16 mismatch");rows<<w<<','<<s.qi<<','<<s.pass<<','<<out.total_us<<'\n';ids.insert(ids.end(),out.ids.begin(),out.ids.end());}
  double elapsed=std::chrono::duration<double>(end-start).count();req(graph_fingerprint(graph->hnsw)==fp&&hash_bytes(pq->codes.data(),pq->codes.size())==codes&&hash_bytes(pq->pq.centroids.data(),pq->pq.centroids.size()*4)==cb,"index mutation");
  save(prefix+"_ids.i64",ids);auto meta=output(prefix+".json");meta<<"{\"status\":\"PASS\",\"ef\":"<<ef<<",\"workers\":"<<T<<",\"policy\":"<<policy<<",\"rep\":"<<rep<<",\"queries\":"<<nq*passes<<",\"elapsed_seconds\":"<<elapsed<<",\"qps\":"<<nq*passes/elapsed<<",\"live_threads\":"<<live<<",\"graph_fingerprint\":\""<<fp<<"\",\"codes_sha256\":\""<<codes<<"\",\"codebook_sha256\":\""<<cb<<"\",\"all_IDs_match_reference\":true}\n";
  std::cout<<stage<<" cell "<<z+1<<'/'<<order.size()<<" PASS"<<std::endl;
 }
 auto f=output(dir+"complete.json");f<<"{\"status\":\"PASS\",\"cells\":"<<order.size()<<"}\n";
 }catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}return 0;}
