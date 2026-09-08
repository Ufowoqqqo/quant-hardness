#include "instrumentation/bounded_retention.h"
#include "graph/faiss_shared_hnsw.h"
#include <faiss/IndexFlat.h>
#include <faiss/impl/VisitedTable.h>
#include <faiss/index_io.h>
#include <openssl/evp.h>
#include <omp.h>
#include <pthread.h>
#include <sched.h>
#include <sys/syscall.h>
#include <fcntl.h>
#include <poll.h>
#include <unistd.h>
#include <algorithm>
#include <barrier>
#include <chrono>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <thread>

using namespace quant_hardness;
using Id=faiss::idx_t;using Clock=std::chrono::steady_clock;
void require(bool b,const std::string& s){if(!b)throw std::runtime_error(s);}
double micros(Clock::time_point a,Clock::time_point b){return std::chrono::duration<double,std::micro>(b-a).count();}
double cpu_seconds(){timespec t;require(clock_gettime(CLOCK_THREAD_CPUTIME_ID,&t)==0,"thread CPU clock");return t.tv_sec+t.tv_nsec*1e-9;}
template<class T>std::vector<T> load(const std::string& p){auto n=std::filesystem::file_size(p);require(n%sizeof(T)==0,"array size "+p);std::vector<T> a(n/sizeof(T));std::ifstream f(p,std::ios::binary);f.read(reinterpret_cast<char*>(a.data()),n);require(f.good(),"read "+p);return a;}
template<class T>void save(const std::string& p,const std::vector<T>& a){require(!std::filesystem::exists(p),"refuse overwrite "+p);std::ofstream f(p,std::ios::binary);f.write(reinterpret_cast<const char*>(a.data()),a.size()*sizeof(T));require(f.good(),"write "+p);}
std::vector<int> ints(std::string s){std::replace(s.begin(),s.end(),',',' ');std::istringstream in(s);std::vector<int> v;int i;while(in>>i)v.push_back(i);return v;}
void pin(int cpu){cpu_set_t mask;CPU_ZERO(&mask);CPU_SET(cpu,&mask);require(pthread_setaffinity_np(pthread_self(),sizeof(mask),&mask)==0,"pin failed");cpu_set_t got;require(pthread_getaffinity_np(pthread_self(),sizeof(got),&got)==0&&CPU_COUNT(&got)==1&&CPU_ISSET(cpu,&got),"affinity mismatch");}
size_t task_count(){return std::distance(std::filesystem::directory_iterator("/proc/self/task"),std::filesystem::directory_iterator());}
void snapshot(const std::string& path,const std::vector<int>& cpus){
    std::ofstream out(path);out<<"unix_ns "<<std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count()<<'\n';
    for(const auto& p:{"/proc/loadavg","/proc/stat","/proc/meminfo"}){std::ifstream f(p);out<<p<<'\n'<<f.rdbuf()<<'\n';}
    for(int cpu:cpus){std::string p="/sys/devices/system/cpu/cpu"+std::to_string(cpu)+"/cpufreq/scaling_cur_freq";std::ifstream f(p);if(f.good())out<<p<<' '<<f.rdbuf()<<'\n';}
}
std::string pq_hash(const faiss::IndexPQ& pq){
    auto* ctx=EVP_MD_CTX_new();require(ctx!=nullptr,"hash allocation");EVP_DigestInit_ex(ctx,EVP_sha256(),nullptr);
    auto add=[&](const auto& v){size_t n=v.size();EVP_DigestUpdate(ctx,&n,sizeof(n));if(n)EVP_DigestUpdate(ctx,v.data(),n*sizeof(v[0]));};
    add(pq.codes);add(pq.pq.centroids);add(pq.pq.transposed_centroids);add(pq.pq.centroids_sq_lengths);add(pq.pq.sdc_table);
    unsigned char out[EVP_MAX_MD_SIZE];unsigned int n=0;EVP_DigestFinal_ex(ctx,out,&n);EVP_MD_CTX_free(ctx);std::ostringstream s;s<<std::hex<<std::setfill('0');for(unsigned int i=0;i<n;++i)s<<std::setw(2)<<int(out[i]);return s.str();
}
struct PerfControl {
    int ctl=-1,ack=-1;
    PerfControl(const std::string& c,const std::string& a){ctl=open(c.c_str(),O_WRONLY|O_NONBLOCK);ack=open(a.c_str(),O_RDONLY|O_NONBLOCK);require(ctl>=0&&ack>=0,"perf FIFO open failed");}
    ~PerfControl(){if(ctl>=0)close(ctl);if(ack>=0)close(ack);}
    void command(const std::string& s){auto cmd=s+"\n";require(write(ctl,cmd.data(),cmd.size())==ssize_t(cmd.size()),"perf write failed");pollfd fd{ack,POLLIN,0};require(poll(&fd,1,10000)>0,"perf ack timed out");char buf[64];ssize_t n=read(ack,buf,sizeof(buf));require(n>=3&&std::string(buf,n).starts_with("ack"),"perf ack failed");}
};
struct Sample {int query=0,pass=0;AccessMode mode=AccessMode::Native;AccessOutput out;};
struct Worker {
    std::vector<Sample> rows;
    std::exception_ptr error;
    Clock::time_point end;
    double cpu=0,start_lag_us=0;
    uint64_t visited_address=0;
    int tid=0,cpu_start=-1,cpu_end=-1,omp_threads=0;
};

int main(int argc,char**argv){try{
    require(argc==3||argc==7,"usage: faiss_phase4d stress|benchmark CONFIG; profile CONFIG T POLICY CTL ACK");
    std::string phase=argv[1];std::map<std::string,std::string> c;std::ifstream cf(argv[2]);std::string line;
    while(std::getline(cf,line)){auto p=line.find('=');if(p!=std::string::npos)c[line.substr(0,p)]=line.substr(p+1);}
    auto num=[&](const auto& key){return std::stoi(c.at(key));};auto root=c.at("run")+"/",old=c.at("phase4b")+"/",prev=c.at("phase4c")+"/";
    int n=num("query_count"),d=num("dimension"),passes=num("passes_per_repetition"),nw=num("warmup_queries_per_worker");
    const auto cpus=ints(c.at("worker_cpus")),levels=ints(c.at("workers"));
    pin(num("coordinator_cpu"));omp_set_num_threads(1);omp_set_dynamic(0);omp_set_max_active_levels(1);
    auto owner=std::unique_ptr<faiss::Index>(faiss::read_index(c.at("graph_path").c_str()));auto* graph=dynamic_cast<faiss::IndexHNSW*>(owner.get());require(graph,"HNSW required");
    auto* flat=dynamic_cast<faiss::IndexFlat*>(graph->storage);require(flat,"FP32 storage required");
    auto po=std::unique_ptr<faiss::Index>(faiss::read_index(c.at("model_path").c_str()));auto* pq=dynamic_cast<faiss::IndexPQ*>(po.get());require(pq,"PQ required");
    require(d==128&&n==14252&&num("k")==10&&num("L")==16&&num("ef_search")==64&&num("internal_threads")==1&&pq->pq.M==64&&pq->pq.nbits==8,"frozen architecture changed");
    const auto fingerprint=graph_fingerprint(graph->hnsw),model_hash=pq_hash(*pq);require(fingerprint==c.at("graph_fingerprint"),"graph mismatch");
    auto queries=load<float>(old+"queries.f32"),warm=load<float>(old+"warmup.f32");require(queries.size()==size_t(n)*d&&warm.size()==size_t(nw)*d,"query shapes");
    auto native=load<Id>(old+"native_ids.i64"),all=load<Id>(old+"all16_ids.i64"),top=load<Id>(prev+"top16_ids.i64");
    auto top_scores=load<float>(prev+"top16_scores.f32");auto top_order=load<uint32_t>(prev+"top16_order.u32");auto offsets=load<int64_t>(old+"candidate_offsets.i64");
    auto stress_queries=load<int32_t>(root+"stress_queries.i32");
    std::unique_ptr<PerfControl> perf;
    if(phase=="profile"){require(argc==7,"profile arguments");perf=std::make_unique<PerfControl>(argv[5],argv[6]);}
    else require(argc==3&&(phase=="stress"||phase=="benchmark"),"invalid mode");
    if(phase!="stress")require(std::filesystem::exists(root+"stress_complete.json"),"stress gate required");
    if(phase=="profile")require(std::filesystem::exists(root+"benchmark_complete.json"),"primary gate required");
    auto validate=[&](const Sample& s){
        const auto& r=s.out.result;size_t q=s.query;bool refine=s.mode==AccessMode::BoundedAll;
        const auto& ids=refine?all:native;
        require(std::equal(r.ids.begin(),r.ids.end(),ids.begin()+q*10),"concurrent returned ID mismatch");
        require(std::equal(r.native_ids.begin(),r.native_ids.end(),native.begin()+q*10),"concurrent native ID mismatch");
        require(r.refined==refine,"refinement flag mismatch");
        if(refine){require(std::equal(r.top_ids.begin(),r.top_ids.end(),top.begin()+q*16)&&std::equal(r.top_order.begin(),r.top_order.end(),top_order.begin()+q*16),"concurrent top16 ID/order mismatch");
            require(std::memcmp(s.out.top_scores.data(),top_scores.data()+q*16,16*sizeof(float))==0&&s.out.candidates==size_t(offsets[q+1]-offsets[q]),"concurrent top16 score/count mismatch");}
    };
    auto cell=[&](int T,int policy,int rep,bool stress){
        require(T>0&&T<=int(cpus.size()),"worker count");const char* name=policy==0?"NATIVE":"BOUNDED_ALL16";
        std::string prefix=root+phase+"/"+(stress?"MIXED":name)+"_T"+std::to_string(T)+"_r"+std::to_string(rep);std::filesystem::create_directories(root+phase);
        require(!std::filesystem::exists(prefix+".json"),"refuse cell overwrite");
        snapshot(prefix+"_before.txt",cpus);
        std::vector<Worker> workers(T);std::vector<std::thread> threads;Clock::time_point start;
        std::barrier ready(T+1);std::barrier launch(T+1,[&]() noexcept {start=Clock::now();});
        for(int w=0;w<T;++w)threads.emplace_back([&,w]{
            auto& worker=workers[w];std::unique_ptr<CandidateAccess> engine;
            try {
                pin(cpus[w]);omp_set_num_threads(1);omp_set_dynamic(0);omp_set_max_active_levels(1);
                worker.tid=syscall(SYS_gettid);worker.omp_threads=omp_get_max_threads();require(worker.omp_threads==1&&!omp_in_parallel(),"nested query parallelism");
                engine=std::make_unique<CandidateAccess>(*graph,*pq,flat->get_xb(),std::stod(c.at("gap_threshold")),64,false);
                worker.visited_address=reinterpret_cast<uintptr_t>(&faiss::VisitedTable::get_reusable(graph->ntotal,graph->hnsw.use_visited_hashset));
                int count=stress?num("stress_calls_per_worker"):passes*((n-1-w)/T+1);worker.rows.resize(count);
                for(int j=0;j<count;++j){auto& s=worker.rows[j];
                    if(stress){int sample=int(stress_queries.size());int idx=j%4==0?(j/4)%sample:j%4==1?(j*17+w*31)%sample:j%4==2?((j-1)*17+w*31)%sample:(sample-1-(j% sample));s.query=stress_queries[idx];s.mode=j%4==0||j%4==3?AccessMode::Native:AccessMode::BoundedAll;s.pass=0;}
                    else {int per=(n-1-w)/T+1;s.query=w+(j%per)*T;s.pass=j/per;s.mode=policy==0?AccessMode::Native:AccessMode::BoundedAll;}
                }
                for(int j=0;j<nw;++j)engine->run(warm.data()+size_t((j+31*w)%nw)*d,policy==0?AccessMode::Native:AccessMode::BoundedAll,0);
            }catch(...){worker.error=std::current_exception();}
            ready.arrive_and_wait();launch.arrive_and_wait();
            if(worker.error){worker.end=Clock::now();return;}
            try {
                auto begin=Clock::now();worker.start_lag_us=micros(start,begin);worker.cpu_start=sched_getcpu();double cpu0=cpu_seconds();
                for(auto& s:worker.rows)s.out=engine->run(queries.data()+size_t(s.query)*d,s.mode,0);
                worker.end=Clock::now();worker.cpu=cpu_seconds()-cpu0;worker.cpu_end=sched_getcpu();
                require(engine->full_scratch_bytes()==0,"full recorder unexpectedly allocated");
            }catch(...){worker.error=std::current_exception();worker.end=Clock::now();}
        });
        ready.arrive_and_wait();size_t live=task_count();std::exception_ptr perf_error;
        if(perf)try{perf->command("enable");}catch(...){perf_error=std::current_exception();}
        launch.arrive_and_wait();for(auto& t:threads)t.join();
        if(perf&&!perf_error)try{perf->command("disable");}catch(...){perf_error=std::current_exception();}
        if(perf_error)std::rethrow_exception(perf_error);
        snapshot(prefix+"_after.txt",cpus);
        require(live==size_t(T+1),"unexpected process threads/nested parallelism");
        for(auto&w:workers)if(w.error)std::rethrow_exception(w.error);
        std::vector<uint64_t> addresses;for(auto&w:workers)addresses.push_back(w.visited_address);std::sort(addresses.begin(),addresses.end());require(std::adjacent_find(addresses.begin(),addresses.end())==addresses.end(),"visited state shared across workers");
        Clock::time_point last=start;double cpu=0;size_t count=0;
        for(int w=0;w<T;++w){auto& a=workers[w];require(a.cpu_start==cpus[w]&&a.cpu_end==cpus[w],"worker migration/affinity");last=std::max(last,a.end);cpu+=a.cpu;count+=a.rows.size();for(auto&s:a.rows)validate(s);}
        require(graph_fingerprint(graph->hnsw)==fingerprint&&pq_hash(*pq)==model_hash,"shared immutable index mutated");
        double elapsed=micros(start,last);std::ofstream rows(prefix+".csv");rows<<"worker,sequence,query_id,pass,policy,latency_us,exact_evals\n"<<std::setprecision(17);
        std::vector<Id> returned;returned.reserve(count*10);size_t exact_calls=0;
        for(int w=0;w<T;++w){size_t j=0;for(auto&s:workers[w].rows){int calls=s.mode==AccessMode::BoundedAll?16:0;exact_calls+=calls;rows<<w<<','<<j++<<','<<s.query<<','<<s.pass<<','<<(s.mode==AccessMode::BoundedAll?1:0)<<','<<s.out.result.total_us<<','<<calls<<'\n';returned.insert(returned.end(),s.out.result.ids.begin(),s.out.result.ids.end());}}
        save(prefix+"_ids.i64",returned);
        std::ofstream f(prefix+".json");f<<std::setprecision(17)<<"{\"status\":\"PASS\",\"workers\":"<<T<<",\"policy\":\""<<(stress?"MIXED":name)<<"\",\"replicate\":"<<rep<<",\"queries\":"<<count<<",\"elapsed_us\":"<<elapsed<<",\"aggregate_qps\":"<<count*1e6/elapsed<<",\"worker_cpu_seconds\":"<<cpu<<",\"allocated_core_utilization\":"<<cpu/(elapsed*1e-6*T)<<",\"live_process_threads_at_gate\":"<<live<<",\"exact_evals\":"<<exact_calls<<",\"model_memory_sha256\":\""<<model_hash<<"\",\"graph_fingerprint\":\""<<fingerprint<<"\",\"per_worker\":[";
        for(int w=0;w<T;++w){auto&a=workers[w];if(w)f<<',';f<<"{\"worker\":"<<w<<",\"cpu\":"<<cpus[w]<<",\"tid\":"<<a.tid<<",\"queries\":"<<a.rows.size()<<",\"cpu_seconds\":"<<a.cpu<<",\"end_offset_us\":"<<micros(start,a.end)<<",\"start_lag_us\":"<<a.start_lag_us<<",\"visited_TLS_address\":"<<a.visited_address<<",\"omp_max_threads\":"<<a.omp_threads<<'}';}f<<"]}\n";
        std::cout<<phase<<' '<<(stress?"MIXED":name)<<" T="<<T<<" rep="<<rep<<" PASS\n"<<std::flush;
    };
    if(phase=="stress"){
        require(!std::filesystem::exists(root+"stress_complete.json"),"refuse stress overwrite");
        for(int T:levels)cell(T,1,0,true);
        std::vector<double> ticks;int nt=num("timestamp_pairs");ticks.reserve(nt);auto begin=Clock::now();
        for(int i=0;i<nt;++i){auto a=Clock::now(),b=Clock::now();ticks.push_back(micros(a,b));}
        double pair_loop=micros(begin,Clock::now())/nt;save(root+"timestamp_intervals.f64",ticks);
        std::ofstream f(root+"stress_complete.json");f<<std::setprecision(17)<<"{\"status\":\"PASS\",\"calls\":"<<std::accumulate(levels.begin(),levels.end(),0)*num("stress_calls_per_worker")<<",\"two_clock_loop_us_per_iteration\":"<<pair_loop<<",\"timestamp_note\":\"includes loop/output storage; diagnostic upper bound, not subtracted\"}\n";
    }else if(phase=="benchmark"){
        require(!std::filesystem::exists(root+"benchmark_complete.json"),"refuse benchmark overwrite");
        auto order=load<int32_t>(root+"benchmark_order.i32");for(size_t i=0;i<order.size();i+=3)cell(order[i+1],order[i+2],order[i],false);
        std::ofstream f(root+"benchmark_complete.json");f<<"{\"status\":\"PASS\",\"cells\":"<<order.size()/3<<",\"all_timed_output_IDs_top16_and_counts_match\":true}\n";
    }else cell(std::stoi(argv[3]),std::stoi(argv[4]),0,false);
    std::cout<<phase<<" COMPLETE\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}return 0;}
