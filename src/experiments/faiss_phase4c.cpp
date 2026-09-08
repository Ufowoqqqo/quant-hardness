#include "instrumentation/bounded_retention.h"
#include "graph/faiss_shared_hnsw.h"
#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>
#include <omp.h>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>

using namespace quant_hardness;
using Id=faiss::idx_t;
void require(bool b,const std::string& m){if(!b)throw std::runtime_error(m);}
template<class T> std::vector<T> load(const std::string& p) {
    size_t bytes=std::filesystem::file_size(p);require(bytes%sizeof(T)==0,"array size "+p);
    std::vector<T> a(bytes/sizeof(T));std::ifstream f(p,std::ios::binary);
    f.read(reinterpret_cast<char*>(a.data()),bytes);require(f.good(),"read "+p);return a;
}
template<class T> void save(const std::string& p,const std::vector<T>& a) {
    require(!std::filesystem::exists(p),"refuse overwrite "+p);std::ofstream f(p,std::ios::binary);
    f.write(reinterpret_cast<const char*>(a.data()),a.size()*sizeof(T));require(f.good(),"write "+p);
}
template<class T,size_t N> bool bits_equal(const std::array<T,N>& a,const std::array<T,N>& b) {return std::memcmp(a.data(),b.data(),N*sizeof(T))==0;}
int main(int argc,char**argv){try{
    require(argc==3,"usage: faiss_phase4c validate|primary|diagnostic|detail CONFIG");
    const std::string mode=argv[1];std::map<std::string,std::string> c;std::ifstream cf(argv[2]);std::string line;
    while(std::getline(cf,line)){auto pos=line.find('=');if(pos!=std::string::npos)c[line.substr(0,pos)]=line.substr(pos+1);}
    auto num=[&](auto key){return std::stoi(c.at(key));};auto root=c.at("run")+"/",old=c.at("phase4b")+"/";
    const int n=num("query_count"),d=num("dimension"),nw=num("warmup_queries");
    omp_set_num_threads(num("threads"));
    auto owner=std::unique_ptr<faiss::Index>(faiss::read_index(c.at("graph_path").c_str()));auto* graph=dynamic_cast<faiss::IndexHNSW*>(owner.get());require(graph,"HNSW required");
    auto* flat=dynamic_cast<faiss::IndexFlat*>(graph->storage);require(flat,"FP32 required");
    auto po=std::unique_ptr<faiss::Index>(faiss::read_index(c.at("model_path").c_str()));auto* pq=dynamic_cast<faiss::IndexPQ*>(po.get());require(pq,"PQ required");
    require(d==128&&num("ef_search")==64&&num("k")==10&&num("L")==16&&pq->pq.M==64&&pq->pq.nbits==8,"frozen configuration");
    const auto fp=graph_fingerprint(graph->hnsw);require(fp==c.at("graph_fingerprint"),"fingerprint mismatch");
    auto queries=load<float>(old+"queries.f32"),warm=load<float>(old+"warmup.f32");require(queries.size()==size_t(n)*d&&warm.size()==size_t(nw)*d,"query size");
    auto native=load<Id>(old+"native_ids.i64"),all=load<Id>(old+"all16_ids.i64"),gapids=load<Id>(old+"gap_ids.i64");
    auto gaps=load<double>(old+"gaps.f64");auto decisions=load<uint8_t>(old+"gap_decisions.u8");auto offsets=load<int64_t>(old+"candidate_offsets.i64");
    CandidateAccess full(*graph,*pq,flat->get_xb(),std::stod(c.at("gap_threshold")),64,true);
    CandidateAccess bounded(*graph,*pq,flat->get_xb(),std::stod(c.at("gap_threshold")),64,false);
    require(bounded.full_scratch_bytes()==0,"bounded allocated full storage");
    const auto full_bytes=full.full_scratch_bytes();
    auto engine=[&](int m)->CandidateAccess&{return m==1||m==3?full:bounded;};
    if(mode=="validate") {
        require(!std::filesystem::exists(root+"validation.json"),"refuse validation overwrite");
        StreamingRefinement legacy(*graph,*pq,flat->get_xb(),std::stod(c.at("gap_threshold")),64);
        auto old_ids=load<int32_t>(old+"candidate_ids.i32");auto old_scores=load<float>(old+"candidate_pq.f32");
        std::vector<Id> top_ids;std::vector<uint32_t> top_order;std::vector<float> top_scores;
        std::ofstream rows(root+"validation_queries.csv");rows<<"query_id,candidates,pq_rank16_boundary_tie,native_equal,top16_equal,all16_equal,gap_equal\n";
        size_t boundary_ties=0;
        for(int q=0;q<n;++q) {
            const float* x=queries.data()+size_t(q)*d;
            auto reference=legacy.run(x,StreamPolicy::All,false,false);
            auto f=full.run(x,AccessMode::FullAll,0);
            require(bits_equal(reference.native_ids,f.result.native_ids)&&bits_equal(reference.native_scores,f.result.native_scores),"legacy native differs");
            require(bits_equal(reference.top_ids,f.result.top_ids)&&bits_equal(reference.top_order,f.result.top_order)&&bits_equal(reference.exact,f.result.exact)&&bits_equal(reference.ids,f.result.ids),"legacy ALL16 differs");
            size_t count=offsets[q+1]-offsets[q];require(full.ids()==legacy.candidate_ids&&full.scores()==legacy.candidate_scores,"legacy population differs");
            require(full.ids().size()==count&&std::equal(full.ids().begin(),full.ids().end(),old_ids.begin()+offsets[q])&&std::equal(full.scores().begin(),full.scores().end(),old_scores.begin()+offsets[q]),"stored4B population differs");
            std::vector<size_t> order(count);std::iota(order.begin(),order.end(),0);
            std::stable_sort(order.begin(),order.end(),[&](size_t a,size_t b){return full.scores()[a]<full.scores()[b];});
            for(int j=0;j<16;++j)require(f.result.top_ids[j]==full.ids()[order[j]]&&f.result.top_order[j]==order[j]&&f.top_scores[j]==full.scores()[order[j]],"full-sort reference differs");
            const bool tied=count>16&&full.scores()[order[15]]==full.scores()[order[16]];boundary_ties+=tied;
            for(int clock=0;clock<=2;++clock) {
                for(int m=0;m<6;++m) {
                    auto r=engine(m).run(x,AccessMode(m),clock);auto& s=r.result;
                    require(bits_equal(s.native_ids,f.result.native_ids)&&bits_equal(s.native_scores,f.result.native_scores),"instrumentation changed native bits");
                    if(m!=0)require(bits_equal(s.top_ids,f.result.top_ids)&&bits_equal(s.top_order,f.result.top_order)&&bits_equal(r.top_scores,f.top_scores)&&r.candidates==count,"bounded top16/population/clock mismatch");
                    const auto& expected=m==3||m==4?all:m==5?gapids:native;
                    require(std::equal(s.ids.begin(),s.ids.end(),expected.begin()+size_t(q)*10),"stored4B outputs differ");
                    if(m==3||m==4)require(bits_equal(s.exact,f.result.exact),"exact16 bits differ");
                    if(m==5)require(s.gap==gaps[q]&&s.refined==bool(decisions[q]),"frozen gap differs");
                }
            }
            require(full.full_scratch_bytes()==full_bytes&&bounded.full_scratch_bytes()==0,"candidate storage growth");
            top_ids.insert(top_ids.end(),f.result.top_ids.begin(),f.result.top_ids.end());top_order.insert(top_order.end(),f.result.top_order.begin(),f.result.top_order.end());top_scores.insert(top_scores.end(),f.top_scores.begin(),f.top_scores.end());
            rows<<q<<','<<count<<','<<tied<<",1,1,1,1\n";
        }
        save(root+"top16_ids.i64",top_ids);save(root+"top16_order.u32",top_order);save(root+"top16_scores.f32",top_scores);
        require(graph_fingerprint(graph->hnsw)==fp,"validation graph changed");
        std::ofstream f(root+"validation.json");f<<"{\"status\":\"PASS\",\"queries\":"<<n<<",\"mode_clock_checks\":"<<n*18<<",\"all_native_bits_identical\":true,\"all_top16_ordered_ID_score_bits_identical\":true,\"all_refined_output_IDs_identical\":true,\"all_legacy_pool_order_scores_identical\":true,\"rank16_boundary_tie_queries\":"<<boundary_ties<<",\"full_scratch_bytes\":"<<full_bytes<<",\"bounded_heap_bytes\":"<<sizeof(BoundedTop16)<<",\"candidate_capacity_growth_queries\":0,\"graph_fingerprint\":\""<<fp<<"\"}\n";
    } else {
        require(mode=="primary"||mode=="diagnostic"||mode=="detail","unknown mode");
        require(std::filesystem::exists(root+"validation.json"),"correctness gate required");
        if(mode!="primary")require(std::filesystem::exists(root+"primary_complete.json"),"primary must finish first");
        require(!std::filesystem::exists(root+mode+"_complete.json"),"refuse benchmark overwrite");
        auto order=load<int32_t>(root+mode+"_order.i32");auto tops=load<Id>(root+"top16_ids.i64");auto toporder=load<uint32_t>(root+"top16_order.u32");auto topscores=load<float>(root+"top16_scores.f32");
        std::filesystem::create_directories(root+mode);
        for(size_t i=0;i<order.size();i+=3) {
            int rep=order[i],m=order[i+1],clock=order[i+2];auto& e=engine(m);
            const auto prefix=root+mode+"/"+access_names[m]+"_r"+std::to_string(rep)+"_c"+std::to_string(clock);
            require(!std::filesystem::exists(prefix+".csv"),"refuse timing overwrite");
            for(int q=0;q<nw;++q)e.run(warm.data()+size_t(q)*d,AccessMode(m),clock);
            std::vector<AccessOutput> output;output.reserve(n);
            auto start=std::chrono::steady_clock::now();
            for(int q=0;q<n;++q)output.push_back(e.run(queries.data()+size_t(q)*d,AccessMode(m),clock));
            auto end=std::chrono::steady_clock::now();
            // No labels, equivalence comparisons, or file writes in measured stream.
            for(int q=0;q<n;++q) {
                auto& r=output[q];auto& s=r.result;const auto& expected=m==3||m==4?all:m==5?gapids:native;
                require(std::equal(s.ids.begin(),s.ids.end(),expected.begin()+size_t(q)*10),"timed output changed");
                require(std::equal(s.native_ids.begin(),s.native_ids.end(),native.begin()+size_t(q)*10),"timed native changed");
                if(m!=0)require(std::equal(s.top_ids.begin(),s.top_ids.end(),tops.begin()+size_t(q)*16)&&std::equal(s.top_order.begin(),s.top_order.end(),toporder.begin()+size_t(q)*16)&&std::equal(r.top_scores.begin(),r.top_scores.end(),topscores.begin()+size_t(q)*16)&&r.candidates==size_t(offsets[q+1]-offsets[q]),"timed top16 changed");
                if(m==5)require(s.gap==gaps[q]&&s.refined==bool(decisions[q]),"timed gap changed");
                require(s.refined==(m==3||m==4||(m==5&&decisions[q])),"timed refinement changed");
            }
            require(full.full_scratch_bytes()==full_bytes&&bounded.full_scratch_bytes()==0,"timed capacity growth");
            std::ofstream f(prefix+".csv");f<<"query_id,total_us,t_search_including_retention,t_candidate_retention_intrusive,t_candidate_finalize,t_exact16,t_final_topk,candidates,refined,exact_evals,gap\n"<<std::setprecision(17);
            std::vector<Id> returned;returned.reserve(size_t(n)*10);
            for(int q=0;q<n;++q){auto&r=output[q];auto&s=r.result;f<<q<<','<<s.total_us<<','<<s.search_us<<','<<r.retention_us<<','<<s.gap_us<<','<<s.refine_us<<','<<s.final_us<<','<<r.candidates<<','<<s.refined<<','<<16*s.refined<<','<<s.gap<<'\n';returned.insert(returned.end(),s.ids.begin(),s.ids.end());}
            save(prefix+"_ids.i64",returned);
            double stream=std::chrono::duration<double,std::micro>(end-start).count();std::ofstream sf(prefix+"_stream.json");sf<<std::setprecision(17)<<"{\"stream_us\":"<<stream<<",\"stream_qps\":"<<n*1e6/stream<<",\"full_scratch_bytes\":"<<e.full_scratch_bytes()<<",\"bounded_heap_bytes\":"<<sizeof(BoundedTop16)<<",\"verified_outputs\":true}\n";
            std::cout<<mode<<' '<<access_names[m]<<" rep="<<rep<<" clock="<<clock<<" complete\n"<<std::flush;
        }
        require(graph_fingerprint(graph->hnsw)==fp,"timed graph changed");
        std::ofstream f(root+mode+"_complete.json");f<<"{\"status\":\"PASS\",\"runs\":"<<order.size()/3<<",\"outputs_top16_verified\":true,\"graph_fingerprint\":\""<<fp<<"\"}\n";
    }
    std::cout<<mode<<" PASS\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}return 0;}
