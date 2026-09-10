#pragma once
#include "instrumentation/bounded_retention.h"
#include "graph/faiss_shared_hnsw.h"
#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/utils/distances.h>
#include <openssl/evp.h>
#include <omp.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <set>
#include <vector>
namespace p5 {
using namespace quant_hardness;
using Id=faiss::idx_t; using Clock=std::chrono::steady_clock;
inline void req(bool ok,const std::string& s){if(!ok)throw std::runtime_error(s);}
inline double seconds(Clock::time_point a){return std::chrono::duration<double>(Clock::now()-a).count();}
inline std::map<std::string,std::string> config(const char* path){std::map<std::string,std::string> c;std::ifstream f(path);std::string s;while(std::getline(f,s)){if(s.empty()||s[0]=='#')continue;auto p=s.find('=');if(p!=std::string::npos)c[s.substr(0,p)]=s.substr(p+1);}return c;}
struct Mapped {
 int fd=-1; size_t bytes=0; void* ptr=nullptr;
 explicit Mapped(const std::string& path){bytes=std::filesystem::file_size(path);fd=open(path.c_str(),O_RDONLY);req(fd>=0,"open "+path);ptr=mmap(nullptr,bytes,PROT_READ,MAP_SHARED,fd,0);req(ptr!=MAP_FAILED,"mmap");}
 ~Mapped(){if(ptr&&ptr!=MAP_FAILED)munmap(ptr,bytes);if(fd>=0)close(fd);}
 const float* data() const{return static_cast<const float*>(ptr);}
};
template<class T> std::vector<T> load(const std::string& path){auto n=std::filesystem::file_size(path);req(n%sizeof(T)==0,"size");std::vector<T> a(n/sizeof(T));std::ifstream f(path,std::ios::binary);f.read(reinterpret_cast<char*>(a.data()),n);req(f.good(),"read "+path);return a;}
template<class T> void save(const std::string& path,const std::vector<T>& a){req(!std::filesystem::exists(path),"refuse overwrite "+path);std::ofstream f(path,std::ios::binary);f.write(reinterpret_cast<const char*>(a.data()),a.size()*sizeof(T));req(f.good(),"write "+path);}
inline std::ofstream output(const std::string& p){req(!std::filesystem::exists(p),"refuse overwrite "+p);std::ofstream f(p);f<<std::setprecision(17);return f;}
inline std::string hash_bytes(const void* data,size_t size){unsigned char hash[32];unsigned int n;EVP_Digest(data,size,hash,&n,EVP_sha256(),nullptr);std::ostringstream s;s<<std::hex<<std::setfill('0');for(unsigned i=0;i<n;++i)s<<std::setw(2)<<int(hash[i]);return s.str();}
inline std::string file_hash(const std::string& p){Mapped m(p);return hash_bytes(m.ptr,m.bytes);}
inline void pin(int cpu){cpu_set_t m;CPU_ZERO(&m);CPU_SET(cpu,&m);req(pthread_setaffinity_np(pthread_self(),sizeof(m),&m)==0,"pin");}
inline double recall(const Id* a,const Id* gt){int n=0;for(int i=0;i<10;++i)for(int j=0;j<10;++j)if(a[i]==gt[j]){++n;break;}return n/10.0;}
inline std::vector<Id> rerank(const float* base,const float* q,int d,const std::vector<int32_t>& ids){std::vector<std::pair<float,size_t>> ranked;for(size_t j=0;j<ids.size();++j)ranked.emplace_back(faiss::fvec_L2sqr(q,base+size_t(ids[j])*d,d),j);req(ranked.size()>=10,"short oracle");std::partial_sort(ranked.begin(),ranked.begin()+10,ranked.end());std::vector<Id> result;for(int j=0;j<10;++j)result.push_back(ids[ranked[j].second]);return result;}
inline void architecture(const faiss::IndexHNSW& graph,const faiss::IndexPQ& pq,int m=768){req((m==768||m==384)&&graph.d==1536&&graph.ntotal==990000&&pq.d==1536&&pq.ntotal==990000&&pq.pq.M==m&&pq.pq.nbits==8&&pq.code_size==size_t(m),"frozen architecture mismatch");}
}
