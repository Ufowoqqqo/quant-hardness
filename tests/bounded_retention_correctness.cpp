#include "instrumentation/bounded_retention.h"
#include <iostream>
#include <map>
#include <random>
#include <set>
using namespace quant_hardness;
void check(bool v){if(!v)throw std::runtime_error("bounded retention reference mismatch");}
int main(){try{
    for(int seed=0;seed<200;++seed) {
        std::mt19937 rng(seed);BoundedTop16 heap;std::vector<RetainedCandidate> ref;std::set<int> seen;
        for(uint32_t j=0;j<3000;++j) {
            int id=rng()%600;float score=seed%3==0?1.f:float((id*17+seed)%47);
            if(seed%3==2)score=float(id);
            heap.add(id,score);if(seen.insert(id).second)ref.push_back({score,id,j});
            if(ref.size()>=16) {
                auto sorted=ref;std::sort(sorted.begin(),sorted.end(),candidate_better);auto got=heap.sorted();
                std::set<int> ids;
                for(int i=0;i<16;++i){check(got[i].id==sorted[i].id&&got[i].score==sorted[i].score&&got[i].order==sorted[i].order);ids.insert(got[i].id);}
                check(ids.size()==16&&heap.size()==16);
            }
        }
        heap.reset();check(heap.size()==0&&heap.calls()==0);bool threw=false;
        try{heap.sorted();}catch(const std::runtime_error&){threw=true;}check(threw);
    }
    std::cout<<"PASS 200 deterministic duplicate/tie/random stream references; bounded bytes="<<sizeof(BoundedTop16)<<'\n';
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}return 0;}
