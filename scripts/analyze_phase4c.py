#!/usr/bin/env python3
"""Regenerate Phase4C summaries/figures solely from frozen raw runs."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import numpy as np
from analyze_phase3a_sift1m import read_conf
from analyze_phase3b_fixed_candidate_ranking import dump
from phase4c_metrics import latency_summary,gate,selective_gate

NAMES=['NATIVE','FULL','BOUNDED','FULL_ALL16','BOUNDED_ALL16','GAP_BOUNDED']
def table(out,name,rows):
    dump(out/f'phase4c_{name}.json',rows)
    with (out/f'phase4c_{name}.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def analyze(root,c,out,fig,derived):
    out.mkdir(parents=True,exist_ok=True);fig.mkdir(parents=True,exist_ok=True);derived.mkdir(parents=True,exist_ok=True)
    old=Path(c['phase4b']);n=int(c['query_count']);reps=int(c['repetitions']);counts=np.diff(np.fromfile(old/'candidate_offsets.i64',dtype='<i8'))
    gt=np.fromfile(old/'gt_ids.i64',dtype='<i8').reshape(n,11)[:,:10]
    arrays={name:np.fromfile(old/f'{file}_ids.i64',dtype='<i8').reshape(n,10) for name,file in [('NATIVE','native'),('ALL','all16'),('GAP','gap')]}
    recalls={name:np.array([len(set(a[q])&set(gt[q]))/10 for q in range(n)]) for name,a in arrays.items()}
    decision=np.fromfile(old/'gap_decisions.u8',dtype='u1');gaps=np.fromfile(old/'gaps.f64',dtype='<f8')
    runs={};repetition_rows=[];component_rows=[];intrusive=[]
    for phase in ['primary','diagnostic','detail']:
        assert json.loads((root/f'{phase}_complete.json').read_text())['status']=='PASS'
        for file in sorted((root/phase).glob('*.csv')):
            name,rc=file.stem.rsplit('_r',1);rep,clock=map(int,rc.split('_c'))
            a=np.genfromtxt(file,delimiter=',',names=True);assert np.array_equal(a['query_id'],np.arange(n))
            expected=arrays['ALL' if name.endswith('ALL16') else 'GAP' if name=='GAP_BOUNDED' else 'NATIVE']
            got=np.fromfile(file.with_name(file.stem+'_ids.i64'),dtype='<i8').reshape(n,10);assert np.array_equal(got,expected)
            refine=np.ones(n) if name.endswith('ALL16') else decision if name=='GAP_BOUNDED' else np.zeros(n)
            assert np.array_equal(a['refined'],refine) and np.array_equal(a['exact_evals'],16*refine)
            if name!='NATIVE':assert np.array_equal(a['candidates'],counts)
            if name=='GAP_BOUNDED':assert np.array_equal(a['gap'],gaps)
            runs[phase,name,rep,clock]=a
            stream=json.loads(file.with_name(file.stem+'_stream.json').read_text())
            repetition_rows.append({'phase':phase,'mode':name,'replicate':rep,'clock_level':clock,**latency_summary(a['total_us']),
                'stream_qps':stream['stream_qps'],'full_scratch_bytes':stream['full_scratch_bytes']})
            if clock:
                cols=['t_search_including_retention','t_candidate_finalize','t_exact16','t_final_topk']
                assert np.allclose(sum(a[key] for key in cols),a['total_us'],rtol=0,atol=1e-8)
                for key in cols:component_rows.append({'phase':phase,'mode':name,'replicate':rep,'clock_level':clock,'component':key,'mean_us':float(a[key].mean())})
            if clock==2:
                intrusive.append({'mode':name,'mean_total_us_intrusive':float(a['total_us'].mean()),
                    'mean_retention_callbacks_us_including_inner_clock_cost':float(a['t_candidate_retention_intrusive'].mean()),
                    'search_minus_measured_retention_us_NOT_counterfactual_core':float((a['t_search_including_retention']-a['t_candidate_retention_intrusive']).mean()),
                    'stage1_primary_mean_us':float(np.mean([r['mean_us'] for r in repetition_rows if r['phase']=='primary' and r['mode']==name and r['clock_level']==1])),
                    'timer_note':'per-callback timers perturb search; not used for optimization gate'})
    table(out,'repetitions',repetition_rows);table(out,'components',component_rows);table(out,'intrusive_diagnostic',intrusive)
    policies=[];clock_rows=[];pooled={}
    for phase,modes in [('primary',NAMES[:5]),('diagnostic',['NATIVE','BOUNDED_ALL16','GAP_BOUNDED'])]:
        for name in modes:
            a=np.concatenate([runs[phase,name,r,0]['total_us'] for r in range(reps)]);pooled[phase,name]=a
            means=[float(runs[phase,name,r,0]['total_us'].mean()) for r in range(reps)]
            key='ALL' if name.endswith('ALL16') else 'GAP' if name=='GAP_BOUNDED' else 'NATIVE'
            nr=n if key=='ALL' else int(decision.sum()) if key=='GAP' else 0
            policies.append({'phase':phase,'mode':name,'recall_at_10':float(recalls[key].mean()),**latency_summary(a),
                'rep_mean_std_us':float(np.std(means,ddof=1)),'rep_mean_min_us':min(means),'rep_mean_max_us':max(means),
                'stream_qps_mean':float(np.mean([r['stream_qps'] for r in repetition_rows if r['phase']==phase and r['mode']==name and r['clock_level']==0])),
                'refined_queries':nr,'refined_fraction':nr/n,'exact_evaluations':16*nr,'exact_evaluations_per_query':16*nr/n})
            diff=[float(runs[phase,name,r,1]['total_us'].mean()-runs[phase,name,r,0]['total_us'].mean()) for r in range(reps)]
            clock_rows.append({'phase':phase,'mode':name,'stage_ON_minus_OFF_mean_us':float(np.mean(diff)),'paired_difference_std_us':float(np.std(diff,ddof=1)),
                'difference_fraction_of_OFF':float(np.mean(diff)/a.mean())})
    table(out,'policies',policies);table(out,'clock_overhead',clock_rows)
    mean=lambda phase,name:next(r['mean_us'] for r in policies if r['phase']==phase and r['mode']==name)
    primary=gate(*(mean('primary',name) for name in NAMES[:5]),overhead_target=float(c['retention_overhead_target']),saving_target=float(c['all16_latency_saving_target']))
    diag=selective_gate(mean('diagnostic','GAP_BOUNDED'),mean('diagnostic','BOUNDED_ALL16'),*(float(recalls[k].mean()) for k in ['NATIVE','GAP','ALL']),saving_target=float(c['gap_latency_saving_target']),gain_target=float(c['gap_gain_retained_target']))
    gates=[]
    for rep in range(reps):
        g=gate(*(float(runs['primary',name,rep,0]['total_us'].mean()) for name in NAMES[:5]))
        gates.append({'replicate':rep,**g})
    table(out,'paired_savings',gates)
    frontier=[]
    for name in ['NATIVE','GAP_BOUNDED','BOUNDED_ALL16']:
        row=next(r for r in policies if r['phase']=='diagnostic' and r['mode']==name)
        dominated=[r['mode'] for r in policies if r['phase']=='diagnostic' and r['mode']!=name and r['mean_us']<=row['mean_us'] and r['recall_at_10']>=row['recall_at_10'] and (r['mean_us']<row['mean_us'] or r['recall_at_10']>row['recall_at_10'])]
        frontier.append({'mode':name,'dominated_by':dominated})
    val=json.loads((root/'validation.json').read_text())
    memory={'candidate_count_mean':float(counts.mean()),'candidate_count_median':float(np.median(counts)),
        'candidate_count_p95':float(np.quantile(counts,.95)),'candidate_count_min':int(counts.min()),'candidate_count_max':int(counts.max()),
        'full_mean_ID_score_bytes':float(counts.mean()*8),'full_mean_ID_score_order_bytes':float(counts.mean()*12),
        'full_capacity_bytes':val['full_scratch_bytes'],'full_tag_bytes':4000000,'full_capacity_per_array':2048,
        'bounded_heap_bytes':val['bounded_heap_bytes'],'bounded_top16_sorted_copy_bytes':192,'bounded_retained_IDs':16,'bounded_retained_scores':16,
        'incremental_collector_dynamic_allocations_per_query_FULL':0,'incremental_collector_dynamic_allocations_per_query_BOUNDED':0,
        'allocation_note':'source accounting plus zero capacity growth; not global malloc profiling; shared FAISS/DC allocations remain',
        'FP32_refinement_bytes_per_all_query':16*128*4,'FP32_side_database_bytes':1000000*128*4}
    dump(out/'phase4c_memory.json',memory)
    summary={'primary':primary,'secondary_frozen_gap':diag,'secondary_frontier':frontier,
        'validation':val,'query_count':n,'primary_clock_level':0,'all_query_outputs_identical':True,
        'timed_traces':int(sum(len(a) for a in runs.values())),'candidate_memory':memory}
    dump(out/'phase4c_summary.json',summary)
    with gzip.open(derived/'queries.jsonl.gz','wt') as f:
        for q in range(n):
            r={'query_id':q,'candidate_count':int(counts[q]),'gap':float(gaps[q]),'gap_refined':bool(decision[q]),
                **{k.lower()+'_recall':float(v[q]) for k,v in recalls.items()}}
            for phase,modes in [('primary',NAMES[:5]),('diagnostic',['NATIVE','BOUNDED_ALL16','GAP_BOUNDED'])]:
                for name in modes:r[phase+'_'+name.lower()+'_mean_us']=float(np.mean([runs[phase,name,i,0]['total_us'][q] for i in range(reps)]))
            f.write(json.dumps(r)+'\n')
    plot(fig,policies,component_rows,pooled,primary,diag,counts)
    print(json.dumps(summary,indent=2))

def plot(out,policies,components,pooled,primary,diag,counts):
    import matplotlib
    matplotlib.use('Agg');matplotlib.rcParams['svg.hashsalt']='phase4c'
    import matplotlib.pyplot as plt
    def save(f,name):
        f.tight_layout();f.savefig(out/f'phase4c_{name}.svg',metadata={'Date':None});plt.close(f)
    p=[r for r in policies if r['phase']=='primary']
    f,ax=plt.subplots(figsize=(9,4));ax.bar([r['mode'] for r in p],[r['mean_us'] for r in p],yerr=[r['rep_mean_std_us'] for r in p]);ax.set(ylabel='Mean latency (us); repeat SD');ax.tick_params(axis='x',rotation=20);save(f,'access_latency')
    for field,name in [('mean_us','recall_mean_latency'),('p95_us','recall_p95_latency')]:
        f,ax=plt.subplots()
        for r in policies:
            if r['phase']!='diagnostic':continue
            ax.scatter(r[field],r['recall_at_10']);ax.annotate(r['mode'],(r[field],r['recall_at_10']),xytext=(3,4),textcoords='offset points')
        ax.set(xlabel=field,ylabel='Recall@10 (contemporaneous GAP diagnostic)');save(f,name)
    f,ax=plt.subplots()
    for r in p:
        quant=np.linspace(0,.999,1000);ax.plot(np.quantile(pooled['primary',r['mode']],quant),quant,label=r['mode'])
    ax.set(xlabel='Latency (us), plotted through p99.9',ylabel='ECDF');ax.legend();save(f,'latency_distributions')
    f,ax=plt.subplots(figsize=(9,4));bottom=np.zeros(5)
    for key in ['t_search_including_retention','t_candidate_finalize','t_exact16','t_final_topk']:
        vals=[np.mean([r['mean_us'] for r in components if r['phase']=='primary' and r['clock_level']==1 and r['mode']==name and r['component']==key]) for name in NAMES[:5]]
        ax.bar(NAMES[:5],vals,bottom=bottom,label=key);bottom+=vals
    ax.set(ylabel='Stage-clock diagnostic mean (us)');ax.tick_params(axis='x',rotation=20);ax.legend(fontsize=7);save(f,'components')
    f,ax=plt.subplots();ax.bar(['Full overhead','Bounded overhead'],[primary['full_overhead_us'],primary['bounded_overhead_us']]);ax.axhline(0,color='gray');ax.set(ylabel='Mean latency minus NATIVE (us)');save(f,'overhead')
    f,ax=plt.subplots();ax.hist(counts,bins=40);ax.axvline(16,color='red',label='bounded16');ax.set(xlabel='Unique L0 evaluated candidates/query',ylabel='Queries');ax.legend();save(f,'candidate_counts')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tables',default='results/tables');p.add_argument('--figures',default='results/figures');p.add_argument('--derived');args=p.parse_args()
    c=read_conf(Path('configs/indexes/phase4c_bounded_candidate_retention.conf'));root=Path(c['run'])
    analyze(root,c,Path(args.tables),Path(args.figures),Path(args.derived) if args.derived else root/'analysis')
