#!/usr/bin/env python3
"""Regenerate measured concurrent QPS, actual latency, scaling and counters."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from analyze_phase3a_sift1m import read_conf
from analyze_phase3b_fixed_candidate_ranking import dump
from phase4d_metrics import latency,summarize,scaling,relative

NAMES=['NATIVE','BOUNDED_ALL16']
def table(out,name,rows):
    dump(out/f'phase4d_{name}.json',rows)
    if rows:
        with (out/f'phase4d_{name}.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def analyze(root,c,out,fig):
    out.mkdir(parents=True,exist_ok=True);fig.mkdir(parents=True,exist_ok=True)
    old=Path(c['phase4b']);n=int(c['query_count']);passes=int(c['passes_per_repetition']);reps=int(c['repetitions']);levels=list(map(int,c['workers'].split(',')))
    gt=np.fromfile(old/'gt_ids.i64',dtype='<i8').reshape(n,11)[:,:10]
    expected={name:np.fromfile(old/f'{file}_ids.i64',dtype='<i8').reshape(n,10) for name,file in zip(NAMES,['native','all16'])}
    runs=[];worker_rows=[];timings={};total=0
    for meta in sorted((root/'benchmark').glob('*.json')):
        m=json.loads(meta.read_text());assert m['status']=='PASS';T=m['workers'];name=m['policy'];rep=m['replicate'];assert m['live_process_threads_at_gate']==T+1
        a=np.genfromtxt(meta.with_suffix('.csv'),delimiter=',',names=True);count=len(a);assert count==passes*n==m['queries']
        q=a['query_id'].astype(int);p=a['pass'].astype(int);w=a['worker'].astype(int)
        assert np.array_equal(np.sort(p*n+q),np.arange(n*passes));assert np.array_equal(w,q%T)
        ids=np.fromfile(meta.with_name(meta.stem+'_ids.i64'),dtype='<i8').reshape(count,10);assert np.array_equal(ids,expected[name][q])
        recall=np.any(ids[:,:,None]==gt[q,None,:],axis=2).sum(1)/10
        calls=16 if name=='BOUNDED_ALL16' else 0;assert np.all(a['exact_evals']==calls) and m['exact_evals']==count*calls
        assert abs(m['aggregate_qps']-count*1e6/m['elapsed_us'])<1e-9
        assert len({r['visited_TLS_address'] for r in m['per_worker']})==T
        end=np.array([r['end_offset_us'] for r in m['per_worker']]);util=m['worker_cpu_seconds']/(m['elapsed_us']*1e-6*T)
        row={'policy':name,'workers':T,'replicate':rep,'queries':count,'recall_at_10':float(recall.mean()),'qps':m['aggregate_qps'],**latency(a['latency_us']),
            'max_latency_us':float(a['latency_us'].max()),'allocated_core_utilization':util,
            'last_worker_drain_fraction':float((end.max()-end.min())/m['elapsed_us']),
            'exact_evals_per_query':calls,'nominal_extra_FP32_bytes_per_query':calls*512,'elapsed_us':m['elapsed_us']}
        runs.append(row);timings[name,T,rep]=a['latency_us'];total+=count
        for x in m['per_worker']:
            assert x['cpu']==list(map(int,c['worker_cpus'].split(',')))[x['worker']] and x['omp_max_threads']==1
            worker_rows.append({'policy':name,'workers':T,'replicate':rep,**x})
    assert len(runs)==len(levels)*2*reps
    table(out,'repetitions',runs);table(out,'workers',worker_rows)
    metrics=['recall_at_10','qps','mean_latency_us','median_latency_us','p90_latency_us','p95_latency_us','p99_latency_us','allocated_core_utilization','last_worker_drain_fraction']
    agg=[];points=[]
    for T in levels:
        for name in NAMES:
            subset=[r for r in runs if r['workers']==T and r['policy']==name];row={'policy':name,'workers':T}
            for metric in metrics:
                stats=summarize([r[metric] for r in subset]);agg.append({'policy':name,'workers':T,'metric':metric,**stats})
                row[metric]=stats['mean'];row[metric+'_std']=stats['std']
            points.append(row)
    table(out,'aggregate',agg);table(out,'endpoints',points)
    lookup=lambda name,T:next(r for r in points if r['policy']==name and r['workers']==T)
    scales=[];rel=[];paired=[];scale_reps=[]
    runmap={(r['policy'],r['workers'],r['replicate']):r for r in runs}
    for T in levels:
        for name in NAMES:
            scales.append({'policy':name,'workers':T,**scaling(lookup(name,T)['qps'],lookup(name,1)['qps'],T)})
            for rep in range(reps):scale_reps.append({'policy':name,'workers':T,'replicate':rep,**scaling(runmap[name,T,rep]['qps'],runmap[name,1,rep]['qps'],T)})
        r=relative(lookup('NATIVE',T),lookup('BOUNDED_ALL16',T));r['throughput_target_pass']=r['throughput_penalty']<=float(c['throughput_penalty_target']);rel.append({'workers':T,**r})
        for rep in range(reps):paired.append({'workers':T,'replicate':rep,**relative(runmap['NATIVE',T,rep],runmap['BOUNDED_ALL16',T,rep])})
    table(out,'scaling',scales);table(out,'scaling_repetitions',scale_reps);table(out,'relative_overhead',rel);table(out,'paired_relative',paired)
    counters=[]
    for path in sorted(root.glob('perf_*_T*.csv')):
        name,T=path.stem.removeprefix('perf_').rsplit('_T',1);meta=root/'profile'/f'{name}_T{T}_r0.json'
        if not meta.exists():continue
        m=json.loads(meta.read_text());assert m['status']=='PASS'
        for fields in csv.reader(path.read_text().splitlines()):
            if len(fields)<5 or fields[0].startswith('#'):continue
            try:value=float(fields[0]);runtime=float(fields[3]);percent=float(fields[4])
            except ValueError:continue
            event=fields[2];counters.append({'policy':name,'workers':int(T),'event':event,'unit':fields[1],'count':value,'per_query':value/m['queries'],'event_run_time_ns':runtime,'percent_running':percent,
                'primary_metric':False,'scope':'user space; warmup/load excluded by enable/disable acknowledgement; includes final thread teardown'})
    table(out,'perf_counters',counters)
    stress=json.loads((root/'stress_complete.json').read_text())
    # Generic counter values are not a measured memory bandwidth estimate.
    payload=[{'workers':T,'policy':'BOUNDED_ALL16','additional_payload_bytes_per_query':8192,'nominal_payload_GB_per_s':8192*lookup('BOUNDED_ALL16',T)['qps']/1e9,'not_measured_DRAM_bandwidth':True} for T in levels]
    table(out,'memory_payload',payload)
    summary={'query_count':n,'passes_per_cell':passes,'repetitions':reps,'primary_cells':len(runs),'primary_query_measurements':total,
        'per_policy_T_query_measurements':n*passes*reps,'all_primary_IDs_identical':True,
        'all_worker_throughput_targets_pass':all(r['throughput_target_pass'] for r in rel),
        'relative_overhead':rel,'scaling':scales,'stress':stress,
        'timestamp_loop_fraction_of_min_mean_latency':stress['two_clock_loop_us_per_iteration']/min(r['mean_latency_us'] for r in points),
        'aggregation':'mean of five repetition metrics; sample SD/min/max in aggregate; scaling ratios of mean QPS',
        'tail_interpretation':'no post-hoc numeric SLA; compare p95/p99 ratios to mean overhead and repetition spread'}
    dump(out/'phase4d_summary.json',summary);plots(fig,points,scales,rel,timings,levels,reps)
    print(json.dumps(summary,indent=2))

def plots(out,points,scales,rel,timings,levels,reps):
    import matplotlib
    matplotlib.use('Agg');matplotlib.rcParams['svg.hashsalt']='phase4d'
    import matplotlib.pyplot as plt
    def save(f,n):f.tight_layout();f.savefig(out/f'phase4d_{n}.svg',metadata={'Date':None});plt.close(f)
    for metric,name,label in [('qps','qps','Aggregate QPS'),('mean_latency_us','mean_latency','Mean query latency (us)')]:
        f,ax=plt.subplots()
        for p in NAMES:
            r=[next(r for r in points if r['policy']==p and r['workers']==t) for t in levels]
            ax.errorbar(levels,[x[metric] for x in r],yerr=[x[metric+'_std'] for x in r],marker='o',capsize=3,label=p)
        ax.set(xlabel='Pinned physical-core query workers',ylabel=label,xticks=levels);ax.legend();save(f,name)
    f,ax=plt.subplots()
    for p in NAMES:
        r=[x for x in scales if x['policy']==p];ax.plot([x['workers'] for x in r],[x['parallel_efficiency'] for x in r],marker='o',label=p)
    ax.set(xlabel='Workers',ylabel='Parallel efficiency (ratio of mean QPS)',xticks=levels);ax.legend();save(f,'efficiency')
    f,axs=plt.subplots(1,2,figsize=(10,4))
    for ax,metric in zip(axs,['p95_latency_us','p99_latency_us']):
        for p in NAMES:
            r=[next(r for r in points if r['policy']==p and r['workers']==t) for t in levels]
            ax.errorbar(levels,[x[metric] for x in r],yerr=[x[metric+'_std'] for x in r],marker='o',capsize=3,label=p)
        ax.set(xlabel='Workers',ylabel=metric,xticks=levels);ax.legend()
    save(f,'tails')
    f,ax=plt.subplots();ax.plot(levels,[x['throughput_penalty'] for x in rel],marker='o');ax.axhline(.15,color='gray',ls='--',label='15% exploratory ceiling');ax.set(xlabel='Workers',ylabel='1 - QPS_ALL16 / QPS_NATIVE',xticks=levels);ax.legend();save(f,'throughput_penalty')
    f,ax=plt.subplots()
    for t in levels:
        r=[next(r for r in points if r['workers']==t and r['policy']==p) for p in NAMES]
        ax.plot([x['qps'] for x in r],[x['recall_at_10'] for x in r],marker='o',label=f'{t} workers')
    ax.set(xlabel='Measured aggregate QPS',ylabel='Recall@10 (NATIVE lower, ALL16 upper)');ax.legend();save(f,'recall_qps')
    f,axs=plt.subplots(1,2,figsize=(10,4))
    for ax,t in zip(axs,[1,8]):
        for p in NAMES:
            a=np.concatenate([timings[p,t,r] for r in range(reps)]);qs=np.linspace(0,.999,1000);ax.plot(np.quantile(a,qs),qs,label=p)
        ax.set(xlabel='Actual query latency (us), through p99.9',ylabel='ECDF',title=f'{t} workers');ax.legend()
    save(f,'latency_distributions')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tables',default='results/tables');p.add_argument('--figures',default='results/figures');args=p.parse_args()
    c=read_conf(Path('configs/indexes/phase4d_multithread_scalability.conf'));analyze(Path(c['run']),c,Path(args.tables),Path(args.figures))
