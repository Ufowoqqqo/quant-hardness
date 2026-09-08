#!/usr/bin/env python3
"""Regenerate Phase4B policy, timing, component, memory and batch reports."""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np
from analyze_phase3a_sift1m import read_conf,spearman,write_csv
from analyze_phase3b_fixed_candidate_ranking import dump,digest
from phase4b_metrics import recall,latency_stats,systems_metrics
from phase4a_tie_sensitivity import tie_hits

NAMES=['NATIVE','GAP','RANDOM','ALL16']

def table(out,name,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r));rows=[{k:r.get(k) for k in keys} for r in rows]
    write_csv(out/f'phase4b_{name}.csv',rows);dump(out/f'phase4b_{name}.json',rows)


def analyze(root,c,out,fig,derived):
    assert json.loads((root/'benchmark_complete.json').read_text())['status']=='PASS'
    assert json.loads((root/'batch_complete.json').read_text())['status']=='PASS'
    for p,h in json.loads((root/'validation.json').read_text())['sha256'].items():assert digest(p)==h,p
    for path in [out,fig,derived]:path.mkdir(parents=True,exist_ok=True)
    n=int(c['query_count']);reps=int(c['repetitions']);gt=np.fromfile(root/'gt_ids.i64',dtype='<i8').reshape(n,11)
    gd=np.fromfile(root/'gt_distances.f32',dtype='<f4').reshape(n,11);gaps=np.fromfile(root/'gaps.f64',dtype='<f8');dec=np.fromfile(root/'gap_decisions.u8',dtype='u1').astype(bool)
    masks=np.fromfile(root/'random_masks.u8',dtype='u1').reshape(int(c['random_replicates']),n).astype(bool)
    idarrays={name:np.fromfile(root/file,dtype='<i8').reshape(n,10) for name,file in [('NATIVE','native_ids.i64'),('GAP','gap_ids.i64'),('ALL16','all16_ids.i64'),('ORACLE','oracle_ids.i64')]}
    idarrays['RANDOM']=np.where(masks[0,:,None],idarrays['ALL16'],idarrays['NATIVE'])
    recalls={key:np.array([recall(x,t) for x,t in zip(ids,gt)]) for key,ids in idarrays.items()}
    loss=np.round(recalls['ORACLE']-recalls['NATIVE'],10);improve=np.round(recalls['GAP']-recalls['NATIVE'],10)
    rr=np.where(masks,recalls['ALL16'][None,:],recalls['NATIVE'][None,:]).mean(1)
    random_rows=[{'replicate':i,'recall':float(x),'refined_queries':int(masks[i].sum()),'exact_evals':int(masks[i].sum())*16} for i,x in enumerate(rr)]
    table(out,'random_replicates',random_rows)
    timing={};repetition_rows=[];component_rows=[]
    for name in NAMES:
        for clocks in [0,1]:
            traces=[]
            for rep in range(reps):
                prefix=root/'timings'/f'{name}_r{rep}_c{clocks}';trace=np.genfromtxt(str(prefix)+'.csv',delimiter=',',names=True)
                assert np.array_equal(trace['query_id'],np.arange(n));expected=dec if name=='GAP' else masks[0] if name=='RANDOM' else np.full(n,name=='ALL16')
                assert np.array_equal(trace['refined'],expected) and np.array_equal(trace['exact_evals'],expected*16)
                result=np.fromfile(str(prefix)+'_ids.i64',dtype='<i8').reshape(n,10);assert np.array_equal(result,idarrays[name])
                if name=='GAP':assert np.array_equal(trace['gap'],gaps)
                stream=json.loads(Path(str(prefix)+'_stream.json').read_text())
                stats=latency_stats(trace['total_us']);repetition_rows.append({'policy':name,'components':clocks,'replicate':rep,**stats,
                    'stream_qps':stream['stream_qps'],'stream_us_per_query':stream['stream_us']/n,'scratch_bytes':stream['scratch_bytes']})
                if clocks:
                    total=sum(trace[k] for k in ['t_search','t_gap','t_refine','t_final_selection'])
                    assert np.allclose(total,trace['total_us'],atol=1e-9,rtol=1e-12)
                    for key in ['t_search','t_gap','t_refine','t_final_selection']:
                        component_rows.append({'policy':name,'replicate':rep,'component':key,'mean_us':float(trace[key].mean()),
                            'fraction_of_total':float(trace[key].sum()/trace['total_us'].sum())})
                traces.append(trace)
            timing[name,clocks]=traces
    table(out,'latency_repetitions',repetition_rows);table(out,'components',component_rows)
    policy_rows=[];overhead=[];pooled={}
    for name in NAMES:
        values=np.concatenate([t['total_us'] for t in timing[name,0]]);pooled[name]=values;stats=latency_stats(values)
        mean0=np.array([float(t['total_us'].mean()) for t in timing[name,0]]);mean1=np.array([float(t['total_us'].mean()) for t in timing[name,1]])
        nr=0 if name=='NATIVE' else n if name=='ALL16' else int(dec.sum())
        stream=[r['stream_qps'] for r in repetition_rows if r['policy']==name and r['components']==0]
        policy_rows.append({'policy':name,'recall':float(recalls[name].mean()),'refined_queries':nr,'refined_fraction':nr/n,
            'exact_evals':nr*16,'exact_evals_per_query':nr*16/n,**stats,
            'rep_mean_std_us':float(mean0.std(ddof=1)),'rep_mean_min_us':float(mean0.min()),'rep_mean_max_us':float(mean0.max()),
            'mean_stream_qps':float(np.mean(stream)),
            'recall_scope':'fixed random mask0 (not mean of30)' if name=='RANDOM' else 'deterministic',
            'random_recall_mean':float(rr.mean()) if name=='RANDOM' else None,
            'random_recall_p025':float(np.quantile(rr,.025)) if name=='RANDOM' else None,
            'random_recall_p975':float(np.quantile(rr,.975)) if name=='RANDOM' else None})
        overhead.append({'policy':name,'component_off_mean_us':float(mean0.mean()),'component_on_mean_us':float(mean1.mean()),
            'on_minus_off_us':float(mean1.mean()-mean0.mean()),'relative_on_minus_off':float(mean1.mean()/mean0.mean()-1),
            'paired_rep_difference_std_us':float((mean1-mean0).std(ddof=1))})
    # Oracle is not timed and must not enter the latency frontier as a point.
    counts=np.diff(np.fromfile(root/'candidate_offsets.i64',dtype='<i8'))
    policy_rows.append({'policy':'ORACLE-offline','recall':float(recalls['ORACLE'].mean()),'exact_evals':int(counts.sum()),'exact_evals_per_query':float(counts.mean()),'latency_excluded':True})
    table(out,'policies',policy_rows);table(out,'clock_overhead',overhead)
    risk=[]
    for selected in [True,False]:
        m=dec==selected;risk.append({'group':'refined' if selected else 'unrefined','queries':int(m.sum()),
            'mean_ranking_loss':float(loss[m].mean()),'harmful_query_fraction':float((loss[m]>0).mean()),
            'native_recall':float(recalls['NATIVE'][m].mean()),'oracle_recall':float(recalls['ORACLE'][m].mean())})
    table(out,'threshold_risk',risk)
    classification=[{'class':key,'queries':int((dec&condition).sum()),'fraction_selected':float(condition[dec].mean())} for key,condition in [('useful',improve>0),('neutral',improve==0),('harmful',improve<0)]]
    table(out,'selected_outcomes',classification)
    pg=next(r for r in policy_rows if r['policy']=='GAP');pa=next(r for r in policy_rows if r['policy']=='ALL16')
    sys=systems_metrics(float(recalls['NATIVE'].mean()),pg['recall'],pa['recall'],pg['mean_us'],pa['mean_us'],int(dec.sum()),n)
    rep_savings=[]
    for rep in range(reps):
        g=timing['GAP',0][rep]['total_us'].mean();a=timing['ALL16',0][rep]['total_us'].mean()
        rep_savings.append({'replicate':rep,'latency_saving':float(1-g/a)})
    table(out,'latency_saving_repetitions',rep_savings)
    dominated=[]
    for metric in ['mean_us','p95_us']:
        competitors=[r for r in policy_rows if r['policy'] in ['NATIVE','RANDOM','ALL16']]
        for r in competitors:
            if r[metric]<=pg[metric] and r['recall']>=pg['recall'] and (r[metric]<pg[metric] or r['recall']>pg['recall']):dominated.append({'metric':metric,'dominating_policy':r['policy']})
    summary={'queries':n,'frozen_threshold':float(c['gap_threshold']),'actual_refined_queries':int(dec.sum()),'actual_refined_fraction':float(dec.mean()),
        'deviation_from_25_percentage_points':float((dec.mean()-.25)*100),
        'gap_beats_random_mean':bool(pg['recall']>rr.mean()),'gap_above_random_p975':bool(pg['recall']>np.quantile(rr,.975)),
        'gap_minus_random_mean_recall':float(pg['recall']-rr.mean()),
        'gap_loss_spearman':spearman(gaps.tolist(),loss.tolist()),**sys,
        'useful_systems_signal':bool(pg['recall']>rr.mean() and sys['latency_saving']>=.2 and sys['all16_gain_retained']>=.6),
        'gap_dominated_by':dominated,'mean_latency_saving_rep_std':float(np.std([r['latency_saving'] for r in rep_savings],ddof=1)),
        'primary_timing':'component clocks OFF, pooled equal-length7 repetitions; on-clock measurements separate',
        'timed_random_mask':0,'GT_tie_query_count':int((gd[:,9]==gd[:,10]).sum())}
    dump(out/'phase4b_summary.json',summary)
    gaplat=np.array([t['total_us'] for t in timing['GAP',0]])
    with gzip.open(derived/'queries.jsonl.gz','wt') as f:
        for q in range(n):f.write(json.dumps({'query_id':q,'gap':float(gaps[q]),'refined':bool(dec[q]),
            'native_recall':float(recalls['NATIVE'][q]),'gap_recall':float(recalls['GAP'][q]),'all16_recall':float(recalls['ALL16'][q]),'oracle_recall':float(recalls['ORACLE'][q]),
            'ranking_loss':float(loss[q]),'gain':float(improve[q]),'outcome':'unrefined' if not dec[q] else 'useful' if improve[q]>0 else 'harmful' if improve[q]<0 else 'neutral',
            'gap_latency_mean_us':float(gaplat[:,q].mean()),'gap_latency_std_us':float(gaplat[:,q].std(ddof=1))})+'\n')
    # Explicit tie-aware sensitivity; never change primary ID-based metrics.
    offsets=np.fromfile(root/'candidate_offsets.i64',dtype='<i8');ci=np.memmap(root/'candidate_ids.i32',dtype='<i4',mode='r');ce=np.memmap(root/'candidate_exact.f32',dtype='<f4',mode='r')
    tie={name:np.zeros(n) for name in idarrays}
    for q in range(n):
        sl=slice(offsets[q],offsets[q+1]);lookup=dict(zip(map(int,ci[sl]),map(float,ce[sl])))
        for name in tie:tie[name][q]=tie_hits([lookup[v] for v in idarrays[name][q]],gd[q])/10
    tr=np.where(masks,tie['ALL16'][None,:],tie['NATIVE'][None,:]).mean(1)
    tie_summary={'secondary_metric':'strict-distance hits plus capped GT boundary ties','recalls':{k:float(v.mean()) for k,v in tie.items()},
        'random_mean_recall':float(tr.mean()),'gap_minus_random_mean':float(tie['GAP'].mean()-tr.mean()),
        'selected_harmful_ID_queries':int((dec&(improve<0)).sum()),'selected_harmful_tie_aware_queries':int((dec&(tie['GAP']<tie['NATIVE'])).sum())}
    dump(out/'phase4b_tie_sensitivity.json',tie_summary)
    memory=[]
    for name in NAMES:
        nr=0 if name=='NATIVE' else n if name=='ALL16' else int(dec.sum())
        memory.append({'policy':name,'FP32_vector_accesses':16*nr,'nominal_FP32_payload_bytes':16*nr*128*4,'nominal_payload_bytes_per_query':16*nr*128*4/n,
            'additional_FP32_residency_if_PQ_only_baseline':512000000 if name!='NATIVE' else 0,
            'retention_tags_bytes':4000000 if name!='NATIVE' else 0,'threshold_bytes':8 if name=='GAP' else 0,
            'temporary_mean_live_candidate_ID_score_order_bytes':float(counts.mean()*12) if name!='NATIVE' else 0})
    table(out,'memory',memory)
    batch=[]
    for p in sorted((root/'batch').glob('*.csv')):
        a=np.genfromtxt(p,delimiter=',',names=True);method=int(p.stem[1]);rep=int(p.stem.split('_r')[1])
        assert int(a['queries'].sum())==n and int(a['exact_evals'].sum())==int(dec.sum())*16
        batch.append({'method':'indexed-batch32' if method else 'scalar-deferred32','replicate':rep,
            'amortized_us_per_query':float(a['total_batch_us'].sum()/n),'refine_final_us_per_query':float(a['refine_and_final_us'].sum()/n),
            'mean_batch_completion_us':float(a['total_batch_us'].mean()),'p95_batch_completion_us':float(np.quantile(a['total_batch_us'],.95)),
            'qps':float(n*1e6/a['total_batch_us'].sum()),'same_streaming_results':True})
    table(out,'batch',batch)
    plots(fig,policy_rows,pooled,risk,component_rows,sys)
    print(json.dumps(summary,indent=2))


def plots(out,policies,pooled,risk,components,system):
    import matplotlib
    matplotlib.use('Agg');matplotlib.rcParams['svg.hashsalt']='phase4b'
    import matplotlib.pyplot as plt
    def save(f,name):
        f.tight_layout();p=out/f'phase4b_{name}.svg';f.savefig(p,metadata={'Date':None});plt.close(f)
    for field,label,name in [('mean_us','Mean query service latency (us)','recall_mean_latency'),('p95_us','p95 query service latency (us)','recall_p95_latency'),('exact_evals_per_query','Exact FP32 evaluations/query','recall_exact_cost')]:
        f,ax=plt.subplots()
        for r in policies:
            if r['policy'] not in NAMES:continue
            ax.scatter(r[field],r['recall']);ax.annotate(r['policy'],(r[field],r['recall']),xytext=(4,4),textcoords='offset points')
        ax.set(xlabel=label,ylabel='Recall@10');save(f,name)
    f,ax=plt.subplots()
    for name,a in pooled.items():
        # Full raw timings preserved; plotting first99.9% keeps central/tail visible.
        quant=np.linspace(0,.999,1000);ax.plot(np.quantile(a,quant),quant,label=name)
    ax.set(xlabel='Per-query latency (us), through p99.9',ylabel='Empirical CDF');ax.legend();save(f,'latency_distributions')
    f,axs=plt.subplots(1,2,figsize=(9,4))
    for ax,key in zip(axs,['mean_ranking_loss','harmful_query_fraction']):ax.bar([r['group'] for r in risk],[r[key] for r in risk]);ax.set(ylabel=key)
    save(f,'threshold_risk')
    f,ax=plt.subplots();bottom=np.zeros(4)
    for key in ['t_search','t_gap','t_refine','t_final_selection']:
        vals=[np.mean([r['mean_us'] for r in components if r['policy']==name and r['component']==key]) for name in NAMES]
        ax.bar(NAMES,vals,bottom=bottom,label=key);bottom+=vals
    ax.set(ylabel='Mean component time (us), clocks ON');ax.legend();save(f,'components')
    f,ax=plt.subplots();ax.bar(['Exact eval saving','Mean latency saving','ALL16 gain retained'],[system['exact_eval_saving'],system['latency_saving'],system['all16_gain_retained']]);ax.axhline(.2,ls='--',color='gray',label='latency target20%');ax.axhline(.6,ls=':',color='gray',label='gain target60%');ax.set(ylabel='Fraction (signed where applicable)');ax.legend();save(f,'systems_gate')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tables',default='results/tables');p.add_argument('--figures',default='results/figures');p.add_argument('--derived');args=p.parse_args()
    c=read_conf(Path('configs/indexes/phase4b_streaming_feasibility.conf'));root=Path(c['run']);analyze(root,c,Path(args.tables),Path(args.figures),Path(args.derived) if args.derived else root/'analysis')
