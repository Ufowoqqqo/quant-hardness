"""Frozen compression comparison from raw outputs; no search or refinement tuning."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('runs/phase5b_compression_robustness_v1')
OLD=Path('runs/phase5a_highdim_external_validity_v1')


def describe(v):
    return dict(mean=float(np.mean(v)),std=float(np.std(v,ddof=1)) if len(v)>1 else 0.,
        min=float(np.min(v)),max=float(np.max(v)))


def ratio(a,b):return float(a/b) if b>1e-12 else None


def rank_records(root=ROOT):
    """Offline approximate ranks only; no L32/L64 exact refinement."""
    gt=np.fromfile(root/'gt_ids.i64',dtype='<i8').reshape(10000,11)[:,:10]
    for ef in (32,64,128):
        path=root/f'recall_ef{ef}'
        offsets=np.fromfile(path/'pq_offsets.i64',dtype='<i8')
        ids=np.memmap(path/'pq_candidates.i32',mode='r',dtype='<i4')
        scores=np.memmap(path/'pq_scores.f32',mode='r',dtype='<f4')
        oracle=np.fromfile(path/'oracle_ids.i64',dtype='<i8').reshape(10000,10)
        for qi in range(10000):
            candidates=ids[offsets[qi]:offsets[qi+1]];values=scores[offsets[qi]:offsets[qi+1]]
            order=np.lexsort((np.arange(len(values)),values))
            lookup={int(candidates[j]):rank+1 for rank,j in enumerate(order)}
            for rank,item in enumerate(oracle[qi]):
                yield dict(ef=ef,query_id=qi,oracle_rank=rank+1,candidate_id=int(item),
                    pq_rank=lookup[int(item)],gt_relevant=int(item in gt[qi]),candidate_count=len(candidates))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['recall','single','final'],required=True)
    parser.add_argument('--tables',default='results/tables');parser.add_argument('--figures',default='results/figures');args=parser.parse_args()
    td=Path(args.tables);fd=Path(args.figures);td.mkdir(parents=True,exist_ok=True);fd.mkdir(parents=True,exist_ok=True)
    def write(name,rows):
        target=td/('phase5b_'+name)
        target.with_suffix('.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
        if rows:
            with target.with_suffix('.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    def fig(name):plt.tight_layout();plt.savefig(fd/('phase5b_'+name+'.png'),dpi=160);plt.close()
    recall=[];distributions=[];per={}
    for m,root in [(768,OLD),(384,ROOT)]:
        for ef in (32,64,128):
            data=np.genfromtxt(root/f'recall_ef{ef}/per_query.csv',delimiter=',',names=True);per[m,ef]=data
            row=dict(pq_m=m,compression=6144/m,ef=ef)
            for k in ['recall_exact','recall_pq','recall_candidate_oracle','recall_all16','total_loss','discovery_loss','ranking_loss','exact_l0_count','pq_l0_count','delta_exact_control']:
                row[k]=float(np.mean(data[k]))
            row['candidate_oracle_recovery']=ratio(row['ranking_loss'],row['total_loss'])
            row['L16_recovery']=ratio(row['recall_all16']-row['recall_pq'],row['ranking_loss'])
            row['top16_boundary_tie_queries']=int(np.sum(data['top16_boundary_tie']))
            recall.append(row)
            for k in ['total_loss','discovery_loss','ranking_loss']:
                v=data[k];distributions.append(dict(pq_m=m,ef=ef,metric=k,**describe(v),median=float(np.median(v)),
                    p90=float(np.percentile(v,90)),p95=float(np.percentile(v,95)),p99=float(np.percentile(v,99)),
                    positive_fraction=float(np.mean(v>1e-12)),zero_fraction=float(np.mean(abs(v)<=1e-12)),negative_fraction=float(np.mean(v<-1e-12))))
    def rec(m,ef):return next(r for r in recall if r['pq_m']==m and r['ef']==ef)
    transitions=[]
    for ef in (32,64,128):
        np.testing.assert_array_equal(per[768,ef]['recall_exact'],per[384,ef]['recall_exact'])
        transitions.append(dict(ef=ef,**{k+'_increase_384_minus_768':rec(384,ef)[k]-rec(768,ef)[k] for k in ['total_loss','discovery_loss','ranking_loss']}))
    write('recall_comparison',recall);write('loss_transition',transitions);write('per_query_distributions',distributions)
    sanity=[]
    old=np.genfromtxt(OLD/'sanity_pairs.csv',delimiter=',',names=True)
    for m,root in [(768,OLD),(384,ROOT)]:
        data=np.genfromtxt(root/'sanity_pairs.csv',delimiter=',',names=True)
        for key in ['sample','query_id','base_id','other_id','exact']:np.testing.assert_array_equal(data[key],old[key])
        for key in ['reconstruction_l2','absolute_relative_error','order_inversion','adc_reference_error']:
            v=data[key][np.isfinite(data[key])]
            sanity.append(dict(pq_m=m,metric=key,valid_count=len(v),**describe(v),median=float(np.median(v)),p95=float(np.percentile(v,95))))
    write('sanity_comparison',sanity)
    plt.figure()
    for k,label in [('recall_exact','Exact'),('recall_pq','PQ384 native'),('recall_candidate_oracle','PQ384 candidate oracle'),('recall_all16','PQ384 ALL16')]:
        plt.plot([32,64,128],[rec(384,ef)[k] for ef in (32,64,128)],'o-',label=label)
    plt.xlabel('efSearch');plt.ylabel('Recall@10');plt.legend();fig('recall_decomposition')
    plt.figure()
    for m in (768,384):
        for k,style in [('discovery_loss','o-'),('ranking_loss','s--')]:plt.plot([32,64,128],[rec(m,ef)[k] for ef in (32,64,128)],style,label=f'PQ{m} {k}')
    plt.axhline(0,color='gray',lw=.5);plt.xlabel('efSearch');plt.ylabel('Signed recall loss');plt.legend();fig('discovery_ranking')
    for key in ['candidate_oracle_recovery','L16_recovery']:
        plt.figure()
        for m in (768,384):plt.plot([32,64,128],[rec(m,ef)[key] for ef in (32,64,128)],'o-',label=f'PQ{m}')
        plt.xlabel('efSearch');plt.ylabel(key+' (unclipped)');plt.legend();fig(key)
    memory=[]
    graph_bytes=json.loads((OLD/'graph.json').read_text())['bytes'];raw_bytes=990000*1536*4
    for m in (768,384):
        code=990000*m;book=1536*256*4
        memory.append(dict(pq_m=m,code_bytes_per_vector=m,code_payload_bytes=code,code_compression=6144/m,
            raw_fp32_payload_bytes=raw_bytes,graph_serialized_bytes_including_fp32=graph_bytes,
            graph_serialized_nonvector_remainder_bytes=graph_bytes-raw_bytes,codebook_bytes=book,
            adc_table_bytes_per_worker=m*256*4,all16_vector_payload_bytes_per_query=16*1536*4,
            deployed_payload_estimate_graph_raw_codes_codebook=graph_bytes+code+book))
    write('memory_accounting',memory)
    if args.stage=='recall':return
    raw=[]
    for stage in (['single'] if args.stage=='single' else ['single','concurrent']):
        directory=ROOT/('benchmark_'+stage);assert json.loads((directory/'complete.json').read_text())['status']=='PASS'
        for path in sorted(directory.glob('ef*.json')):
            metadata=json.loads(path.read_text());v=np.genfromtxt(path.with_suffix('.csv'),delimiter=',',names=True)['latency_us']
            row=dict(stage=stage,ef=metadata['ef'],workers=metadata['workers'],policy=metadata['policy'],rep=metadata['rep'],qps=metadata['qps'],
                recall=rec(384,metadata['ef'])['recall_all16' if metadata['policy'] else 'recall_pq'])
            row.update(mean_latency_us=float(np.mean(v)),median_latency_us=float(np.median(v)),
                p90_latency_us=float(np.percentile(v,90)),p95_latency_us=float(np.percentile(v,95)),p99_latency_us=float(np.percentile(v,99)))
            raw.append(row)
    write('benchmark_repetitions',raw);endpoints=[]
    for key in sorted({(r['stage'],r['ef'],r['workers'],r['policy']) for r in raw}):
        group=[r for r in raw if (r['stage'],r['ef'],r['workers'],r['policy'])==key];assert len(group)==5
        row=dict(zip(['stage','ef','workers','policy'],key));row.update(pq_m=384,recall=group[0]['recall'])
        for metric in ['qps','mean_latency_us','median_latency_us','p90_latency_us','p95_latency_us','p99_latency_us']:
            row.update({metric+'_'+k:v for k,v in describe([r[metric] for r in group]).items()})
        endpoints.append(row)
    write('endpoints',endpoints)
    overhead=[]
    for key in sorted({(r['stage'],r['ef'],r['workers']) for r in endpoints}):
        group=[r for r in endpoints if (r['stage'],r['ef'],r['workers'])==key];a=next(r for r in group if r['policy']==0);b=next(r for r in group if r['policy']==1)
        overhead.append(dict(stage=key[0],ef=key[1],workers=key[2],throughput_penalty=1-b['qps_mean']/a['qps_mean'],
            mean_latency_overhead=b['mean_latency_us_mean']/a['mean_latency_us_mean']-1,
            p95_latency_overhead=b['p95_latency_us_mean']/a['p95_latency_us_mean']-1,p99_latency_overhead=b['p99_latency_us_mean']/a['p99_latency_us_mean']-1))
    write('overhead',overhead)
    old_end=json.loads(Path('results/tables/phase5a_endpoints.json').read_text())
    combined=[dict(pq_m=768,**r) for r in old_end if r['stage']=='single']+[r for r in endpoints if r['stage']=='single']
    frontier=[]
    for r in combined:
        dominated=any(o['qps_mean']>=r['qps_mean'] and o['recall']>=r['recall'] and (o['qps_mean']>r['qps_mean'] or o['recall']>r['recall']) for o in combined)
        frontier.append(dict(pq_m=r['pq_m'],ef=r['ef'],policy=r['policy'],recall=r['recall'],qps=r['qps_mean'],qps_std=r['qps_std'],pareto=not dominated))
    write('combined_frontier',frontier)
    pairs=[]
    for a in combined:
        if a['pq_m']!=384 or a['policy']!=1:continue
        for b in combined:
            if b['policy']!=0:continue
            pairs.append(dict(all16_384_ef=a['ef'],native_pq_m=b['pq_m'],native_ef=b['ef'],
                recall_difference=a['recall']-b['recall'],qps_ratio=a['qps_mean']/b['qps_mean'],
                dominates=a['recall']>=b['recall'] and a['qps_mean']>=b['qps_mean']))
    write('observed_comparisons',pairs)
    for pareto_only in [False,True]:
        plt.figure(figsize=(8,5))
        for m in (768,384):
            for policy in (0,1):
                group=sorted([r for r in frontier if r['pq_m']==m and r['policy']==policy],key=lambda r:r['ef'])
                label=f'PQ{m} '+('ALL16' if policy else 'Native')
                plt.errorbar([r['qps'] for r in group],[r['recall'] for r in group],xerr=[r['qps_std'] for r in group],fmt='o-' if not pareto_only else 'o',label=label)
                for r in group:plt.annotate(str(r['ef']),(r['qps'],r['recall']),fontsize=8)
        if pareto_only:
            f=sorted([r for r in frontier if r['pareto']],key=lambda r:r['qps']);plt.plot([r['qps'] for r in f],[r['recall'] for r in f],'k--',label='Observed nondominated points')
        plt.xlabel('QPS (mean ± SD; Phase5A historical reference)');plt.ylabel('Recall@10');plt.legend(fontsize=8);fig('pareto' if pareto_only else 'compression_frontier')
    if args.stage=='single':
        marker=ROOT/'single_analysis_complete.json'
        if not marker.exists():marker.write_text(json.dumps(dict(status='PASS',points=frontier),indent=2))
        return
    # Authorized offline diagnostic only, after primary single-thread results.
    ranks_path=ROOT/'candidate_oracle_item_ranks.csv'
    if not ranks_path.exists():
        iterator=iter(rank_records());first=next(iterator)
        with ranks_path.open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(first));w.writeheader();w.writerow(first);w.writerows(iterator)
    ranks=np.genfromtxt(ranks_path,delimiter=',',names=True)
    rank_summary=[];plt.figure()
    for ef in (32,64,128):
        for population in ['GT_relevant_oracle_members','all_oracle_members']:
            subset=ranks[ranks['ef']==ef]
            if population.startswith('GT_'):subset=subset[subset['gt_relevant']==1]
            values=subset['pq_rank'];row=dict(ef=ef,population=population,item_count=len(values),
                queries_with_zero_relevant_items=10000-len(np.unique(subset['query_id'])),**describe(values),
                median=float(np.median(values)),p95=float(np.percentile(values,95)),p99=float(np.percentile(values,99)))
            row.update({f'fraction_within_{k}':float(np.mean(values<=k)) for k in (10,16,32,64)})
            rank_summary.append(row)
            if population.startswith('GT_'):
                ordered=np.sort(values);plt.step(ordered,np.arange(1,len(values)+1)/len(values),where='post',label=f'ef{ef}')
    write('candidate_ranks',rank_summary);plt.xscale('log');plt.xlabel('PQ384 approximate rank of GT-relevant oracle item');plt.ylabel('Empirical CDF (item-weighted)');plt.legend();fig('candidate_rank_cdf')
    scaling=[];plt.figure()
    for policy in (0,1):
        group=sorted([r for r in endpoints if r['stage']=='concurrent' and r['policy']==policy],key=lambda r:r['workers']);one=group[0]
        for r in group:scaling.append(dict(policy=policy,workers=r['workers'],qps=r['qps_mean'],speedup=r['qps_mean']/one['qps_mean'],parallel_efficiency=r['qps_mean']/one['qps_mean']/r['workers']))
        plt.errorbar([r['workers'] for r in group],[r['qps_mean'] for r in group],yerr=[r['qps_std'] for r in group],fmt='o-',label='ALL16' if policy else 'Native')
    write('scaling',scaling);plt.xlabel('Pinned physical-core workers');plt.ylabel('QPS (mean ± SD)');plt.legend();fig('concurrency')
    print(json.dumps(dict(recall=recall,transitions=transitions,overhead=overhead,frontier=frontier,rank_summary=rank_summary),indent=2))


if __name__=='__main__':main()
