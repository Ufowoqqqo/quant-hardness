#!/usr/bin/env python3
"""Regenerate Phase 5A primary tables/figures from raw, without tuning."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def table(path,rows):
    with path.with_suffix('.json').open('w') as f:json.dump(rows,f,indent=2,allow_nan=False)
    if rows:
        with path.with_suffix('.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def ratio(a,b):return float(a/b) if b>1e-12 else None


def describe(v):
    return dict(mean=float(np.mean(v)),std=float(np.std(v,ddof=1)) if len(v)>1 else 0.,
                min=float(np.min(v)),max=float(np.max(v)))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['recall','single','final'],required=True)
    parser.add_argument('--tables',default='results/tables');parser.add_argument('--figures',default='results/figures');args=parser.parse_args()
    root=Path('runs/phase5a_highdim_external_validity_v1');td=Path(args.tables);fd=Path(args.figures);td.mkdir(parents=True,exist_ok=True);fd.mkdir(parents=True,exist_ok=True)
    def write(name,rows):table(td/('phase5a_'+name),rows)
    def fig(name):plt.tight_layout();plt.savefig(fd/('phase5a_'+name+'.png'),dpi=160);plt.close()
    recall=[];distribution=[]
    for ef in [32,64,128]:
        r=np.genfromtxt(root/f'recall_ef{ef}/per_query.csv',delimiter=',',names=True)
        row={'ef':ef}
        for k in ['recall_exact','recall_pq','recall_candidate_oracle','recall_all16','total_loss','discovery_loss','ranking_loss','coverage_exact','coverage_pq','exact_l0_count','pq_l0_count','delta_exact_control']:
            row[k]=float(np.mean(r[k]))
        row['candidate_oracle_recovery']=ratio(row['ranking_loss'],row['total_loss'])
        row['L16_recovery_fraction']=ratio(row['recall_all16']-row['recall_pq'],row['ranking_loss'])
        row['discovery_fraction']=ratio(row['discovery_loss'],row['total_loss'])
        row['top16_boundary_tie_queries']=int(np.sum(r['top16_boundary_tie']))
        recall.append(row)
        for k in ['total_loss','discovery_loss','ranking_loss']:
            v=r[k];distribution.append(dict(ef=ef,metric=k,**describe(v),median=float(np.median(v)),
                p90=float(np.percentile(v,90)),p95=float(np.percentile(v,95)),p99=float(np.percentile(v,99)),
                positive=float(np.mean(v>1e-12)),zero=float(np.mean(np.abs(v)<=1e-12)),negative=float(np.mean(v<-1e-12))))
    write('recall',recall);write('per_query_distributions',distribution)
    s=np.genfromtxt(root/'sanity_pairs.csv',delimiter=',',names=True)
    sanity=[]
    for k in ['absolute_relative_error','reconstruction_l2','order_inversion','adc_reference_error']:
        v=s[k][np.isfinite(s[k])]
        sanity.append(dict(metric=k,valid_count=len(v),undefined_count=len(s)-len(v),**describe(v),median=float(np.median(v)),p95=float(np.percentile(v,95))))
    write('pq_sanity',sanity)
    efs=[r['ef'] for r in recall]
    plt.figure()
    for k,label in [('recall_exact','Exact'),('recall_pq','PQ-native'),('recall_candidate_oracle','PQ candidate oracle'),('recall_all16','Bounded ALL16')]:plt.plot(efs,[r[k] for r in recall],'o-',label=label)
    plt.xlabel('efSearch');plt.ylabel('Recall@10');plt.legend();fig('recall_decomposition')
    plt.figure()
    for k in ['discovery_loss','ranking_loss']:plt.plot(efs,[r[k] for r in recall],'o-',label=k)
    plt.axhline(0,color='gray',lw=.6);plt.xlabel('efSearch');plt.ylabel('Signed recall loss');plt.legend();fig('discovery_ranking')
    plt.figure()
    for k in ['candidate_oracle_recovery','L16_recovery_fraction']:plt.plot(efs,[r[k] for r in recall],'o-',label=k)
    plt.xlabel('efSearch');plt.ylabel('Recovery fraction (unclipped)');plt.legend();fig('recovery')
    if args.stage=='recall':return
    raw=[]
    for stage in (['single'] if args.stage=='single' else ['single','concurrent']):
        directory=root/('benchmark_'+stage);assert json.loads((directory/'complete.json').read_text())['status']=='PASS'
        for path in sorted(directory.glob('ef*.json')):
            m=json.loads(path.read_text());r=np.genfromtxt(path.with_suffix('.csv'),delimiter=',',names=True);v=r['latency_us']; rr=next(x for x in recall if x['ef']==m['ef'])
            raw.append(dict(stage=stage,ef=m['ef'],workers=m['workers'],policy=m['policy'],rep=m['rep'],
                recall=rr['recall_all16' if m['policy'] else 'recall_pq'],qps=m['qps'],mean_latency_us=float(np.mean(v)),
                median_latency_us=float(np.median(v)),p90_latency_us=float(np.percentile(v,90)),p95_latency_us=float(np.percentile(v,95)),p99_latency_us=float(np.percentile(v,99))))
    write('benchmark_repetitions',raw)
    endpoints=[]
    for key in sorted(set((r['stage'],r['ef'],r['workers'],r['policy']) for r in raw)):
        group=[r for r in raw if (r['stage'],r['ef'],r['workers'],r['policy'])==key];assert len(group)==5
        row=dict(zip(['stage','ef','workers','policy'],key));row['recall']=group[0]['recall']
        for k in ['qps','mean_latency_us','median_latency_us','p90_latency_us','p95_latency_us','p99_latency_us']:
            for stat,value in describe([r[k] for r in group]).items():row[k+'_'+stat]=value
        endpoints.append(row)
    write('endpoints',endpoints)
    overhead=[]
    for key in sorted(set((r['stage'],r['ef'],r['workers']) for r in endpoints)):
        a=next(r for r in endpoints if (r['stage'],r['ef'],r['workers'],r['policy'])==(*key,0));b=next(r for r in endpoints if (r['stage'],r['ef'],r['workers'],r['policy'])==(*key,1))
        overhead.append(dict(stage=key[0],ef=key[1],workers=key[2],throughput_penalty=1-b['qps_mean']/a['qps_mean'],
            mean_latency_overhead=b['mean_latency_us_mean']/a['mean_latency_us_mean']-1,
            p95_latency_overhead=b['p95_latency_us_mean']/a['p95_latency_us_mean']-1,p99_latency_overhead=b['p99_latency_us_mean']/a['p99_latency_us_mean']-1,
            additional_vector_payload_bytes_per_query=98304))
    write('overhead',overhead)
    single=[r for r in endpoints if r['stage']=='single'];frontier=[];pairs=[]
    for r in single:
        dominated=any(o['qps_mean']>=r['qps_mean'] and o['recall']>=r['recall'] and (o['qps_mean']>r['qps_mean'] or o['recall']>r['recall']) for o in single)
        frontier.append(dict(ef=r['ef'],policy=r['policy'],qps=r['qps_mean'],recall=r['recall'],pareto=not dominated))
    for a in single:
        for b in single:
            if a['policy']==1 and b['policy']==0 and a['ef']<b['ef']:
                pairs.append(dict(all16_ef=a['ef'],native_ef=b['ef'],recall_difference=a['recall']-b['recall'],qps_ratio=a['qps_mean']/b['qps_mean']))
    write('observed_frontier',frontier);write('observed_cross_ef_pairs',pairs)
    plt.figure()
    for policy,label in [(0,'Native'),(1,'Bounded ALL16')]:
        a=sorted([r for r in single if r['policy']==policy],key=lambda r:r['ef']);plt.errorbar([r['qps_mean'] for r in a],[r['recall'] for r in a],xerr=[r['qps_std'] for r in a],fmt='o-',label=label)
        for r in a:plt.annotate('ef'+str(r['ef']),(r['qps_mean'],r['recall']))
    plt.xlabel('QPS (mean ± SD across 5 repetitions)');plt.ylabel('Recall@10');plt.legend();fig('recall_qps')
    plt.figure()
    a=[r for r in overhead if r['stage']=='single']
    for k in ['throughput_penalty','mean_latency_overhead','p99_latency_overhead']:plt.plot([r['ef'] for r in a],[r[k] for r in a],'o-',label=k)
    plt.xlabel('efSearch');plt.ylabel('Relative overhead (unclipped)');plt.legend();fig('overhead')
    if args.stage=='single':
        marker=root/'single_analysis_complete.json'
        if not marker.exists():marker.write_text(json.dumps(dict(status='PASS',six_points=frontier,observed_pairs=pairs),indent=2))
        return
    scaling=[]
    for policy in [0,1]:
        group=[r for r in endpoints if r['stage']=='concurrent' and r['policy']==policy];one=next(r for r in group if r['workers']==1)
        for r in group:scaling.append(dict(policy=policy,workers=r['workers'],qps=r['qps_mean'],speedup=r['qps_mean']/one['qps_mean'],parallel_efficiency=r['qps_mean']/one['qps_mean']/r['workers']))
    write('scaling',scaling)
    plt.figure()
    for policy,label in [(0,'Native'),(1,'Bounded ALL16')]:
        a=sorted([r for r in endpoints if r['stage']=='concurrent' and r['policy']==policy],key=lambda r:r['workers']);plt.errorbar([r['workers'] for r in a],[r['qps_mean'] for r in a],yerr=[r['qps_std'] for r in a],fmt='o-',label=label)
    plt.xlabel('Pinned physical-core query workers');plt.ylabel('QPS (mean ± SD)');plt.legend();fig('concurrency')
    old=list(csv.DictReader(Path('results/tables/phase4d_endpoints.csv').open()))
    sn=next(r for r in old if r['policy']=='NATIVE' and r['workers']=='1');sa=next(r for r in old if r['policy']=='BOUNDED_ALL16' and r['workers']=='1')
    high=next(r for r in recall if r['ef']==64);hn=next(r for r in single if r['ef']==64 and r['policy']==0);ha=next(r for r in single if r['ef']==64 and r['policy']==1)
    sift_pq=float(sn['recall_at_10']);sift_all=float(sa['recall_at_10']);sift_oracle=.9503648610721304
    cross=[dict(dataset='SIFT1M Phase4B/C/D stream',dimension=128,workload='image descriptors',metric='squared L2',pq='64x8',dsub=2,compression=8,ef=64,exact_recall=None,pq_recall=sift_pq,candidate_oracle=sift_oracle,discovery_loss=None,ranking_loss=sift_oracle-sift_pq,candidate_oracle_recovery=None,all16_recall=sift_all,L16_recovery=ratio(sift_all-sift_pq,sift_oracle-sift_pq),native_qps=float(sn['qps']),all16_qps=float(sa['qps']),throughput_penalty=1-float(sa['qps'])/float(sn['qps']),p99_overhead=float(sa['p99_latency_us'])/float(sn['p99_latency_us'])-1),
           dict(dataset='DBPedia OpenAI3 Phase5A',dimension=1536,workload='held-out entity-to-entity embeddings',metric='normalized FP32 squared L2 / cosine equivalent',pq='768x8',dsub=2,compression=8,ef=64,exact_recall=high['recall_exact'],pq_recall=high['recall_pq'],candidate_oracle=high['recall_candidate_oracle'],discovery_loss=high['discovery_loss'],ranking_loss=high['ranking_loss'],candidate_oracle_recovery=high['candidate_oracle_recovery'],all16_recall=high['recall_all16'],L16_recovery=high['L16_recovery_fraction'],native_qps=hn['qps_mean'],all16_qps=ha['qps_mean'],throughput_penalty=1-ha['qps_mean']/hn['qps_mean'],p99_overhead=ha['p99_latency_us_mean']/hn['p99_latency_us_mean']-1)]
    write('cross_dataset',cross)
    plt.figure(figsize=(8,4));pos=np.arange(2)
    for shift,key,label in [(-.25,'pq_recall','PQ-native'),(0,'all16_recall','ALL16'),(.25,'candidate_oracle','Candidate oracle')]:plt.bar(pos+shift,[r[key] for r in cross],width=.25,label=label)
    plt.xticks(pos,['SIFT1M','DBPedia OpenAI3']);plt.ylabel('Recall@10, ef64');plt.legend();fig('cross_dataset')
    print(json.dumps(dict(recall=recall,overhead=overhead,observed_pairs=pairs,scaling=scaling),indent=2))


if __name__=='__main__':main()
