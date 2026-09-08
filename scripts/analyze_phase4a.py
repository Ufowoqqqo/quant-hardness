#!/usr/bin/env python3
"""Independent validation, then frozen primary analysis, then diagnostics."""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np
from analyze_phase3a_sift1m import read_conf, spearman, write_csv
from analyze_phase3b_fixed_candidate_ranking import digest, dump
from phase4a_metrics import score_gap, hit_count, refinement, selector, oracle_selector, random_order, ratio, policy_value
from prepare_phase4a import fvecs


def table(out,name,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    rows=[{k:r.get(k) for k in keys} for r in rows]
    write_csv(out/f'phase4a_{name}.csv',rows);dump(out/f'phase4a_{name}.json',rows)


def arrays(root,n):
    def a(name,dtype,shape=None):
        v=np.memmap(root/name,dtype=dtype,mode='r');return v.reshape(shape) if shape else v
    return {'offset':a('candidate_offsets.i64','<i8'), 'ids':a('candidate_ids.i64','<i8'),
        'exact':a('candidate_exact.f32','<f4'),'pq':a('candidate_pq.f32','<f4'),
        'native':a('pq_native_ids.i64','<i8',(n,10)), 'native_scores':a('pq_native_scores.f32','<f4',(n,10)),
        'gt':a('gt_ids.i64','<i8',(n,11)),'gd':a('gt_distances.f32','<f4',(n,11)),
        'en':a('exact_native_ids.i64','<i8',(n,10)),'ens':a('exact_native_scores.f32','<f4',(n,10)),
        'eo':a('exact_control_oracle_ids.i64','<i8',(n,10)),
        'eos':a('exact_control_oracle_scores.f32','<f4',(n,10))}


def validate(c,root):
    assert json.loads((root/'export_validation.json').read_text())['status']=='PASS'
    p=json.loads((root/'provenance.json').read_text())
    for name,h in p['sha256'].items():assert digest(name)==h, name
    n=int(c['query_count']);a=arrays(root,n);base=fvecs(c['base_path'])
    q=np.memmap(root/'queries.f32',dtype='<f4',mode='r').reshape(n,128)
    rng=np.random.default_rng(int(c['gt_validation_seed']))
    sampled=np.sort(rng.choice(n,int(c['gt_validation_queries']),replace=False))
    gt_rows=[]
    for qi in sampled:
        # Independent direct FP64 squared differences, no FAISS / BLAS identity.
        ds=np.empty(len(base),dtype=np.float64)
        for start in range(0,len(base),10000):
            diff=base[start:start+10000].astype(np.float64)-q[qi].astype(np.float64)
            ds[start:start+len(diff)]=(diff*diff).sum(1)
        threshold=float(np.partition(ds,9)[9]);ids=a['gt'][qi,:10]
        assert np.all(ds[ids]<=threshold) and set(np.flatnonzero(ds<threshold)).issubset(set(ids))
        assert np.array_equal(ds[ids],a['gd'][qi,:10]), 'integer SIFT distances must be exactly representable'
        gt_rows.append({'query_id':int(qi),'gt_threshold':threshold,'strictly_closer':int((ds<threshold).sum()),'ties_at_threshold':int((ds==threshold).sum()),'status':'PASS'})
    # All stored candidate exact distances independently checked with FP64.
    candidate_records=0;control_ties=[];native_ties=[];control_anomalies=[]
    for qi in range(n):
        sl=slice(a['offset'][qi],a['offset'][qi+1]);ids=a['ids'][sl];de=a['exact'][sl];dp=a['pq'][sl]
        assert len(ids)>=64 and len(set(ids))==len(ids)
        delta=base[ids].astype(np.float64)-q[qi].astype(np.float64)
        assert np.array_equal((delta*delta).sum(1),de),'candidate distance mismatch'
        order=np.argsort(dp,kind='stable');ni=a['native'][qi]
        lookup={int(v):i for i,v in enumerate(ids)}
        assert all(int(v) in lookup for v in ni)
        actual=np.array([dp[lookup[int(v)]] for v in ni]);assert np.array_equal(actual,a['native_scores'][qi])
        assert np.array_equal(np.sort(actual),dp[order[:10]]),'native not global PQ top10 modulo ties'
        if not np.array_equal(ni,ids[order[:10]]):native_ties.append(qi)
        # Control IDs can differ at exact score ties, but never at a strict distance.
        assert np.array_equal(a['ens'][qi],a['eos'][qi]),'exact control strict score discrepancy'
        nh=hit_count(a['en'][qi],a['gt'][qi,:10]);oh=hit_count(a['eo'][qi],a['gt'][qi,:10])
        if nh!=oh:
            changed=set(a['en'][qi])^set(a['eo'][qi]);threshold=float(a['ens'][qi,9])
            dd=((base[list(changed)].astype(float)-q[qi].astype(float))**2).sum(1)
            assert np.all(dd==threshold) and threshold==a['gd'][qi,9], 'unexplained exact-control anomaly'
            control_anomalies.append({'query_id':qi,'delta_exact_control':(oh-nh)/10,'tie_distance':threshold,'changed_ids':list(map(int,changed))})
        if not np.array_equal(a['en'][qi],a['eo'][qi]):control_ties.append(qi)
        candidate_records+=len(ids)
    dump(root/'gt_independent_validation.json',gt_rows)
    result={'status':'PASS','gt_validation_queries':len(sampled),'all_candidate_exact_distances_checked':candidate_records,
        'native_pq_global_order_tie_queries':native_ties,'exact_control_order_tie_queries':control_ties,
        'exact_control_recall_anomalies':control_anomalies,'non_tie_anomalies':0,
        'export_sha256':{str(p):digest(p) for p in sorted(root.iterdir()) if p.suffix in ('.i64','.f32','.index')},
        'source_sha256':{str(p):digest(p) for p in [Path('src/experiments/faiss_phase4a.cpp'),Path('scripts/analyze_phase4a.py'),Path('scripts/phase4a_metrics.py'),Path('scripts/prepare_phase4a.py')]}}
    dump(root/'validation.json',result);print('Validation PASS',candidate_records,'candidate distances;',len(control_anomalies),'verified GT-boundary control ties',flush=True)


def analyze(c,root,out,fig,derived):
    v=json.loads((root/'validation.json').read_text());assert v['status']=='PASS'
    for name,h in v['export_sha256'].items():assert digest(name)==h
    out.mkdir(parents=True,exist_ok=True);fig.mkdir(parents=True,exist_ok=True);derived.mkdir(parents=True,exist_ok=True)
    n=int(c['query_count']);a=arrays(root,n);Ls=list(map(int,c['depths'].split(',')));fs=list(map(float,c['fractions'].split(',')));P=int(c['random_replicates'])
    gap=np.zeros(n);rel=np.zeros(n);nh=np.zeros(n,dtype=int);oh=nh.copy();en=nh.copy();eo=nh.copy()
    rh=np.zeros((len(Ls),n),dtype=int);ref_ids=np.zeros((len(Ls),n,10),dtype='<i8');oracle_ids=np.zeros((n,10),dtype='<i8')
    with gzip.open(derived/'queries.jsonl.gz','wt') as f:
        for qi in range(n):
            sl=slice(a['offset'][qi],a['offset'][qi+1]);ids=a['ids'][sl];de=a['exact'][sl];dp=a['pq'][sl];gt=a['gt'][qi,:10]
            order=np.argsort(de,kind='stable');oracle_ids[qi]=ids[order[:10]];gap[qi],rel[qi]=score_gap(dp)
            nh[qi]=hit_count(a['native'][qi],gt);oh[qi]=hit_count(oracle_ids[qi],gt)
            en[qi]=hit_count(a['en'][qi],gt);eo[qi]=hit_count(a['eo'][qi],gt)
            r={'query_id':qi,'shard':qi%4,'g9_12_pq':gap[qi],'g9_12_rel':rel[qi],
               'candidate_count':len(ids),'native_recall':nh[qi]/10,'oracle_recall':oh[qi]/10,
               'ranking_recall_loss':(oh[qi]-nh[qi])/10,'exact_native_recall':en[qi]/10,
               'exact_L0_oracle_recall':eo[qi]/10,'delta_discovery':(eo[qi]-oh[qi])/10,
               'delta_exact_control':(eo[qi]-en[qi])/10,'native_ids':list(map(int,a['native'][qi])),
               'oracle_ids':oracle_ids[qi].tolist(),'ground_truth_ids':list(map(int,gt))}
            for li,L in enumerate(Ls):
                result,selected=refinement(ids,dp,de,L);ref_ids[li,qi]=result;rh[li,qi]=hit_count(result,gt)
                r[f'top{L}_ids']=result.tolist();r[f'top{L}_recall']=rh[li,qi]/10
            f.write(json.dumps(r)+'\n')
    policies=[];draws=[];references=[];quintiles=[];correlations=[];selected_records={}
    for scope,ix in [('all',np.arange(n))]+[(f'shard{s}',np.flatnonzero(np.arange(n)%4==s)) for s in range(4)]:
        scope_id=0 if scope=='all' else int(scope[-1])+1;nn=len(ix);bn=nh[ix];bo=oh[ix];recgap=float((bo-bn).mean()/10)
        references.append({'scope':scope,'reference':'PQ native','recall':float(bn.mean()/10),'cost':0})
        references.append({'scope':scope,'reference':'full PQ candidate oracle','recall':float(bo.mean()/10),'cost':float(np.diff(a['offset'])[ix].mean())})
        for li,L in enumerate(Ls):references.append({'scope':scope,'reference':f'top{L} all','recall':float(rh[li,ix].mean()/10),'cost':L})
        randoms=[random_order(nn,int(c['random_seed']),scope_id,p) for p in range(P)]
        for li,L in enumerate(Ls):
            br=rh[li,ix];gains=br-bn
            for fraction in fs:
                count=int(fraction*nn);gsel=selector(gap[ix],count);osel=oracle_selector(gains,count);fsel=oracle_selector(bo-bn,count)
                rs=[policy_value(bn,br,r[:count],L) for r in randoms];random_recall=np.array([r['recall'] for r in rs]);rm=float(random_recall.mean())
                lo,hi=map(float,np.quantile(random_recall,[.025,.975]));ov=policy_value(bn,br,osel,L)['recall']
                for p,r in enumerate(rs):draws.append({'scope':scope,'L':L,'fraction':fraction,'replicate':p,**r})
                for name,sel in [('GAP',gsel),('ORACLE-QUERY',osel),('FULL-LOSS-SELECTOR',fsel)]:
                    value=policy_value(bn,br,sel,L)
                    assert value['recall']<=ov+1e-12
                    row={'scope':scope,'L':L,'fraction':fraction,'policy':name,'selected_queries':count,**value,
                        'recovery_fraction':ratio(value['recall_gain'],recgap),
                        'recall_gain_per_exact_eval':ratio(value['recall_gain'],value['exact_distance_evals_per_query']),
                        'random_mean_recall':rm,'random_recall_p025':lo,'random_recall_p975':hi,
                        'gap_vs_random_recall':value['recall']-rm,'recovery_vs_random':ratio(value['recall']-rm,recgap),
                        'selector_efficiency':ratio(value['recall']-rm,ov-rm),'exceeds_random_p975':bool(value['recall']>hi)}
                    policies.append(row);selected_records[f'{scope}_L{L}_f{fraction}_{name}']=ix[sel]
                rmean={'recall':rm,'recall_gain':rm-float(bn.mean()/10),'exact_distance_evals_per_query':count*L/nn}
                policies.append({'scope':scope,'L':L,'fraction':fraction,'policy':'RANDOM','selected_queries':count,**rmean,
                    'recovery_fraction':ratio(rmean['recall_gain'],recgap),'recall_gain_per_exact_eval':ratio(rmean['recall_gain'],count*L/nn),
                    'random_mean_recall':rm,'random_recall_p025':lo,'random_recall_p975':hi,'random_std':float(random_recall.std(ddof=1))})
        for p,r in enumerate(randoms):selected_records[f'{scope}_random_order_{p}']=ix[r]
        for feature,x in [('g9_12_pq',gap),('g9_12_rel',rel)]:
            correlations.append({'scope':scope,'feature':feature,'spearman_loss':spearman(x[ix].tolist(),((bo-bn)/10).tolist())})
            for quint,ii in enumerate(np.array_split(ix[np.argsort(x[ix],kind='stable')],5),1):
                loss=(oh[ii]-nh[ii])/10
                quintiles.append({'scope':scope,'feature':feature,'quintile':quint,'queries':len(ii),'min':float(x[ii].min()),'max':float(x[ii].max()),
                    'mean_ranking_loss':float(loss.mean()),'harmful_fraction':float((loss>0).mean()),'available_rerank_gain':float(loss.mean()),
                    **{f'top{L}_gain':float((rh[li,ii]-nh[ii]).mean()/10) for li,L in enumerate(Ls)}})
    for name,rows in [('policies',policies),('random_draws',draws),('references',references),('quintiles',quintiles),('correlations',correlations)]:table(out,name,rows)
    np.savez_compressed(derived/'selections.npz',**selected_records)
    gr=[r for r in policies if r['scope']=='all' and r['policy']=='GAP']
    useful_L=[L for L in Ls if all(next(r for r in gr if r['L']==L and r['fraction']==f)['exceeds_random_p975'] for f in [.25,.5])]
    summary={'queries':n,'native_recall':float(nh.mean()/10),'full_candidate_oracle_recall':float(oh.mean()/10),
        'exact_native_recall':float(en.mean()/10),'exact_L0_oracle_recall':float(eo.mean()/10),
        'mean_delta_discovery':float((eo-oh).mean()/10),'mean_delta_exact_control':float((eo-en).mean()/10),
        'recoverable_gap':float((oh-nh).mean()/10),'useful_heldout_stratification_pass':len(useful_L)>=2,
        'depths_passing_both_primary_budgets':useful_L,'budgets_exceeding_random_p975':sum(r['exceeds_random_p975'] for r in gr),
        'gap_spearman':correlations[0]['spearman_loss'],'negative_ranking_loss_queries':int((oh<nh).sum()),
        'min_candidate_count':int(np.diff(a['offset']).min()),'short_pool_queries':int((np.diff(a['offset'])<12).sum()),
        'candidate_semantics':'PQ-guided level-0 distance evaluations; exact control never used in selection',
        'raw_candidate_sha256':{key:digest(root/file) for key,file in [('ids','candidate_ids.i64'),('offsets','candidate_offsets.i64'),('pq','candidate_pq.f32')]}}
    dump(out/'phase4a_summary.json',summary)
    # Primary completion checkpoint precedes candidate-level diagnostics.
    dump(derived/'primary_checkpoint.json',{'status':'PASS','summary':summary,'policy_table_sha256':digest(out/'phase4a_policies.json')})
    secondary=[]
    for li,L in enumerate(Ls):
        for fraction in fs:
            qi_ids=selected_records[f'all_L{L}_f{fraction}_GAP'];corrected=[];missed=[];rescued_ranks=[];wrong_removed=[];rescued_oracle_ranks=[]
            for qi in qi_ids:
                sl=slice(a['offset'][qi],a['offset'][qi+1]);ids=a['ids'][sl];de=a['exact'][sl];dp=a['pq'][sl]
                native=set(a['native'][qi]);oracle=set(oracle_ids[qi]);ref=set(ref_ids[li,qi]);topL=set(ids[np.argsort(dp,kind='stable')[:L]])
                corrected.append(len((ref-native)&oracle));wrong_removed.append(len((native-ref)-oracle));missed.append(len(oracle-topL))
                rank={int(ids[j]):r+1 for r,j in enumerate(np.argsort(de,kind='stable'))}
                rescued_ranks.extend(rank[int(v)] for v in ref-native)
                rescued_oracle_ranks.extend(rank[int(v)] for v in (ref-native)&oracle)
            secondary.append({'L':L,'fraction':fraction,'selected_queries':len(qi_ids),'mean_correct_oracle_members_added':float(np.mean(corrected)),
                'mean_wrong_native_members_removed':float(np.mean(wrong_removed)),
                'oracle_top10_missing_from_topL_query_fraction':float((np.array(missed)>0).mean()),'mean_oracle_members_outside_topL':float(np.mean(missed)),
                'new_result_rank_count':len(rescued_ranks),'new_result_rank_median':float(np.median(rescued_ranks)) if rescued_ranks else None,
                'new_result_rank_p95':float(np.quantile(rescued_ranks,.95)) if rescued_ranks else None,
                'rescued_oracle_rank_median':float(np.median(rescued_oracle_ranks)) if rescued_oracle_ranks else None,
                'rescued_oracle_rank_p95':float(np.quantile(rescued_oracle_ranks,.95)) if rescued_oracle_ranks else None})
            np.savez_compressed(derived/f'secondary_L{L}_f{fraction}.npz',query_ids=qi_ids,corrected=corrected,wrong_removed=wrong_removed,missed_oracle=missed,new_result_ranks=rescued_ranks,rescued_oracle_ranks=rescued_oracle_ranks)
    table(out,'candidate_diagnostics',secondary)
    figures(fig,policies,references,quintiles)
    print(json.dumps(summary,indent=2))


def figures(out,policies,references,quintiles):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['svg.hashsalt']='phase4a'
    def save(f,name):f.tight_layout();f.savefig(out/f'phase4a_{name}.svg',metadata={'Date':None});plt.close(f)
    for y,name in [('recall','recall_cost'),('recovery_fraction','recovery_cost')]:
        f,axs=plt.subplots(1,3,figsize=(13,4))
        for ax,L in zip(axs,[16,32,64]):
            for policy in ['GAP','RANDOM','ORACLE-QUERY']:
                rr=[r for r in policies if r['scope']=='all' and r['L']==L and r['policy']==policy]
                ax.plot([r['exact_distance_evals_per_query'] for r in rr],[r[y] for r in rr],'-o',label=policy)
                if policy=='RANDOM' and y=='recall':ax.fill_between([r['exact_distance_evals_per_query'] for r in rr],[r['random_recall_p025'] for r in rr],[r['random_recall_p975'] for r in rr],alpha=.2)
            if y=='recall':
                for r in references:
                    if r['scope']=='all' and r['reference'] in ['PQ native','full PQ candidate oracle']:ax.axhline(r['recall'],linestyle='--',label=r['reference']+' (reference)')
            ax.set(title=f'L={L}',xlabel='Exact evaluations/workload query',ylabel=y);ax.legend(fontsize=7)
        save(f,name)
    f,ax=plt.subplots()
    for L in [16,32,64]:
        rr=[r for r in policies if r['scope']=='all' and r['L']==L and r['policy']=='GAP']
        x=[r['fraction'] for r in rr];y=[r['gap_vs_random_recall'] for r in rr]
        ax.plot(x,y,'-o',label=f'L={L}')
        ax.fill_between(x,[r['recall']-r['random_recall_p975'] for r in rr],[r['recall']-r['random_recall_p025'] for r in rr],alpha=.15)
    ax.axhline(0,color='black');ax.set(xlabel='Fraction refined',ylabel='GAP minus RANDOM recall');ax.legend();save(f,'advantage')
    f,ax=plt.subplots();rr=[r for r in quintiles if r['scope']=='all' and r['feature']=='g9_12_pq']
    ax.plot([r['quintile'] for r in rr],[r['mean_ranking_loss'] for r in rr],'-o');ax.set(xlabel='Frozen raw gap quintile (low to high)',ylabel='Mean held-out ranking loss');save(f,'risk_quintiles')
    f,axs=plt.subplots(1,3,figsize=(12,4))
    for ax,L in zip(axs,[16,32,64]):
        for s in range(4):
            rr=[r for r in policies if r['scope']==f'shard{s}' and r['L']==L and r['policy']=='GAP']
            ax.plot([r['fraction'] for r in rr],[r['gap_vs_random_recall'] for r in rr],'-o',label=f'shard{s}')
        ax.axhline(0,color='black');ax.set(title=f'L={L}',xlabel='Fraction refined',ylabel='GAP minus random recall');ax.legend()
    save(f,'shards')
    f,ax=plt.subplots();rr=[r for r in references if r['scope']=='all' and (r['reference'].startswith('top') or r['reference']=='PQ native')]
    ax.plot([r['cost'] for r in rr],[r['recall'] for r in rr],'-o');full=next(r for r in references if r['scope']=='all' and r['reference'].startswith('full'))
    ax.axhline(full['recall'],linestyle='--',label=f"full oracle: {full['cost']:.1f} evaluations/query");ax.legend();ax.set(xlabel='Exact evaluations/query (all queries refined)',ylabel='Recall@10');save(f,'topL_upper_bound')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['validate','analyze']);p.add_argument('--tables',default='results/tables');p.add_argument('--figures',default='results/figures');p.add_argument('--derived');args=p.parse_args()
    c=read_conf(Path('configs/indexes/phase4a_gap_guided_refinement.conf'));root=Path(c['run'])
    if args.stage=='validate':validate(c,root)
    else:analyze(c,root,Path(args.tables),Path(args.figures),Path(args.derived) if args.derived else root/'analysis')
