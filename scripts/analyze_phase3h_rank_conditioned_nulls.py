#!/usr/bin/env python3
"""Primary hierarchy analysis, followed by separately gated PQ-only feasibility."""
import argparse
import json
from pathlib import Path
import numpy as np
from analyze_phase3a_sift1m import read_conf,spearman,write_csv
from analyze_phase3b_fixed_candidate_ranking import dump,digest
from analyze_phase3c_precision_transition import jsonl,read_candidates,gz_writer
from analyze_phase3d_quantizer_seed_stability import savefig
from phase3g_metrics import summary
from phase3h_metrics import CONDITIONS,OBSERVABLES,closure,pq_observables,quintiles


def table(root,name,rows):
    keys=list(dict.fromkeys(key for row in rows for key in row))
    uniform=[{key:r.get(key) for key in keys} for r in rows]
    write_csv(root/f'phase3h_{name}.csv',uniform);dump(root/f'phase3h_{name}.json',uniform)


def load(config,run):
    cfg=read_conf(config);names=cfg['models'].split(',')
    rows={name:list(jsonl(run/name/'queries.jsonl.gz')) for name in names}
    geometry=list(jsonl(Path(cfg['geometry_source'])));oracle=np.array([g['oracle_recall'] for g in geometry])
    for name,rr in rows.items():assert [r['query_id'] for r in rr]==list(range(len(geometry)))
    return cfg,names,rows,geometry,oracle


def primary(config,run,tables,figures):
    assert json.loads((run/'derive_validation.json').read_text())['status']=='PASS'
    cfg,names,rows,geometry,oracle=load(config,run);k=int(cfg['k']);P=int(cfg['permutations'])
    tables.mkdir(parents=True,exist_ok=True);figures.mkdir(parents=True,exist_ok=True)
    aggregate=[];draws=[];closures=[];correlations=[];residual=[];model_draws={}
    for name in names:
        rr=rows[name];obs={'loss':np.array([r['observed_loss'] for r in rr]),'recall':np.array([r['observed_recall'] for r in rr]),
            'inversions':np.array([r['observed_inversions'] for r in rr]),'agreement':np.array([r['observed_agreement'] for r in rr]),
            'harmful_fraction':np.array([r['observed_loss']>0 for r in rr])}
        condition_draws={}
        for scope,mask in [('all',np.ones(len(rr),dtype=bool)),('oracle1',oracle==1)]:
            condition_draws[scope]={}
            for c in ['observed']+CONDITIONS:
                if c=='observed':data={key:np.full(P,float(value[mask].mean())) for key,value in obs.items()}
                else:
                    with np.load(run/name/f'{c}.npz') as f:
                        lh=f['loss_hits'][mask];data={'loss':lh.sum(0)/(int(mask.sum())*k),
                            'recall':f['hits'][mask].sum(0)/(int(mask.sum())*k),
                            'inversions':f['inversions'][mask].mean(0),'agreement':f['agreement_hits'][mask].sum(0)/(int(mask.sum())*k),
                            'harmful_fraction':(lh>0).mean(0)}
                condition_draws[scope][c]=data
                for metric,value in data.items():
                    aggregate.append({'model':name,'scope':scope,'condition':c,'metric':metric,'queries':int(mask.sum()),**summary(value)})
                    for p,v in enumerate(value):draws.append({'model':name,'scope':scope,'condition':c,'metric':metric,'permutation':p,'value':float(v)})
            for metric in ('loss','inversions'):
                n0=condition_draws[scope]['N0'][metric];o=float(obs[metric][mask].mean())
                for c in CONDITIONS:
                    v=condition_draws[scope][c][metric]
                    ratios=[closure(a,b,o,float(cfg['closure_min_denominator'])) for a,b in zip(n0,v)];valid=[v for v in ratios if v is not None]
                    record={'model':name,'scope':scope,'condition':c,'metric':metric,
                        'observed':o,'n0_mean':float(n0.mean()),'condition_mean':float(v.mean()),'denominator':float(n0.mean()-o),
                        'gap_closure':closure(float(n0.mean()),float(v.mean()),o,float(cfg['closure_min_denominator'])),
                        'invalid_draw_denominators':P-len(valid)}
                    if len(valid)>1:record.update({'draw_'+key:value for key,value in summary(valid).items()})
                    closures.append(record)
        model_draws[name]=condition_draws
        residual.extend({'model':name,**r} for r in json.loads((run/name/'residual_diagnostics.json').read_text()))
    for name in names+['five_model_mean']:
        used=names if name=='five_model_mean' else [name]
        obs=np.mean([[r['observed_loss'] for r in rows[n]] for n in used],axis=0)
        for scope,mask in [('all',np.ones(len(oracle),dtype=bool)),('oracle1',oracle==1)]:
            x=np.array([g['boundary_gap_r2'] for g in geometry])[mask]
            correlations.append({'model':name,'scope':scope,'condition':'observed','target':'loss','queries':int(mask.sum()),'spearman':spearman(x.tolist(),obs[mask].tolist())})
            for c in CONDITIONS:
                null=np.mean([[r['nulls'][c]['loss']['mean'] for r in rows[n]] for n in used],axis=0)
                excess=np.mean([[r['nulls'][c]['excess_loss'] for r in rows[n]] for n in used],axis=0)
                for target,y in [('loss',null),('excess',excess)]:correlations.append({'model':name,'scope':scope,'condition':c,'target':target,
                    'queries':int(mask.sum()),'spearman':spearman(x.tolist(),y[mask].tolist())})
    combined=[]
    for scope in ('all','oracle1'):
        for c in ['observed']+CONDITIONS:
            for metric in ('loss','recall','inversions','agreement','harmful_fraction'):
                values=np.array([model_draws[name][scope][c][metric] for name in names]);draw=values.mean(0);means=values.mean(1)
                combined.append({'scope':scope,'condition':c,'metric':metric,**summary(draw),
                    'model_mean_min':float(means.min()),'model_mean_max':float(means.max()),'model_mean_std':float(means.std(ddof=1))})
    for label,data in [('aggregate',aggregate),('aggregate_draws',draws),('gap_closure',closures),('geometry_correlations',correlations),
        ('residual_diagnostics',residual),('combined',combined)]:table(tables,label,data)
    dump(tables/'phase3h_primary_summary.json',{'combined':combined,'closure':closures,'geometry':correlations,
        'queries':len(oracle),'oracle1_queries':int((oracle==1).sum()),'permutations':P,'models':names,
        'n_distance_is_rank_quartiles':True})
    plots_primary(names,combined,closures,residual,correlations,figures)
    checkpoint=run/'primary_checkpoint.json'
    if not checkpoint.exists():
        dump(checkpoint,{'stage':'primary complete before PQ-only bridge',
            'tables_sha256':{str(p):digest(p) for p in sorted(tables.glob('phase3h_*'))},
            'figures_sha256':{str(p):digest(p) for p in sorted(figures.glob('phase3h_*'))}})
    print(json.dumps({'combined_all':[r for r in combined if r['scope']=='all'],'geometry_oracle1_mean':[r for r in correlations if r['scope']=='oracle1' and r['model']=='five_model_mean']},indent=2),flush=True)


def plots_primary(names,combined,closures,residual,correlations,out):
    import matplotlib
    matplotlib.use('Agg');matplotlib.rcParams['svg.hashsalt']='phase3h'
    import matplotlib.pyplot as plt
    labels=['observed']+CONDITIONS
    for metric in ('loss','inversions'):
        rr=[r for r in combined if r['scope']=='all' and r['metric']==metric]
        fig,ax=plt.subplots(figsize=(9,4));ax.errorbar(range(7),[r['mean'] for r in rr],yerr=[r['std'] for r in rr],fmt='o-',capsize=3)
        ax.set_xticks(range(7),labels);ax.set(ylabel=metric+' (five-model mean; MC SD)');fig.tight_layout();savefig(fig,out/f'phase3h_{metric}.svg');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4))
    for name in names:
        rr=[r for r in closures if r['model']==name and r['scope']=='all' and r['metric']=='loss'];ax.plot(range(6),[r['gap_closure'] for r in rr],'o-',label=name)
    ax.axhline(1,color='k',ls='--',lw=.8);ax.set_xticks(range(6),CONDITIONS);ax.set(ylabel='Descriptive loss-gap closure');ax.legend();fig.tight_layout();savefig(fig,out/'phase3h_gap_closure.svg');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,key in zip(axes,('mae','variance','signed_mean')):
        for name in names:
            rr=[next(r for r in residual if r['model']==name and r['group']==f'rank_{j}') for j in range(1,31)]
            ax.plot(range(1,31),[r[key] for r in rr],label=name)
        ax.set(xlabel='Exact candidate rank',ylabel=key);ax.axvline(10.5,color='k',ls='--',lw=.7)
    axes[0].legend();fig.tight_layout();savefig(fig,out/'phase3h_residual_rank.svg');plt.close(fig)
    for target in ('loss','excess'):
        fig,ax=plt.subplots(figsize=(9,4));cs=labels if target=='loss' else CONDITIONS
        for name in names+['five_model_mean']:
            rr=[next(r for r in correlations if r['model']==name and r['scope']=='oracle1' and r['target']==target and r['condition']==c) for c in cs]
            ax.plot(range(len(cs)),[r['spearman'] for r in rr],'o-',label=name)
        ax.set_xticks(range(len(cs)),cs);ax.set(ylabel='Spearman(d12-d9, '+target+'); oracle=1');ax.legend(fontsize=8);fig.tight_layout()
        savefig(fig,out/f'phase3h_geometry_{target}.svg');plt.close(fig)


def bridge(config,run,tables,figures,out):
    cfg,names,rows,geometry,oracle=load(config,run);checkpoint=json.loads((run/'primary_checkpoint.json').read_text())
    for path,h in checkpoint['tables_sha256'].items():assert digest(path)==h,path
    out.mkdir(parents=True,exist_ok=False);correlations=[];binned=[];features={};feature_rows=[]
    for name in names:
        score=np.memmap(Path(cfg['model_source'])/name/'scores.f32',dtype='<f4',mode='r');offset=0;ff=[]
        for r in rows[name]:
            n=r['candidate_count'];values=pq_observables(score[offset:offset+n],float(cfg['epsilon']));offset+=n
            ff.append(values);feature_rows.append({'model':name,'query_id':r['query_id'],**values,
                'ranking_loss_label':r['observed_loss'],'oracle_recall_label_control':r['oracle_recall']})
        assert offset==len(score);features[name]=ff
        loss=np.array([r['observed_loss'] for r in rows[name]])
        for scope,mask in [('all',np.ones(len(loss),dtype=bool)),('oracle1',oracle==1)]:
            for key in OBSERVABLES:
                x=np.array([f[key] for f in ff])[mask];y=loss[mask];labels=quintiles(x,int(cfg['observable_quintiles']))
                rho=spearman(x.tolist(),y.tolist());correlations.append({'model':name,'scope':scope,'observable':key,'queries':len(x),'spearman':rho,'populated_quintiles':len(set(labels))})
                for b in sorted(set(labels)):
                    m=labels==b;binned.append({'model':name,'scope':scope,'observable':key,'quintile':int(b),
                        'queries':int(m.sum()),'observable_min':float(x[m].min()),'observable_max':float(x[m].max()),
                        'observable_mean':float(x[m].mean()),'mean_loss':float(y[m].mean()),'harmful_rate':float((y[m]>0).mean())})
    ranking=[]
    for key in OBSERVABLES:
        rs=[r['spearman'] for r in correlations if r['observable']==key and r['scope']=='oracle1']
        valid=all(r is not None for r in rs);consistent=valid and (all(r>0 for r in rs) or all(r<0 for r in rs))
        ranking.append({'observable':key,'consistent_sign':consistent,'minimum_absolute_spearman':min(abs(r) for r in rs) if valid else None,
            'median_absolute_spearman':float(np.median(np.abs(rs))) if valid else None,'spearman_by_model':rs})
    eligible=[r for r in ranking if r['consistent_sign']]
    winner=sorted(eligible,key=lambda r:(-r['minimum_absolute_spearman'],-r['median_absolute_spearman'],r['observable']))[0] if eligible else None
    with gz_writer(out/'observables.jsonl.gz') as f:
        for row in feature_rows:f.write(json.dumps(row,allow_nan=False)+'\n')
    for label,data in [('bridge_correlations',correlations),('bridge_quintiles',binned),('bridge_feature_ranking',ranking)]:table(tables,label,data)
    result={'selected_observable':winner,'feature_ranking':ranking,'reconstruction_error_scalar':'not present in saved inputs; omitted',
        'primary_checkpoint_sha256':digest(run/'primary_checkpoint.json'),'feature_function_accepts_only_pq_scores':True,
        'fixed_pool_selection_is_exact_not_deployed_pq_search':True,'selection_is_exploratory_not_heldout':True}
    dump(tables/'phase3h_bridge_summary.json',result);dump(out/'validation.json',result)
    import matplotlib
    matplotlib.use('Agg');matplotlib.rcParams['svg.hashsalt']='phase3h'
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,4,figsize=(17,8))
    for ax,key in zip(axes.flat,OBSERVABLES):
        for name in names:
            rr=[r for r in binned if r['model']==name and r['scope']=='all' and r['observable']==key]
            ax.plot([r['quintile']+1 for r in rr],[r['mean_loss'] for r in rr],'o-',label=name)
        ax.set(title=key,xlabel='PQ-only observable quintile (ascending)',ylabel='Mean ranking loss')
    axes.flat[-1].axis('off');axes.flat[0].legend(fontsize=8);fig.tight_layout();savefig(fig,figures/'phase3h_bridge_quintiles.svg');plt.close(fig)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['primary','bridge'])
    p.add_argument('--config',type=Path,default=Path('configs/indexes/phase3h_rank_conditioned_nulls.conf'))
    p.add_argument('--run',type=Path,default=Path('runs/phase3h_rank_conditioned_nulls_v1'));p.add_argument('--tables',type=Path,default=Path('results/tables'))
    p.add_argument('--figures',type=Path,default=Path('results/figures'));p.add_argument('--bridge-output',type=Path)
    a=p.parse_args()
    if a.stage=='primary':primary(a.config,a.run,a.tables,a.figures)
    else:bridge(a.config,a.run,a.tables,a.figures,a.bridge_output or a.run/'bridge')
