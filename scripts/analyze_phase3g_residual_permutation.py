#!/usr/bin/env python3
"""Regenerate Phase 3G tables/figures from preserved query and permutation data."""
import argparse
import json
from pathlib import Path
import numpy as np
from analyze_phase3a_sift1m import read_conf, spearman, write_csv
from analyze_phase3b_fixed_candidate_ranking import dump, digest
from analyze_phase3c_precision_transition import jsonl
from analyze_phase3d_quantizer_seed_stability import savefig
from phase3g_metrics import summary

DESCRIPTORS = ['boundary_gap_r1','boundary_gap_r2','boundary_gap_r5','relative_boundary_margin',
               'relative_span_r2','local_distance_concentration','candidate_count','d1','dk','dk_plus_1','d20','oracle_recall']


def table(root, name, rows):
    write_csv(root/f'phase3g_{name}.csv',rows); dump(root/f'phase3g_{name}.json',rows)


def safe_mean(x):
    a = [v for v in x if v is not None]; return float(np.mean(a)) if a else None


def analyze(config, run, tables, figures):
    assert json.loads((run/'derive_validation.json').read_text())['status'] == 'PASS'
    cfg = read_conf(config); names = cfg['models'].split(','); k = int(cfg['k'])
    tables.mkdir(parents=True,exist_ok=True); figures.mkdir(parents=True,exist_ok=True)
    geometry = list(jsonl(Path(cfg['geometry_source']))); nq = len(geometry)
    for g in geometry: g['relative_span_r2'] = g['boundary_gap_r2']/max(g['dk'],float(cfg['epsilon']))
    oracle = np.array([g['oracle_recall'] for g in geometry]); all_rows = {}; aggregate = []; draws = []; query_summary = []
    residual_ranks = []; regions = []; residual_dependence = []; diagnostic_bins = []; pair_comparisons = []; rank_summary = []
    residual_associations = []
    for name in names:
        rows = list(jsonl(run/name/'queries.jsonl.gz')); all_rows[name] = rows
        raw = np.load(run/name/'permutations.npz'); diag = np.load(run/name/'diagnostics.npz')
        assert [r['query_id'] for r in rows] == list(range(nq))
        for scope, mask in [('all',np.ones(nq,dtype=bool)),('oracle1',oracle==1)]:
            selected = [r for r,keep in zip(rows,mask) if keep]
            data = {'recall': (np.array([r['observed_recall'] for r in selected]),raw['hits'][mask]/k),
                    'loss': (np.array([r['observed_loss'] for r in selected]),raw['loss_hits'][mask]/k),
                    'inversions': (np.array([r['observed_inversions'] for r in selected]),raw['inversions'][mask]),
                    'harmful_fraction': (np.array([r['observed_loss']>0 for r in selected]),raw['loss_hits'][mask]>0)}
            for metric,(obs,null) in data.items():
                means = null.mean(0); s = summary(means); observed = float(obs.mean())
                aggregate.append({'model':name,'scope':scope,'queries':int(mask.sum()),'metric':metric,'observed':observed,
                    **{'null_'+key:value for key,value in s.items()},'observed_minus_null':observed-s['mean'],
                    'null_over_observed':s['mean']/observed if observed else None})
                for p,v in enumerate(means): draws.append({'model':name,'scope':scope,'metric':metric,'permutation':p,'value':float(v)})
            for field in ('loss_rank','inversion_rank'):
                values = [r[field]['midrank'] for r in selected]
                rank_summary.append({'model':name,'scope':scope,'metric':field,**summary(values),
                    'fraction_below_all_null':float(np.mean([r[field]['above']==int(cfg['permutations']) for r in selected])),
                    'fraction_above_all_null':float(np.mean([r[field]['below']==int(cfg['permutations']) for r in selected])),
                    'mean_equal_null_fraction':float(np.mean([r[field]['equal']/int(cfg['permutations']) for r in selected]))})
            for diagnostic in ('residual_distance_spearman','residual_norm_spearman'):
                vals = [r[diagnostic] for r in selected if r[diagnostic] is not None]
                residual_dependence.append({'model':name,'scope':scope,'diagnostic':diagnostic,'valid_queries':len(vals),**summary(vals)})
            targets = [r['excess_loss'] for r in selected]
            predictors = {'top10_minus_near_signed_bias':[r['regions']['top10']['mean']-r['regions']['near11_20']['mean'] for r in selected],
                'top10_minus_far_mae':[r['regions']['top10']['mae']-r['regions']['far21plus']['mae'] for r in selected],
                'top10_minus_far_signed_bias':[r['regions']['top10']['mean']-r['regions']['far21plus']['mean'] for r in selected],
                'residual_distance_spearman':[r['residual_distance_spearman'] for r in selected]}
            for key,xx in predictors.items():
                keep = [i for i,x in enumerate(xx) if x is not None]
                residual_associations.append({'model':name,'scope':scope,'predictor':key,'target':'excess_loss',
                    'spearman':spearman([xx[i] for i in keep],[targets[i] for i in keep])})
        for r in rows:
            query_summary.append({'model':name,'query_id':r['query_id'],'oracle_recall':r['oracle_recall'],
                'observed_loss':r['observed_loss'],'null_mean_loss':r['null_loss']['mean'],'excess_loss':r['excess_loss'],
                'observed_inversions':r['observed_inversions'],'null_mean_inversions':r['null_inversions']['mean'],
                'excess_inversions':r['excess_inversions'],'loss_midrank':r['loss_rank']['midrank']})
        rank = diag['rank_residuals']
        for j in range(rank.shape[1]):
            v = rank[:,j]; residual_ranks.append({'model':name,'exact_rank':j+1,'count':len(v),'signed_mean':float(v.mean()),
                'mae':float(np.abs(v).mean()),'variance':float(v.var()),'std':float(v.std()),'p05':float(np.quantile(v,.05)),'p95':float(np.quantile(v,.95))})
        for label in ('top10','near11_20','far21plus'):
            total = sum(r['regions'][label]['count'] for r in rows)
            mu = sum(r['regions'][label]['mean']*r['regions'][label]['count'] for r in rows)/total
            second = sum((r['regions'][label]['variance']+r['regions'][label]['mean']**2)*r['regions'][label]['count'] for r in rows)/total
            regions.append({'model':name,'region':label,'candidates':total,'candidate_weighted_signed_mean':mu,
                'candidate_weighted_mae':sum(r['regions'][label]['mae']*r['regions'][label]['count'] for r in rows)/total,
                'candidate_weighted_variance':second-mu**2,
                'query_mean_signed_residual':float(np.mean([r['regions'][label]['mean'] for r in rows])),
                'query_mean_mae':float(np.mean([r['regions'][label]['mae'] for r in rows]))})
        sample = diag['sampled_id_distance_residual_norm'].reshape(-1,4)
        for col, descriptor in [(1,'exact_distance'),(3,'candidate_norm')]:
            x = sample[:,col]; e = sample[:,2]; edges = np.unique(np.quantile(x,np.linspace(0,1,int(cfg['quantile_bins'])+1)))
            labels = np.searchsorted(edges[1:-1],x,side='right')
            for b in sorted(set(labels)):
                ix = labels==b; diagnostic_bins.append({'model':name,'descriptor':descriptor,'bin':int(b),'candidates':int(ix.sum()),
                    'descriptor_mean':float(x[ix].mean()),'signed_mean':float(e[ix].mean()),'mae':float(np.abs(e[ix]).mean()),'variance':float(e[ix].var())})
        pairs = list(jsonl(run/name/'replacement_pairs.jsonl.gz')); nt = diag['null_replacement_totals']
        pair_comparisons.append({'model':name,'observed_replacement_pairs':len(pairs),
            'observed_pair_mean_exact_margin':safe_mean([p['exact_margin'] for p in pairs]),
            'observed_pair_mean_error_difference':safe_mean([p['observed_error_difference'] for p in pairs]),
            'same_pair_shuffled_mean_error_difference':safe_mean([p['null_same_pair_difference']['mean'] for p in pairs]),
            'same_pair_null_inversion_fraction':safe_mean([p['null_same_pair_strict_inversion_fraction'] for p in pairs]),
            'shuffled_selected_pair_count_mean':float(nt[:,0].mean()),
            'shuffled_selected_pair_mean_error_difference':float(nt[:,1].sum()/nt[:,0].sum()),
            'shuffled_selected_pair_mean_exact_margin':float(nt[:,2].sum()/nt[:,0].sum())})
    correlations = []; bins = []
    for name in names+['five_model_mean']:
        ys = {target:np.mean([[r[key] if nested is None else r[key][nested] for r in all_rows[n]]
                             for n in (names if name=='five_model_mean' else [name])],axis=0)
              for target,key,nested in [('observed_loss','observed_loss',None),('null_mean_loss','null_loss','mean'),('excess_loss','excess_loss',None)]}
        for scope,mask in [('all',np.ones(nq,dtype=bool)),('oracle1',oracle==1)]:
            for descriptor in DESCRIPTORS:
                x = np.array([g[descriptor] for g in geometry])[mask]
                for target,y in ys.items():
                    y = y[mask]; correlations.append({'model':name,'scope':scope,'descriptor':descriptor,'target':target,
                        'queries':len(x),'spearman':spearman(x.tolist(),y.tolist())})
                    edges = np.unique(np.quantile(x,np.linspace(0,1,int(cfg['quantile_bins'])+1))); labels = np.searchsorted(edges[1:-1],x,side='right')
                    for b in sorted(set(labels)):
                        ix = labels==b; bins.append({'model':name,'scope':scope,'descriptor':descriptor,'target':target,'bin':int(b),'queries':int(ix.sum()),
                            'descriptor_mean':float(x[ix].mean()),'target_mean':float(y[ix].mean())})
    for name, data in [('aggregate',aggregate),('aggregate_permutation_draws',draws),('query_summary',query_summary),
        ('residual_rank',residual_ranks),('residual_regions',regions),('residual_dependence',residual_dependence),
        ('residual_bins',diagnostic_bins),('replacement_pairs',pair_comparisons),('geometry_correlations',correlations),
        ('geometry_bins',bins),('empirical_ranks',rank_summary),('residual_excess_associations',residual_associations)]: table(tables,name,data)
    result = {'models':names,'queries':nq,'oracle1_queries':int((oracle==1).sum()),'permutations':int(cfg['permutations']),
        'aggregate':aggregate,'geometry_d12_d9':[r for r in correlations if r['descriptor']=='boundary_gap_r2'],
        'residual_regions':regions,'pair_comparisons':pair_comparisons,
        'negative_observed_losses':sum(r['observed_loss']<0 for rows in all_rows.values() for r in rows),
        'negative_null_losses':sum(int((np.load(run/n/'permutations.npz')['loss_hits']<0).sum()) for n in names)}
    dump(tables/'phase3g_summary.json',result)
    plots(all_rows,bins,residual_ranks,aggregate,figures)
    print(json.dumps({'aggregate_all':[r for r in aggregate if r['scope']=='all'],'geometry_oracle1':[r for r in result['geometry_d12_d9'] if r['scope']=='oracle1']},indent=2),flush=True)


def plots(rows,bins,ranks,aggregate,out):
    import matplotlib
    matplotlib.use('Agg'); matplotlib.rcParams['svg.hashsalt']='phase3g'
    import matplotlib.pyplot as plt
    for metric,obs,nkey in [('loss','observed_loss','null_loss'),('inversions','observed_inversions','null_inversions')]:
        fig,axes = plt.subplots(1,5,figsize=(17,3.3))
        for ax,(name,rr) in zip(axes,rows.items()):
            x = [r[nkey]['mean'] for r in rr]; y = [r[obs] for r in rr]
            ax.hexbin(x,y,gridsize=30,mincnt=1); limit=max(max(x),max(y)); ax.plot([0,limit],[0,limit],'k--',lw=.7)
            ax.set(title=name,xlabel='Null mean '+metric,ylabel='Observed '+metric)
        fig.tight_layout(); savefig(fig,out/f'phase3g_observed_null_{metric}.svg'); plt.close(fig)
    for target in ('observed_loss','null_mean_loss','excess_loss'):
        fig,ax=plt.subplots(figsize=(6,4))
        for scope in ('all','oracle1'):
            rr=[r for r in bins if r['model']=='five_model_mean' and r['scope']==scope and r['descriptor']=='boundary_gap_r2' and r['target']==target]
            ax.plot([r['descriptor_mean'] for r in rr],[r['target_mean'] for r in rr],'o-',label=scope)
        ax.set(xlabel='Exact d12-d9 (quantile bins)',ylabel='Five-model mean '+target); ax.legend(); fig.tight_layout()
        savefig(fig,out/f'phase3g_geometry_{target}.svg'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4))
    for name,rr in rows.items(): ax.hist([r['excess_loss'] for r in rr],bins=np.linspace(-1,1,81),histtype='step',label=name)
    ax.set(xlabel='Observed loss - null mean loss',ylabel='Queries'); ax.legend(); fig.tight_layout(); savefig(fig,out/'phase3g_excess_loss.svg'); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,metric in zip(axes,('signed_mean','mae','variance')):
        for name in rows:
            rr=[r for r in ranks if r['model']==name]; ax.plot([r['exact_rank'] for r in rr],[r[metric] for r in rr],label=name)
        ax.axvline(10.5,color='k',ls='--',lw=.7); ax.set(xlabel='Exact candidate rank',ylabel='Residual '+metric)
    axes[0].legend(); fig.tight_layout(); savefig(fig,out/'phase3g_residual_rank.svg'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4)); rr=[r for r in aggregate if r['scope']=='all' and r['metric']=='recall']; x=np.arange(len(rr))
    ax.plot(x,[r['observed'] for r in rr],'o-',label='Observed PQ')
    ax.errorbar(x,[r['null_mean'] for r in rr],yerr=[r['null_std'] for r in rr],fmt='s-',label='Null mean +/- permutation SD')
    ax.set_xticks(x,[r['model'] for r in rr]); ax.set(ylabel='Fixed-candidate Recall@10'); ax.legend(); fig.tight_layout()
    savefig(fig,out/'phase3g_aggregate_recall.svg'); plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,default=Path('configs/indexes/phase3g_residual_permutation.conf'))
    p.add_argument('--run',type=Path,default=Path('runs/phase3g_residual_permutation_v1'))
    p.add_argument('--tables',type=Path,default=Path('results/tables')); p.add_argument('--figures',type=Path,default=Path('results/figures'))
    a=p.parse_args(); analyze(a.config,a.run,a.tables,a.figures)
