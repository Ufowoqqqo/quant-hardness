#!/usr/bin/env python3
"""Reproducible fixed-pool 5x3 basis stability, with post-primary diagnostics."""
import argparse
import itertools
import json
import shutil
from pathlib import Path
import numpy as np
from analyze_phase3a_sift1m import read_conf, spearman, write_csv
from analyze_phase3b_fixed_candidate_ranking import digest, dump
from analyze_phase3c_precision_transition import read_candidates, jsonl, gz_writer, stats
from analyze_phase3d_quantizer_seed_stability import verify_hashes, variability, savefig
from phase3d_metrics import single, GEOMETRY
from phase3f_metrics import measure_rotations, NAMES, CATEGORIES

FIELDS=['fixed_candidate_recall','ranking_recall_loss','cross_boundary_inversion_count','global_mae','random_pair_inversion_rate','topk_agreement']
PAIR_FIELDS=['harmful_jaccard','loss_spearman','inversion_spearman','pair_jaccard']
VAR_FIELDS=['within_rotation_initialization_variance','between_rotation_variance','total_model_variance',
            'within_init_sample_variance','rotation_means_sample_variance','signed_rotation_variance_contrast',
            'two_way_init_main_variance','two_way_rotation_main_variance','two_way_interaction_variance']


def table(root,name,rows):
    write_csv(root/f'phase3f_{name}.csv',rows);dump(root/f'phase3f_{name}.json',rows)


def distribution(values):
    values=[x for x in values if x is not None]
    return {**stats(values),'std':float(np.std(values,ddof=1)) if len(values)>1 else None}


def validate(config,run):
    cfg=read_conf(config);v=json.loads((run/'invariance.json').read_text());assert v['status']=='PASS'
    verify_hashes(json.loads((run/'input_provenance.json').read_text())['input_sha256'])
    manifests=[]
    for m,name in enumerate(NAMES):
        root=run/name;data=json.loads((root/'manifest.json').read_text());manifests.append(data)
        assert data['model_name']==name and data['rotation_index']==m//3 and data['initialization_index']==m%3+1
        assert data['seed']==int(cfg['training_seeds'].split(',')[m%3])
        assert data['training_sample_seed']==int(cfg['training_sample_seed']) and data['training_count']==65536
        for file,key in (('scores.f32','scores_sha256'),('pq64.index','model_file_sha256'),('training_ids.i32','training_ids_sha256')):
            assert digest(root/file)==data[key],(name,file)
        rr=v['rotations'][m//3]
        for key,target in (('rotation_sha256','R_sha256'),('rotated_base_sha256','base_sha256'),('rotated_queries_sha256','queries_sha256')):assert data[key]==rr[target]
    for key in ('training_ids_sha256','requests_sha256','graph_fingerprint','config_sha256','faiss_commit'):
        assert len({m[key] for m in manifests})==1,key
    assert len({m['codebook_sha256'] for m in manifests})==15
    for r in range(5):assert len({m['training_vectors_sha256'] for m in manifests[3*r:3*r+3]})==1
    # Identity models reproduce3E A1-A3, including scores; stronger regression
    # than aggregate agreement and no reuse of existing PQ codebooks.
    for i in range(3):
        old=json.loads((Path('runs/phase3e_training_sample_stability_v1')/f'A{i+1}'/'manifest.json').read_text())
        for key in ('training_ids_sha256','training_vectors_sha256','codebook_sha256','codes_sha256','scores_sha256'):
            assert manifests[i][key]==old[key],('identity regression',i,key)
    return manifests


def derive(config,run,out):
    cfg=read_conf(config);manifests=validate(config,run)
    maps=[np.memmap(run/name/'scores.f32',dtype='<f4',mode='r') for name in NAMES]
    prior=iter(jsonl(Path(cfg['phase3c_analysis'])/'queries.jsonl'))
    randomref=np.memmap(Path(cfg['phase3c_analysis'])/'random_pair_indices.bin',dtype='<i4',mode='r').reshape(10000,-1,2)
    out.mkdir(parents=True,exist_ok=False);offset=0;hist=np.zeros((16,6),dtype=np.int64)
    inter=np.zeros((15,15),dtype=np.int64);union=inter.copy()
    with gz_writer(out/'queries.jsonl.gz') as output,gz_writer(out/'replacement_pairs.jsonl.gz') as rep,gz_writer(out/'inversion_pairs.jsonl.gz') as pairs:
        for meta,a in read_candidates(Path(cfg['candidate_source']),jsonl(Path(cfg['candidate_source'])/'queries.jsonl')):
            qid=meta['query_id'];n=len(a);old=next(prior)
            assert old['query_id']==qid
            row,repl,ps,random=measure_rotations(a['id'],a['exact'],np.array([p[offset:offset+n] for p in maps]),meta['ground_truth_ids'],
                int(cfg['k']),int(cfg['random_pair_seed'])+qid,int(cfg['random_pairs_per_query']),float(cfg['epsilon']))
            assert np.array_equal(random,randomref[qid])
            for key in GEOMETRY+['exact_topk_ids','exact_topk_set','exact_boundary_tie']:assert row[key]==old[key],(qid,key)
            row.update(query_id=qid,candidate_order_sha256=meta['candidate_order_sha256'],ground_truth_ids=meta['ground_truth_ids'],score_offset=offset)
            assert row['candidate_order_sha256']==old['candidate_order_sha256'] and row['ground_truth_ids']==old['ground_truth_ids']
            output.write(json.dumps(row,allow_nan=False)+'\n')
            for r in repl:r['query_id']=qid;rep.write(json.dumps(r,allow_nan=False)+'\n')
            for r in ps:
                r['query_id']=qid;pairs.write(json.dumps(r,allow_nan=False)+'\n')
                mask=np.array(r['model_mask']);inter+=mask[:,None]&mask[None,:];union+=mask[:,None]|mask[None,:]
                hist[r['model_count'],r['rotation_frequency']]+=1
            offset+=n
            if (qid+1)%1000==0:print(f'derive {qid+1}/10000',flush=True)
    assert next(prior,None) is None and all(len(p)==offset for p in maps) and offset==10353047
    dump(out/'pair_summary.json',{'histogram':hist.tolist(),'intersection':inter.tolist(),'union':union.tolist()})
    dump(out/'validation.json',{'queries':10000,'candidates':offset,'models':NAMES,'identity_models_reproduced_3E':True,
        'exact_gt_geometry_candidate_order_unchanged':True,'random_pairs_unchanged':True,'variance_identity':True,
        'model_manifest_sha256':{name:digest(run/name/'manifest.json') for name in NAMES},
        'analysis_source_sha256':digest(__file__),'metrics_source_sha256':digest(Path(__file__).with_name('phase3f_metrics.py'))})


def pairwise(rows,pairdata=None):
    loss=np.array([[r['models'][m]['ranking_recall_loss'] for r in rows] for m in range(15)])
    inv=np.array([[r['models'][m]['cross_boundary_inversion_count'] for r in rows] for m in range(15)])
    records=[]
    for a,b in itertools.combinations(range(15),2):
        A=loss[a]>0;B=loss[b]>0;u=int((A|B).sum());pa=float(A.mean());pb=float(B.mean())
        records.append({'model_a':NAMES[a],'model_b':NAMES[b],'rotation_a':a//3,'rotation_b':b//3,
            'relation':'within' if a//3==b//3 else 'between','same_init':a%3==b%3,
            'basis_relation':'same' if a//3==b//3 else 'identity_to_random' if a//3==0 else 'random_to_random',
            'harmful_prevalence_a':pa,'harmful_prevalence_b':pb,'harmful_jaccard':float((A&B).sum()/u) if u else 1.,
            'independent_marginal_jaccard':pa*pb/(pa+pb-pa*pb) if pa+pb else 1.,
            'loss_spearman':spearman(loss[a].tolist(),loss[b].tolist()),'inversion_spearman':spearman(inv[a].tolist(),inv[b].tolist()),
            'pair_jaccard':pairdata['intersection'][a][b]/pairdata['union'][a][b] if pairdata and pairdata['union'][a][b] else None})
    return records


def summarize(config,run,derived,tables,figures):
    cfg=read_conf(config);rows=list(jsonl(derived/'queries.jsonl.gz'));assert len(rows)==10000
    tables.mkdir(parents=True,exist_ok=True);figures.mkdir(parents=True,exist_ok=True)
    aggregate=[]
    for m,name in enumerate(NAMES):
        manifest=json.loads((run/name/'manifest.json').read_text())
        aggregate.append({'model':name,'rotation':m//3,'initialization':m%3+1,
            **{key:float(np.mean([r['models'][m][key] for r in rows])) for key in FIELDS},
            'harmful_prevalence':float(np.mean([r['models'][m]['harmful'] for r in rows])),
            'candidate_weighted_mae':float(sum(r['models'][m]['global_mae']*r['candidate_count'] for r in rows)/sum(r['candidate_count'] for r in rows)),
            'codebook_sha256':manifest['codebook_sha256'],'training_ids_sha256':manifest['training_ids_sha256']})
    table(tables,'models',aggregate)
    quality=[{'rotation':r,'metric':key,**variability([m[key] for m in aggregate if m['rotation']==r])} for r in range(5) for key in FIELDS+['harmful_prevalence','candidate_weighted_mae']]
    table(tables,'rotation_quality',quality)
    pd=json.loads((derived/'pair_summary.json').read_text());pairs=pairwise(rows,pd);table(tables,'pairwise',pairs)
    distributions=[]
    groups={'within':[p for p in pairs if p['relation']=='within'],'between':[p for p in pairs if p['relation']=='between'],
        'identity_to_random':[p for p in pairs if p['basis_relation']=='identity_to_random'],
        'random_to_random':[p for p in pairs if p['basis_relation']=='random_to_random'],
        'between_matched_init':[p for p in pairs if p['relation']=='between' and p['same_init']],
        'between_unmatched_init':[p for p in pairs if p['relation']=='between' and not p['same_init']]}
    for label,values in groups.items():
        for key in PAIR_FIELDS+['independent_marginal_jaccard']:distributions.append({'group':label,'metric':key,'pairs':len(values),**distribution([p[key] for p in values])})
    table(tables,'pair_distributions',distributions)
    sensitivity=[]
    for scope in ('oracle1','no_exact_boundary_ties'):
        selected=[r for r in rows if (r['oracle_recall']==1 if scope=='oracle1' else not r['exact_boundary_tie'])]
        sensitivity.extend({'scope':scope,'queries':len(selected),**r} for r in pairwise(selected))
    table(tables,'stability_sensitivity',sensitivity)
    table(tables,'variance',[{'metric':key,**distribution([r[key] for r in rows])} for key in VAR_FIELDS])
    categories=[];geometry=[];correlations=[];bins=[];mean_vs_realized=[]
    for scope in ('all','oracle1'):
        selected=rows if scope=='all' else [r for r in rows if r['oracle_recall']==1]
        total=sum(r['overall_mean_loss']*15 for r in selected)
        for category in CATEGORIES:
            members=[r for r in selected if r['category']==category];loss=sum(r['overall_mean_loss']*15 for r in members)
            categories.append({'scope':scope,'category':category,'queries':len(members),'fraction':len(members)/len(selected),
                'signed_loss_sum':loss,'fraction_total_signed_loss':loss/total if total else None,
                'positive_loss_sum':sum(max(m['ranking_recall_loss'],0) for r in members for m in r['models'])})
            for key in GEOMETRY:geometry.append({'scope':scope,'category':category,'descriptor':key,'queries':len(members),**stats([r[key] for r in members])})
        for key in GEOMETRY:
            xs=[r[key] for r in selected]
            for target in ('overall_mean_loss','harmful_rotation_frequency','within_rotation_initialization_variance','between_rotation_variance'):
                correlations.append({'scope':scope,'descriptor':key,'target':target,'queries':len(selected),'spearman':spearman(xs,[r[target] for r in selected])})
                edges=np.unique(np.quantile(xs,np.linspace(0,1,int(cfg['quantile_bins'])+1)));labels=np.searchsorted(edges[1:-1],xs,side='right')
                for b in sorted(set(labels.tolist())):
                    members=[r for r,l in zip(selected,labels) if l==b]
                    bins.append({'scope':scope,'descriptor':key,'target':target,'bin':int(b),'queries':len(members),
                        'descriptor_mean':float(np.mean([r[key] for r in members])),'target_mean':float(np.mean([r[target] for r in members]))})
            for m in range(15):mean_vs_realized.append({'scope':scope,'descriptor':key,'model':NAMES[m],
                'realized_loss_spearman':spearman(xs,[r['models'][m]['ranking_recall_loss'] for r in selected]),
                'mean_loss_spearman':spearman(xs,[r['overall_mean_loss'] for r in selected])})
    table(tables,'categories',categories);table(tables,'geometry',geometry);table(tables,'geometry_correlations',correlations)
    table(tables,'geometry_bins',bins);table(tables,'geometry_mean_vs_realized',mean_vs_realized)
    freq=[{'harmful_rotation_count':i,'frequency':i/5,'queries':sum(r['harmful_rotation_count']==i for r in rows)} for i in range(6)]
    table(tables,'harmful_rotation_frequency',freq)
    hist=np.array(pd['histogram']);total=int(hist.sum())
    pair_frequency={'union_pairs':total,'one_model_fraction':float(hist[1].sum()/total),'one_rotation_fraction':float(hist[:,1].sum()/total),
        'at_least_two_rotations_fraction':float(hist[:,2:].sum()/total),'all_five_rotations_fraction':float(hist[:,5].sum()/total),
        'all15_models_fraction':float(hist[15].sum()/total)}
    table(tables,'inversion_frequency',[{'model_count':m,'rotation_count':r,'pairs':int(hist[m,r]),'fraction_union':float(hist[m,r]/total)} for m in range(1,16) for r in range(1,6)])
    dump(tables/'phase3f_pair_frequency_summary.json',pair_frequency)
    w=float(np.mean([r['within_rotation_initialization_variance'] for r in rows]));b=float(np.mean([r['between_rotation_variance'] for r in rows]))
    loss=np.array([[r['models'][m]['ranking_recall_loss'] for r in rows] for m in range(15)])
    prevalence=(loss>0).mean(axis=1);majority=np.array([r['rotation_harmful_frequency'] for r in rows])>=2/3
    summary={'queries':len(rows),'oracle_recall':float(np.mean([r['oracle_recall'] for r in rows])),
        'model_quality':aggregate,'rotation_quality':quality,'pair_distributions':distributions,'categories':categories,'pair_frequency':pair_frequency,
        'variance_means':{key:float(np.mean([r[key] for r in rows])) for key in VAR_FIELDS},
        'within_observed_variance_share':w/(w+b),'between_observed_variance_share':b/(w+b),
        'negative_query_model_losses':int((loss<0).sum()),'negative_variance_contrasts':sum(r['signed_rotation_variance_contrast'] < -1e-14 for r in rows),
        'zero_model_variance_queries':sum(r['total_model_variance']<1e-14 for r in rows),
        'model_harmful_prevalence':prevalence.tolist(),'rotation_majority_prevalence':majority.mean(axis=0).tolist(),
        'independent_rotation_marginals_expected_persistent':float(10000*np.prod(majority.mean(axis=0))),
        'oracle1_queries':sum(r['oracle_recall']==1 for r in rows),'exact_boundary_tie_queries':sum(r['exact_boundary_tie'] for r in rows),
        'pq_boundary_tie_queries':[sum(r['models'][m]['pq_boundary_tie'] for r in rows) for m in range(15)]}
    dump(tables/'phase3f_summary.json',summary);plots(rows,pairs,freq,bins,hist,quality,figures)
    print(json.dumps({'rotation_quality':quality,'pair_distributions':distributions,'pair_frequency':pair_frequency}),flush=True)


def plots(rows,pairs,freq,bins,hist,quality,out):
    import matplotlib
    matplotlib.use('Agg');matplotlib.rcParams['svg.hashsalt']='phase3f'
    import matplotlib.pyplot as plt
    for metric,name in (('harmful_jaccard','jaccard'),('loss_spearman','loss_spearman')):
        fig,ax=plt.subplots(figsize=(6,4));ax.boxplot([[r[metric] for r in pairs if r['relation']==rel] for rel in ('within','between')],tick_labels=['Within rotation (15)','Between rotation (90)'])
        ax.set(ylabel=metric);fig.tight_layout();savefig(fig,out/f'phase3f_{name}.svg');plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4));ax.bar([r['harmful_rotation_count'] for r in freq],[r['queries'] for r in freq]);ax.set(xlabel='Rotations with >=2/3 harmful initializations',ylabel='Queries');fig.tight_layout();savefig(fig,out/'phase3f_harmful_frequency.svg');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,target in zip(axes,('overall_mean_loss','harmful_rotation_frequency')):
        for scope in ('all','oracle1'):
            v=[r for r in bins if r['scope']==scope and r['descriptor']=='boundary_gap_r2' and r['target']==target]
            ax.plot([r['descriptor_mean'] for r in v],[r['target_mean'] for r in v],'o-',label=scope)
        ax.set(xlabel='Exact d12-d9',ylabel=target);ax.legend()
    fig.tight_layout();savefig(fig,out/'phase3f_geometry.svg');plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4));im=ax.hexbin([r['within_rotation_initialization_variance'] for r in rows],[r['between_rotation_variance'] for r in rows],gridsize=30,mincnt=1);fig.colorbar(im,ax=ax,label='Queries');ax.set(xlabel='Within-init population variance',ylabel='Variance of rotation means');fig.tight_layout();savefig(fig,out/'phase3f_variance.svg');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4));axes[0].bar(range(1,16),hist[1:].sum(1));axes[1].bar(range(1,6),hist[:,1:].sum(0));axes[0].set(xlabel='Models inverting pair',ylabel='Union pairs');axes[1].set(xlabel='Rotations inverting pair (>=1 init)',ylabel='Union pairs');fig.tight_layout();savefig(fig,out/'phase3f_pair_frequency.svg');plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4));data=[r for r in quality if r['metric']=='fixed_candidate_recall'];ax.errorbar(range(5),[r['mean'] for r in data],yerr=[r['std'] for r in data],fmt='o-',capsize=3);ax.set_xticks(range(5),['Identity','R1','R2','R3','R4']);ax.set(ylabel='Recall@10 (mean +/- init sample SD)');fig.tight_layout();savefig(fig,out/'phase3f_rotation_recall.svg');plt.close(fig)


def checkpoint(run,tables):
    assert not (run/'ensemble').exists();out=run/'primary_checkpoint';out.mkdir(exist_ok=False)
    for path in sorted(tables.glob('phase3f_*')):shutil.copyfile(path,out/path.name)
    dump(out/'checkpoint.json',{'stage':'primary complete before diagnostics','summary_sha256':digest(out/'phase3f_summary.json')})


def ensemble(config,run,out,tables,figures):
    cfg=read_conf(config);check=run/'primary_checkpoint'/'phase3f_summary.json';primary=json.loads(check.read_text())
    maps=[np.memmap(run/name/'scores.f32',dtype='<f4',mode='r') for name in NAMES]
    groups={'rotation_I1':[0,3,6,9,12],**{f'init_R{r}':list(range(r*3,r*3+3)) for r in range(5)}}
    out.mkdir(parents=True,exist_ok=False);records=[];offset=0
    with gz_writer(out/'queries.jsonl.gz') as target:
        for meta,a in read_candidates(Path(cfg['candidate_source']),jsonl(Path(cfg['candidate_source'])/'queries.jsonl')):
            qid=meta['query_id'];n=len(a);row={'query_id':qid,'candidate_order_sha256':meta['candidate_order_sha256']}
            for name,ms in groups.items():
                score=np.array([maps[m][offset:offset+n] for m in ms],dtype=float).mean(0)
                metric,g,_,_=single(a['id'],a['exact'],score,meta['ground_truth_ids'],int(cfg['k']),int(cfg['random_pair_seed'])+qid,int(cfg['random_pairs_per_query']),float(cfg['epsilon']))
                row[name]=metric;row['oracle_recall']=g['oracle_recall']
            records.append(row);target.write(json.dumps(row,allow_nan=False)+'\n');offset+=n
    summary=[]
    for name,ms in groups.items():
        r={'diagnostic':name,'members':','.join(NAMES[m] for m in ms),'model_count':len(ms)}
        for key in ('fixed_candidate_recall','ranking_recall_loss','cross_boundary_inversion_count'):
            r[key]=float(np.mean([v[name][key] for v in records]));r['member_mean_'+key]=float(np.mean([primary['model_quality'][m][key] for m in ms]))
        r['fraction_member_loss_recovered']=1-r['ranking_recall_loss']/r['member_mean_ranking_recall_loss'];summary.append(r)
    table(tables,'ensemble',summary);dump(out/'summary.json',{'diagnostics':summary,'primary_checkpoint_sha256':digest(check),'accumulation':'uniform float64; five rotations versus three initializations is confounded by model count'})
    import matplotlib
    matplotlib.use('Agg');matplotlib.rcParams['svg.hashsalt']='phase3f'
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,key in zip(axes,('fixed_candidate_recall','ranking_recall_loss','cross_boundary_inversion_count')):
        ax.bar(range(6),[r[key] for r in summary]);ax.set_xticks(range(6),['5 bases']+[f'R{r}\n3 inits' for r in range(5)]);ax.set(ylabel=key)
    fig.tight_layout();savefig(fig,figures/'phase3f_ensemble.svg');plt.close(fig);print(json.dumps(summary),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['derive','summarize','checkpoint','ensemble'])
    p.add_argument('--config',type=Path,default=Path('configs/indexes/phase3f_rotation_stability.conf'))
    p.add_argument('--run',type=Path,default=Path('runs/phase3f_rotation_stability_v1'));p.add_argument('--derived',type=Path)
    p.add_argument('--tables',type=Path,default=Path('results/tables'));p.add_argument('--figures',type=Path,default=Path('results/figures'))
    a=p.parse_args();derived=a.derived or a.run/'analysis'
    if a.mode=='derive':derive(a.config,a.run,derived)
    elif a.mode=='summarize':summarize(a.config,a.run,derived,a.tables,a.figures)
    elif a.mode=='checkpoint':checkpoint(a.run,a.tables)
    else:ensemble(a.config,a.run,a.derived or a.run/'ensemble',a.tables,a.figures)


if __name__=='__main__':main()
