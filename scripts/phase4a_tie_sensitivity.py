#!/usr/bin/env python3
"""Post-primary tie-aware sensitivity ONLY; primary ID-based metrics unchanged."""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np
from analyze_phase3a_sift1m import read_conf
from analyze_phase3b_fixed_candidate_ranking import dump
from analyze_phase4a import arrays,table


def tie_hits(distances,gt_distances):
    threshold=gt_distances[9]
    slots=10-int(np.sum(np.asarray(gt_distances[:10])<threshold))
    d=np.asarray(distances)
    return int((d<threshold).sum())+min(slots,int((d==threshold).sum()))


def main(out):
    c=read_conf(Path('configs/indexes/phase4a_gap_guided_refinement.conf'));root=Path(c['run']);n=int(c['query_count'])
    assert (root/'analysis/primary_checkpoint.json').exists()
    rows=[json.loads(l) for l in gzip.open(root/'analysis/queries.jsonl.gz','rt')];a=arrays(root,n)
    names=['native','oracle','top16','top32','top64'];hits={key:np.zeros(n,dtype=int) for key in names};controls=[]
    for qi in range(n):
        sl=slice(a['offset'][qi],a['offset'][qi+1]);lookup=dict(zip(map(int,a['ids'][sl]),map(float,a['exact'][sl])))
        for name in names:hits[name][qi]=tie_hits([lookup[v] for v in rows[qi][name+'_ids']],a['gd'][qi])
        ne=tie_hits(a['ens'][qi],a['gd'][qi]);oe=tie_hits(a['eos'][qi],a['gd'][qi]);assert ne==oe;controls.append(ne)
    sel=np.load(root/'analysis/selections.npz');records=[]
    for scope,ix in [('all',np.arange(n))]+[(f'shard{s}',np.arange(s,n,4)) for s in range(4)]:
        nn=len(ix);bn=hits['native'][ix];base=bn.sum();gap=(hits['oracle'][ix]-bn).mean()/10
        for L in [16,32,64]:
            for f in [.1,.25,.5]:
                count=int(nn*f);gain=hits[f'top{L}']-hits['native']
                rr=np.array([(base+gain[sel[f'{scope}_random_order_{p}'][:count]].sum())/(10*nn) for p in range(100)])
                for policy in ['GAP','ORACLE-QUERY','FULL-LOSS-SELECTOR']:
                    selected=sel[f'{scope}_L{L}_f{f}_{policy}'];r=(base+gain[selected].sum())/(10*nn)
                    records.append({'scope':scope,'L':L,'fraction':f,'policy':policy,'metric':'tie-aware recall, unchanged primary selections',
                        'recall':float(r),'random_mean':float(rr.mean()),'random_p975':float(np.quantile(rr,.975)),
                        'advantage':float(r-rr.mean()),'exceeds_random_p975':bool(r>np.quantile(rr,.975)),
                        'candidate_recoverable_gap':float(gap)})
    table(out,'tie_sensitivity',records)
    good=[L for L in [16,32,64] if all(next(r for r in records if r['scope']=='all' and r['policy']=='GAP' and r['L']==L and r['fraction']==f)['exceeds_random_p975'] for f in [.25,.5])]
    summary={'metric_definition':'strict d<d10 hits plus min(GT boundary slots, retrieved d=d10 hits), divided by10; exact equality on integer SIFT squared distances',
        'post_primary_sensitivity':True,'unchanged_GAP_and_random_selection':True,
        'primary_oracle_selectors_not_reoptimized_for_secondary_metric':True,
        'tie_aware_exact_control_discrepancies':0,'tie_aware_exact_native_recall':float(np.mean(controls)/10),
        'tie_aware_recalls':{k:float(v.mean()/10) for k,v in hits.items()},'passing_depths':good,'useful_criterion_pass':len(good)>=2}
    dump(out/'phase4a_tie_sensitivity_summary.json',summary);print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tables',default='results/tables');args=p.parse_args();main(Path(args.tables))
