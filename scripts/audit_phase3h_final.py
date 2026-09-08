#!/usr/bin/env python3
"""Reproduction and immutable-artifact audit for completed Phase3H."""
import argparse
import json
from pathlib import Path
from analyze_phase3a_sift1m import read_conf
from analyze_phase3b_fixed_candidate_ranking import digest,dump


def audit(config,run,reproduction):
    cfg=read_conf(config);v=json.loads((run/'verification.json').read_text());assert v['status']=='PASS'
    provenance=json.loads((run/'provenance.json').read_text())
    for p,h in provenance['input_sha256'].items():assert digest(p)==h,p
    for p,h in provenance['source_sha256'].items():assert digest(p)==h,p
    matched=[]
    for kind in ('tables','figures'):
        for p in sorted((Path('results')/kind).glob('phase3h_*')):
            other=reproduction/kind/p.name;assert digest(p)==digest(other),p;matched.append(str(p))
    assert digest(run/'bridge/observables.jsonl.gz')==digest(reproduction/'bridge/observables.jsonl.gz')
    for field in ('tables_sha256','figures_sha256'):
        for p,h in json.loads((run/'primary_checkpoint.json').read_text())[field].items():assert digest(p)==h,p
    files=[p for p in run.rglob('*') if p.is_file() and p.name!='final_audit.json']
    files += [Path(p) for p in matched]
    files += [config,Path('docs/phase3h_rank_conditioned_nulls.md'),Path('tests/test_phase3h_metrics.py')]
    files += [p for p in Path('scripts').glob('*phase3h*.py')]
    dump(run/'final_audit.json',{'status':'PASS','byte_identical_tables_and_figures':len(matched),
        'byte_identical_bridge_observables':True,'independent_reproduction_directory':str(reproduction),
        'primary_checkpoint_unchanged_after_bridge':True,'all_frozen_inputs_unchanged':True,
        'saved_samples_audited':v['total_saved_samples_audited'],'sampled_full_permutation_replays':v['total_sampled_replays'],
        'sha256':{str(p):digest(p) for p in files}})
    print(json.dumps({'status':'PASS','byte_identical_files':len(matched),'saved_samples_audited':v['total_saved_samples_audited'],
        'sampled_replays':v['total_sampled_replays']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=Path('configs/indexes/phase3h_rank_conditioned_nulls.conf'))
    p.add_argument('--run',type=Path,default=Path('runs/phase3h_rank_conditioned_nulls_v1'));p.add_argument('--reproduction',type=Path,required=True)
    a=p.parse_args();audit(a.config,a.run,a.reproduction)
