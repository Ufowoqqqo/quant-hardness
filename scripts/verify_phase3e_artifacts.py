#!/usr/bin/env python3
"""Frozen-input/model integrity and independent Phase 3E reproduction audit."""
import argparse
import json
from pathlib import Path
from analyze_phase3b_fixed_candidate_ranking import digest, dump
from analyze_phase3d_quantizer_seed_stability import verify_hashes
from analyze_phase3e_training_sample_stability import NAMES, validate_models


def main():
    p=argparse.ArgumentParser();p.add_argument("run",type=Path);p.add_argument("reproduction",type=Path)
    p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    assert not a.output.exists(),"refusing to overwrite verification artifact"
    for file in ("input_provenance.json","secondary_provenance.json"):
        verify_hashes(json.loads((a.run/file).read_text())["input_sha256"])
    manifests,_=validate_models(a.run/"resolved_config.conf",a.run)
    for name,m in zip(NAMES,manifests):
        assert digest("build/faiss_sample_stability")==m["binary_sha256"]
        assert digest("src/experiments/faiss_sift1m.cpp")==m["source_sha256"]
        assert digest(a.run/name/"resolved_config.conf")==m["config_sha256"]
    checked=[]
    for directory in ("analysis","ensemble"):
        for original in sorted((a.run/directory).iterdir()):
            if not original.is_file():continue
            other=a.reproduction/directory/original.name
            assert other.exists() and digest(original)==digest(other),str(original)
            checked.append(str(original))
    for directory in ("tables","figures"):
        for original in sorted((Path("results")/directory).glob("phase3e_*")):
            other=a.reproduction/directory/original.name
            assert other.exists() and digest(original)==digest(other),str(original)
            checked.append(str(original))
    paths=[p for p in a.run.rglob("*") if p.is_file()]
    paths+=list(Path("results/tables").glob("phase3e_*"))+list(Path("results/figures").glob("phase3e_*"))
    paths+=list(Path("scripts").glob("*phase3e*"))+list(Path("tests").glob("*phase3e*"))
    paths+=[Path("docs/phase3e_training_sample_stability.md"),Path("configs/indexes/phase3e_training_sample_stability.conf")]
    dump(a.output,{"status":"PASS","reproduced_files":checked,"sources_and_historical_O_unchanged":True,
        "nine_models_samples_scores_validated":True,"sha256":{str(p):digest(p) for p in sorted(paths)}})
    print(f"verification PASS reproduced_files={len(checked)} hashed_files={len(paths)}")


if __name__=="__main__":main()
