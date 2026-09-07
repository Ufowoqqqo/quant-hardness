#!/usr/bin/env python3
"""Verify frozen sources/models and independent regeneration, without writes to inputs."""
import argparse
import json
from pathlib import Path
from analyze_phase3b_fixed_candidate_ranking import digest, dump
from analyze_phase3d_quantizer_seed_stability import verify_hashes


def main():
    p=argparse.ArgumentParser();p.add_argument("run",type=Path);p.add_argument("reproduction",type=Path)
    p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    assert not a.output.exists(),"refusing to overwrite verification"
    provenance=json.loads((a.run/"input_provenance.json").read_text())
    verify_hashes(provenance["input_sha256"])
    manifest_paths=[]
    for s in range(5):
        directory=a.run/f"seed_{s}";m=json.loads((directory/"manifest.json").read_text())
        for file,key in (("pq64.index","model_file_sha256"),("scores.f32","scores_sha256"),("training_ids.i32","training_ids_sha256")):
            assert digest(directory/file)==m[key],(s,file)
        assert digest("build/faiss_seed_stability")==m["binary_sha256"]
        assert digest("src/experiments/faiss_sift1m.cpp")==m["source_sha256"]
        assert digest(directory/"resolved_config.conf")==m["config_sha256"]
        manifest_paths.append(directory/"manifest.json")
    checks=[]
    for sub in ("analysis","ensemble"):
        for original in sorted((a.run/sub).glob("*")):
            if not original.is_file():continue
            other=a.reproduction/sub/original.name
            assert other.exists() and digest(original)==digest(other),str(original)
            checks.append(str(original))
    for sub in ("tables","figures"):
        for original in sorted((Path("results")/sub).glob("phase3d_*")):
            other=a.reproduction/sub/original.name
            assert other.exists() and digest(original)==digest(other),str(original)
            checks.append(str(original))
    current=[x for x in a.run.rglob("*") if x.is_file()]
    current+=list(Path("results/tables").glob("phase3d_*"))+list(Path("results/figures").glob("phase3d_*"))
    current+=list(Path("scripts").glob("*phase3d*"))+list(Path("tests").glob("*phase3d*"))
    current+=[Path("docs/phase3d_quantizer_seed_stability.md"),Path("configs/indexes/phase3d_quantizer_seed_stability.conf")]
    dump(a.output,{"status":"PASS","reproduced_files":checks,"input_sources_unchanged":True,
                   "model_score_source_binary_identity":True,"sha256":{str(x):digest(x) for x in sorted(current)}})
    print(f"verification PASS reproduced_files={len(checks)} hashed_files={len(current)}")


if __name__=="__main__":main()
