#!/usr/bin/env python3
"""Validate Phase 3C raw hashes and independent, database-free re-derivation."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from analyze_phase3b_fixed_candidate_ranking import digest, dump


def verify(run,reproduced,tables,figures,output):
    if output.exists():raise FileExistsError("refusing to overwrite verification artifact")
    hashes={};compared=[]
    for condition in ("exact_pool","pq64_pool_control"):
        manifest=json.loads((run/condition/"manifest.json").read_text())
        assert manifest["source_sha256"]==digest("src/experiments/faiss_sift1m.cpp")
        assert manifest["binary_sha256"]==digest("build/faiss_sift1m")
        validation=json.loads((run/"analysis"/condition/"validation.json").read_text())
        assert validation["shared_candidate_hashes_identical"]
        for path,expected in validation["input_sha256"].items():
            assert digest(path)==expected,path
            hashes[path]=expected
        for filename in ("queries.jsonl","replacement_pairs.jsonl.gz",
                         "inversion_transitions.jsonl.gz","random_pair_indices.bin"):
            first=run/"analysis"/condition/filename
            second=reproduced/"analysis"/condition/filename
            assert digest(first)==digest(second),str(first)
            compared.append(str(first))
    provenance=json.loads((run/"control_input_provenance.json").read_text())
    assert digest(run/"control_candidates.bin")==provenance["control_candidates_sha256"]
    for path,expected in provenance["input_sha256"].items():assert digest(path)==expected,path
    for folder,kind in ((tables,"tables"),(figures,"figures")):
        for first in sorted(folder.glob("phase3c_*")):
            assert digest(first)==digest(reproduced/kind/first.name),str(first)
            if first.suffix==".svg":ET.parse(first)
            if first.suffix==".json":json.loads(first.read_text())
            compared.append(str(first));hashes[str(first)]=digest(first)
    for pattern in ("scripts/*phase3c*.py","tests/test_phase3c_metrics.py",
                    "src/experiments/faiss_sift1m.cpp","docs/phase3c_precision_transition.md",
                    "docs/experiment_log.md","configs/indexes/phase3c_precision_transition.conf"):
        for path in Path(".").glob(pattern):hashes[str(path)]=digest(path)
    for path in sorted(run.rglob("*")):
        if path.is_file():hashes[str(path)]=digest(path)
    dump(output,{"status":"PASS","raw_inputs_unchanged":True,
                 "independent_regeneration_byte_identical":compared,
                 "final_artifact_sha256":hashes,
                 "package_versions":{name:importlib.metadata.version(name) for name in
                     ("numpy","matplotlib","contourpy","cycler","fonttools","kiwisolver",
                      "pillow","pyparsing","python-dateutil","packaging","importlib-resources","zipp")}})
    print(f"status=PASS reproduced_files={len(compared)} hashed_files={len(hashes)}")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("run","reproduced","tables","figures","output"):p.add_argument(name,type=Path)
    a=p.parse_args();verify(a.run,a.reproduced,a.tables,a.figures,a.output)
