#!/usr/bin/env python3
"""Verify Phase 3B raw inputs and independent regeneration; record hashes."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from analyze_phase3b_fixed_candidate_ranking import digest, dump


def verify(run, reproduction, tables, figures, output):
    if output.exists():
        raise FileExistsError("refusing to overwrite verification artifact")
    compared=[]
    hashes={}
    for ef in (64,128):
        raw=run/f"ef{ef}"
        manifest=json.loads((raw/"manifest.json").read_text())
        assert digest("src/experiments/faiss_sift1m.cpp")==manifest["export_source_sha256"]
        assert digest("build/faiss_sift1m")==manifest["binary_sha256"]
        derived=run/"analysis"/f"ef{ef}"
        validation=json.loads((derived/"validation.json").read_text())
        assert not validation["non_tied_native_mismatch_query_ids"]
        for name,expected in validation["input_sha256"].items():
            actual=digest(name)
            assert actual==expected,name
            hashes[name]=actual
        for name in ("queries.jsonl","harmful_pairs.jsonl","random_pair_indices.bin"):
            original=derived/name
            copy=reproduction/"analysis"/f"ef{ef}"/name
            assert digest(original)==digest(copy),str(original)
            compared.append(str(original))
        audit=json.loads((derived/"native_heap_audit.json").read_text())
        assert audit["queries"]==10000 and audit["heap_reference_list_mismatches"]==0
    for directory in (tables,figures):
        kind="tables" if directory==tables else "figures"
        for path in sorted(directory.glob("phase3b_*")):
            assert digest(path)==digest(reproduction/kind/path.name),str(path)
            if path.suffix==".json": json.loads(path.read_text())
            if path.suffix==".svg": ET.parse(path)
            compared.append(str(path))
            hashes[str(path)]=digest(path)
    for pattern in ("scripts/*phase3b*.py", "tests/test_phase3b_metrics.py",
                    "src/experiments/faiss_sift1m.cpp", "docs/phase3b_fixed_candidate_ranking.md",
                    "docs/experiment_log.md"):
        for path in Path(".").glob(pattern): hashes[str(path)]=digest(path)
    for path in sorted(run.rglob("*")):
        if path.is_file(): hashes[str(path)]=digest(path)
    versions={name:importlib.metadata.version(name) for name in
              ("numpy","matplotlib","contourpy","cycler","fonttools","kiwisolver",
               "pillow","pyparsing","python-dateutil","packaging","importlib-resources","zipp")}
    dump(output,{"status":"PASS", "raw_inputs_unchanged":True,
                 "independent_regeneration_byte_identical":compared,
                 "python_package_versions":versions,
                 "final_artifact_sha256":hashes})
    print(f"status=PASS reproduced_files={len(compared)} hashed_files={len(hashes)}")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("run","reproduction","tables","figures","output"):p.add_argument(key,type=Path)
    a=p.parse_args();verify(a.run,a.reproduction,a.tables,a.figures,a.output)
