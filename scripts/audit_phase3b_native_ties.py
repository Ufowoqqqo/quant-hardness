#!/usr/bin/env python3
"""Independent reference for FAISS result-heap semantics, without traversal."""
import argparse
import heapq
import json
from pathlib import Path
import numpy as np
from analyze_phase3b_fixed_candidate_ranking import DTYPE, dump, digest


def heap_reference(ids, scores, k):
    heap = []
    for identifier, score in zip(ids, scores):
        identifier, score = int(identifier), float(score)
        item = (-score, -identifier, identifier)
        if len(heap) < k:
            heapq.heappush(heap, item)
        elif score < -heap[0][0]:
            heapq.heapreplace(heap, item)
    return [item[2] for item in sorted(heap, key=lambda x: (-x[0], x[2]))]


def main(raw, derived):
    maps = {p.name:np.memmap(p,dtype=DTYPE,mode="r") for p in raw.glob("candidate_scores_*.bin")}
    examples = []
    checked, differences, native_loss_diff = 0,0,0
    for line in (raw/"queries.jsonl").open():
        q=json.loads(line)
        start=q["byte_offset"]//12
        a=maps[q["candidate_file"]][start:start+q["candidate_count"]]
        reference=heap_reference(a["id"],a["pq"],10)
        if reference != q["native_ids"]:
            raise AssertionError(f"heap reference differs at query {q['query_id']}")
        po=np.argsort(a["pq"],kind="stable")[:10]
        global_ids=a["id"][po].tolist()
        if set(global_ids)!=set(reference):
            differences+=1
            truth=set(q["ground_truth_ids"])
            hit_difference=len(set(reference)&truth)-len(set(global_ids)&truth)
            native_loss_diff += hit_difference
            lookup={int(v):i for i,v in enumerate(a["id"])}
            changed=sorted(set(global_ids)^set(reference))
            examples.append({"query_id":q["query_id"],"native_ids":reference,
                             "pq_global_stable_ids":global_ids,
                             "exchanged_ids":changed,
                             "exchanged_pq_scores":[float(a["pq"][lookup[v]]) for v in changed],
                             "exchanged_exact_distances":[float(a["exact"][lookup[v]]) for v in changed],
                             "native_minus_global_gt_hits":hit_difference})
        checked+=1
    dump(derived/"native_tie_cases.json",examples)
    dump(derived/"native_heap_audit.json",{
        "queries":checked,"heap_reference_list_mismatches":0,
        "native_global_set_mismatches":differences,
        "mean_native_minus_global_recall_due_to_ties":native_loss_diff/(10*checked),
        "audit_source_sha256":digest(__file__),
        "semantics":"strict distance-only admission; worst (distance,ID) evicted; output ordered by (distance,ID). No traversal replay or native substitution."})
    print(f"reference_heap_checked={checked} mismatches=0 tied_set_differences={differences}")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("raw",type=Path);p.add_argument("derived",type=Path)
    a=p.parse_args();main(a.raw,a.derived)
