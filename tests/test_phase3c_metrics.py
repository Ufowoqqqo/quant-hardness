import json
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from phase3c_metrics import measure,transition_flags
from analyze_phase3c_precision_transition import overlap_summary


class PrecisionTests(unittest.TestCase):
    def test_top_query_overlap_with_discrete_ties(self):
        rows=[{"query_id":i,"loss_hits_32":1,"loss_hits_64":1} for i in range(10)]
        persistence,overlaps=overlap_summary(rows)
        self.assertEqual(persistence["harmful_jaccard"],1)
        self.assertEqual(overlaps[0]["exact_count_per_set"],1)
        self.assertEqual(overlaps[0]["cutoff_ties_32"],10)
        self.assertEqual(overlaps[0]["tie_inclusive_count_32"],10)
        self.assertEqual(overlaps[0]["overlap_fraction"],1)

    def case(self,d,p32,p64,truth=(0,1)):
        return measure(np.arange(len(d)),d,p32,p64,truth,2,11,128,1e-12)

    def test_exact_scoring_is_control(self):
        row,repl,pairs,_=self.case([1,2,3,4],[1,2,3,4],[1,2,3,4])
        self.assertEqual(row["ranking_recall_loss_32"],0)
        self.assertEqual(row["cross_boundary_inversion_count_32"],0)
        self.assertTrue(row["stable_both"])
        self.assertEqual(repl,[]);self.assertEqual(pairs,[])

    def test_rescued_persistent_overlap_and_regression(self):
        f=transition_flags(4,2)
        self.assertTrue(f["rescued_improved"] and f["persistent_harmful"])
        self.assertFalse(f["fully_rescued"])
        f=transition_flags(1,2)
        self.assertTrue(f["regressed"] and f["persistent_harmful"])
        self.assertTrue(transition_flags(1,0)["fully_rescued"])
        self.assertTrue(transition_flags(0,-1)["rescued_improved"])

    def test_pair_transitions_against_nested_loop(self):
        d=np.array([1,2,3,4,5.]);a=np.array([1,6,2,3,4.]);b=np.array([4,2,1,5,6.])
        row,repl,pairs,_=self.case(d,a,b)
        expected={}
        for i in (0,1):
            for j in (2,3,4):
                ia=d[i]<d[j] and a[i]>a[j];ib=d[i]<d[j] and b[i]>b[j]
                if ia or ib:expected[(i,j)]="persistent" if ia and ib else "corrected" if ia else "new"
        self.assertEqual({(p["inside_id"],p["outside_id"]):p["state"] for p in pairs},expected)
        self.assertEqual(row["delta_critical_inversions"],row["corrected_inversion_count"]-row["new_inversion_count"])
        for p in pairs:
            i,j=p["inside_id"],p["outside_id"]
            self.assertEqual(p["violation_32"],a[i]-a[j])
            self.assertEqual(p["violation_64"],b[i]-b[j])
        json.dumps([row,repl,pairs],allow_nan=False)

    def test_tied_exact_replacement_retained_not_strict_inversion(self):
        row,repl,pairs,_=self.case([1,2,2],[1,3,2],[1,2,3])
        self.assertTrue(row["exact_boundary_tie"])
        self.assertEqual(row["cross_boundary_inversion_count_32"],0)
        self.assertEqual(len(repl),1)
        self.assertEqual(repl[0]["exact_margin"],0)
        self.assertFalse(repl[0]["strict_inversion"])

    def test_same_pair_sample_and_monotonicity_not_assumed(self):
        args=([1,2,3,4],[1,2,3,4],[4,3,2,1])
        row,_,_,sample=self.case(*args)
        self.assertTrue(row["regressed"])
        self.assertLess(row["delta_loss_precision"],0)
        self.assertTrue(np.array_equal(sample,self.case(*args)[3]))
        self.assertTrue(np.all(sample[:,0]!=sample[:,1]))

    def test_selected_intruders_distinct_from_all_overtakers(self):
        row,_,_,_=self.case([1,2,3,4,5],[1,9,2,3,4],[1,2,3,4,5])
        self.assertEqual(row["intruder_count_32"],1)
        self.assertEqual(row["max_selected_intruders_overtaking_one_topk_32"],1)
        self.assertEqual(row["max_outside_overtaking_one_topk_32"],3)


if __name__=="__main__":unittest.main()
