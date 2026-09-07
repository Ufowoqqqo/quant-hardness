import sys
import json
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from phase3b_metrics import measure
from audit_phase3b_native_ties import heap_reference
from analyze_phase3b_fixed_candidate_ranking import binned


class FixedRankingTests(unittest.TestCase):
    def test_quantile_bins_preserve_ties_and_missing_ratio(self):
        rows=[{"x":x,"ranking_recall_loss":.1,"ranking_disagreement":.2,"class":"Harmful"}
              for x in [0,0,0,0,1,1,2,None]]
        bins=binned(rows,"x",4)
        self.assertEqual(sum(r["count"] for r in bins),len(rows))
        self.assertEqual(bins[-1]["bin"],"zero_margin")
        self.assertEqual(bins[-1]["count"],1)
        finite=bins[:-1]
        for first,second in zip(finite,finite[1:]):
            self.assertLess(first["max_x"],second["min_x"])

    def test_heap_ties_use_id_for_eviction_but_score_for_admission(self):
        # Both 8 and 3 precede 1 at score 4. A later strictly better item
        # evicts the larger ID, while a later boundary-tied ID is not admitted.
        self.assertEqual(heap_reference([8,3,1,9], [4,4,4,2], 2), [9,3])

    def run_case(self, d, p, gt, native=None):
        ids = np.arange(len(d))
        if native is None:
            native = np.argsort(p, kind="stable")[:2]
        return measure(ids, d, p, native, gt, 2, 7, 512)

    def test_classes(self):
        self.assertEqual(self.run_case([1,2,3], [1,2,3], [0,1])[0]["class"], "Stable")
        self.assertEqual(self.run_case([1,2,3], [1,4,3], [0,1])[0]["class"], "Harmful")
        self.assertEqual(self.run_case([1,2,3], [1,4,3], [0,9])[0]["class"], "Benign")
        self.assertEqual(self.run_case([1,2,2], [1,4,2], [0,2])[0]["class"], "Lucky")

    def test_pair_identity_and_reference(self):
        d, p = np.array([1,2,3,4,5.]), np.array([1,7,2,3,8.])
        row, pairs, samples = self.run_case(d, p, [0,1])
        expected = sum(d[i]<d[j] and p[i]>p[j] for i in (0,1) for j in (2,3,4))
        self.assertEqual(row["cross_boundary_inversion_count"], expected)
        self.assertEqual(row["topk_items_overtaken"], 1)
        self.assertEqual(row["max_outside_overtaking_one_topk"], 2)
        self.assertEqual(row["boundary_margin"], 1)
        self.assertTrue(row["boundary_overtaken"])
        for pair in pairs:
            json.dumps(pair, allow_nan=False)
            i,j=pair["displaced_id"],pair["intruder_id"]
            self.assertEqual(pair["violation"], p[i]-p[j])
        self.assertTrue(np.all(samples[:,0]!=samples[:,1]))
        self.assertTrue(np.array_equal(samples,self.run_case(d,p,[0,1])[2]))

    def test_ties_are_not_forced_equal(self):
        row, _, _ = self.run_case([1,2,3], [1,2,2], [0,1], [0,2])
        self.assertFalse(row["native_set_equal"])
        self.assertFalse(row["native_list_equal"])
        self.assertTrue(row["native_order_equal_modulo_pq_ties"])
        bad,_,_=self.run_case([1,2,3], [1,2,3], [0,1], [0,2])
        self.assertFalse(bad["native_order_equal_modulo_pq_ties"])
        tied,_,_=self.run_case([1,2,2], [1,4,2], [0,1])
        self.assertIsNone(tied["error_scale_over_boundary_margin"])
        self.assertEqual(tied["cross_boundary_inversion_count"],0)

    def test_boundary_window_and_random_reference(self):
        ids=np.arange(25)
        d=ids.astype(float)+1
        p=d+np.where(ids%2,3.,-1.)
        native=np.argsort(p,kind="stable")[:10]
        row,_,pairs=measure(ids,d,p,native,ids[:10],10,123,512)
        self.assertEqual(row["boundary_window_ranks"], list(range(5,16)))
        self.assertEqual(len(row["boundary_window_absolute_errors"]),11)
        self.assertEqual(row["boundary_gap_r5"],9)
        expected=sum((d[i]<d[j] and p[i]>p[j]) or (d[i]>d[j] and p[i]<p[j]) for i,j in pairs)
        self.assertEqual(expected,row["random_inversion_count"])


if __name__ == "__main__":
    unittest.main()
