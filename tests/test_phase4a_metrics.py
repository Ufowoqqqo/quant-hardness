import itertools
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from phase4a_metrics import *


class Phase4ATests(unittest.TestCase):
    def test_tie_aware_boundary_capacity(self):
        from phase4a_tie_sensitivity import tie_hits
        self.assertEqual(tie_hits([10]*10,list(range(1,12))),1)
        self.assertEqual(tie_hits(list(range(1,11)),list(range(1,12))),10)
        self.assertEqual(tie_hits([1]*10,[1]*11),10)

    def test_gap_is_frozen_raw_and_short_rule(self):
        self.assertEqual(score_gap(np.arange(20.))[0],3)
        self.assertTrue(np.isinf(score_gap(np.arange(11.))[0]))
        self.assertEqual(score_gap(np.arange(20.)+100)[0],3)

    def test_rerank_reference_and_cost(self):
        rng=np.random.default_rng(4)
        for L in [16,32,64]:
            ids=np.arange(90);p=rng.integers(0,40,90);d=rng.integers(0,60,90)
            got,selected=refinement(ids,p,d,L)
            chosen=sorted(range(90),key=lambda i:(p[i],i))[:L]
            ref=sorted(chosen,key=lambda i:(d[i],i))[:10]
            np.testing.assert_array_equal(got,ref);self.assertEqual(len(selected),L)
        with self.assertRaises(ValueError):refinement(ids[:11],p[:11],d[:11],16)

    def test_gap_selection_no_labels_and_ties(self):
        np.testing.assert_array_equal(selector([3,1,1,4],2),[1,2])

    def test_oracle_is_budget_upper_bound(self):
        native=np.array([5,5,5,5]);ref=np.array([7,4,9,6])
        upper=policy_value(native,ref,oracle_selector(ref-native,2),16)
        for ix in itertools.combinations(range(4),2):
            p=policy_value(native,ref,ix,16)
            self.assertLessEqual(p['recall'],upper['recall'])
            self.assertEqual(p['exact_distance_evals_per_query'],8)

    def test_signed_recall_and_ratios(self):
        self.assertEqual(hit_count(range(10),range(5,15)),5)
        self.assertLess(policy_value([8,8],[7,8],[0],16)['recall_gain'],0)
        self.assertIsNone(ratio(1,0));self.assertEqual(ratio(-1,2),-.5)

    def test_random_exact_count_reproducible(self):
        x=random_order(100,1,0,2);np.testing.assert_array_equal(x,random_order(100,1,0,2))
        self.assertEqual(len(set(x[:25])),25)
        self.assertFalse(np.array_equal(x,random_order(100,1,1,2)))


if __name__=='__main__':unittest.main()
