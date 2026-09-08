import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from phase4b_metrics import *


class Phase4BTests(unittest.TestCase):
    def test_fixed_numeric_inclusive_threshold(self):
        g=np.array([710.40624,710.40625,710.40626])
        np.testing.assert_array_equal(g<=710.40625,[True,True,False])

    def test_random_matches_actual_not_nominal(self):
        m=mask_replicates(101,31,3,30)
        self.assertTrue(np.all(m.sum(1)==31));np.testing.assert_array_equal(m,mask_replicates(101,31,3,30))

    def test_latency_units_and_quantiles(self):
        x=latency_stats([10,20,30]);self.assertEqual(x['mean_us'],20);self.assertEqual(x['qps'],50000)
        self.assertEqual(x['p95_us'],29)
        with self.assertRaises(ValueError):latency_stats([0,1])

    def test_negative_system_savings_not_clipped(self):
        r=systems_metrics(.8,.85,.9,110,100,20,100)
        self.assertAlmostEqual(r['latency_saving'],-.1);self.assertAlmostEqual(r['all16_gain_retained'],.5)
        self.assertAlmostEqual(r['exact_eval_saving'],.8)

    def test_id_recall_and_validation(self):
        self.assertEqual(recall(range(10),range(5,15)),.5)
        with self.assertRaises(ValueError):recall([0]*10,range(10))


if __name__=='__main__':unittest.main()
