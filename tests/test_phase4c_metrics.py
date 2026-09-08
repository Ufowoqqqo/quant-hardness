import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from phase4c_metrics import latency_summary,gate,selective_gate

class Metrics(unittest.TestCase):
    def test_latency(self):
        s=latency_summary([1,2,3]);self.assertEqual(s['mean_us'],2);self.assertEqual(s['service_qps'],500000)
        with self.assertRaises(ValueError):latency_summary([0])
    def test_overhead_and_equality(self):
        s=gate(100,120,110,130,125)
        self.assertTrue(s['overhead_gate_pass']);self.assertTrue(s['meaningful_optimization'])
        self.assertEqual(s['overhead_reduction'],.5)
    def test_speedup_only(self):
        s=gate(100,120,118,150,120);self.assertFalse(s['overhead_gate_pass']);self.assertTrue(s['all16_gate_pass'])
        self.assertEqual(s['all16_speedup'],1.25)
    def test_negative_not_clipped(self):
        s=gate(100,120,125,130,140);self.assertLess(s['overhead_reduction'],0);self.assertFalse(s['meaningful_optimization'])
        self.assertIsNone(gate(100,100,110,130,125)['overhead_reduction'])
    def test_selectivity_requires_both(self):
        s=selective_gate(80,100,.8,.83,.9);self.assertFalse(s['selectivity_gate_pass'])
        self.assertTrue(selective_gate(80,100,.8,.88,.9)['selectivity_gate_pass'])

if __name__=='__main__':unittest.main()
