import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from phase4d_metrics import latency,summarize,scaling,relative

class Metrics(unittest.TestCase):
    def test_actual_latency(self):
        a=latency([100,200,300]);self.assertEqual(a['mean_latency_us'],200);self.assertEqual(a['median_latency_us'],200)
        with self.assertRaises(ValueError):latency([0])
    def test_repetition_statistics(self):
        a=summarize([1,2,3]);self.assertEqual(a,{'mean':2.,'std':1.,'min':1.,'max':3.})
    def test_scaling(self):
        self.assertEqual(scaling(3000,1000,4),{'speedup':3.,'parallel_efficiency':.75})
        with self.assertRaises(ValueError):scaling(1,0,2)
    def test_signed_overhead(self):
        n={'qps':1000,'mean_latency_us':100,'p95_latency_us':150,'p99_latency_us':200}
        a={'qps':1100,'mean_latency_us':95,'p95_latency_us':160,'p99_latency_us':250}
        r=relative(n,a);self.assertAlmostEqual(r['throughput_penalty'],-.1);self.assertAlmostEqual(r['mean_latency_overhead'],-.05);self.assertEqual(r['p99_ratio'],1.25)
    def test_deterministic_worker_shards(self):
        for n in [1,7,14252]:
            for t in [1,2,4,8]:
                stream=[(p,q) for w in range(t) for p in range(4) for q in range(w,n,t)]
                self.assertEqual(sorted(stream),[(p,q) for p in range(4) for q in range(n)])

if __name__=='__main__':unittest.main()
