import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from phase3h_metrics import bins,conditioned_permutations,verify_assignment,closure,pq_observables,quintiles,CONDITIONS
from phase3g_metrics import evaluate,scalar_reference,permutations


class RankNullTests(unittest.TestCase):
    def test_exact_registered_edges(self):
        order=np.arange(55)[::-1]
        self.assertEqual([len(b) for b in bins(order,'N4')],[5,3,2,2,3,5,20,15])
        self.assertEqual([len(b) for b in bins(order,'N3')],[5,5,5,5,20,15])
        for n in (1,3,6,10,11,17,20,35,40,41):
            for c in CONDITIONS:
                bs=bins(order[:n],c)
                expected=np.arange(n) if c=='N0' else order[:n]
                np.testing.assert_array_equal(np.sort(np.concatenate(bs)),np.sort(expected))
                self.assertTrue(all(len(b)>0 for b in bs))

    def test_distance_quantiles_are_rank_quartiles(self):
        d=np.array([3.,1.,1.,9.,7.,2.,5.,4.,6.]);o=np.argsort(d,kind='stable')
        b=bins(o,'N_distance');self.assertEqual([len(x) for x in b],[3,2,2,2])
        np.testing.assert_array_equal(np.concatenate(b),o)

    def test_preserves_bin_multisets_and_reference(self):
        rng=np.random.default_rng(8);ids=rng.permutation(55);d=rng.integers(1,100,55).astype(float);e=rng.normal(size=55)*10
        order=np.argsort(d,kind='stable');gt=list(range(10))
        for c in CONDITIONS:
            idx=conditioned_permutations(order,5,9,1,7,c);verify_assignment(idx,order,c)
            np.testing.assert_array_equal(idx,conditioned_permutations(order,5,9,1,7,c))
            v=evaluate(ids,d,d+e[idx],gt)
            for p in range(5):
                r=scalar_reference(ids,d,d+e[idx[p]],gt)
                for key in r:np.testing.assert_array_equal(r[key],v[key][p])
                for b in bins(order,c):np.testing.assert_array_equal(np.sort(e[idx[p,b]]),np.sort(e[b]))

    def test_n0_matches_3g_prefix(self):
        order=np.arange(31)[::-1]
        np.testing.assert_array_equal(conditioned_permutations(order,100,9,2,8,'N0')[:50],permutations(31,50,90700011,2,8))

    def test_rejects_cross_bin_exchange(self):
        o=np.arange(30);idx=np.array([o.copy()]);idx[0,[0,15]]=idx[0,[15,0]]
        with self.assertRaises(AssertionError):verify_assignment(idx,o,'N1')

    def test_closure_signed_and_guarded(self):
        self.assertAlmostEqual(closure(.2,.3,.1),-1)
        self.assertAlmostEqual(closure(.2,.05,.1),1.5)
        self.assertIsNone(closure(.1,.1,.1));self.assertIsNone(closure(.1,.2,.2))

    def test_pq_observables_only_scores_and_scale(self):
        p=np.arange(30,dtype=float)+1;o=pq_observables(p)
        self.assertEqual(o['g10_11_pq'],1);self.assertEqual(o['g9_12_pq'],3)
        self.assertEqual(o['count_1pct'],1)
        p[9:12]=10.;self.assertEqual(pq_observables(p)['count_1pct'],3)
        a=pq_observables(np.zeros(30));self.assertEqual(a['count_5pct'],30);self.assertEqual(a['relative_g9_12_pq'],0)
        for key in ('relative_g10_11_pq','relative_g9_12_pq','count_1pct','count_2pct','count_5pct'):
            self.assertEqual(pq_observables(p)[key],pq_observables(p*2)[key])

    def test_tie_preserving_quintiles(self):
        x=np.array([0]*10+[1]*10+[2]*10);labels=quintiles(x)
        for v in np.unique(x):self.assertEqual(len(set(labels[x==v])),1)
        self.assertLessEqual(len(set(labels)),5)


if __name__=='__main__':unittest.main()
