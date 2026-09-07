import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from phase3e_metrics import variance_components, classify, inversion_union, measure_factorial
from phase3d_metrics import single
from analyze_phase3e_training_sample_stability import pairwise, NAMES, OLD_NAMES


class FactorialTests(unittest.TestCase):
    def test_constant(self):
        v=variance_components(np.full((3,3),.1))
        self.assertAlmostEqual(v["total_model_variability"],0)
        self.assertAlmostEqual(v["signed_subset_variance_contrast"],0)
        self.assertEqual(v["total_model_variability"],0)
        self.assertEqual(v["within_subset_variance"],0)
        self.assertEqual(v["between_subset_variance"],0)

    def test_pure_subset(self):
        v=variance_components([[0,0,0],[1,1,1],[2,2,2]])
        self.assertEqual(v["within_subset_variance"],0)
        self.assertAlmostEqual(v["between_subset_variance"],2/3)
        self.assertEqual(v["signed_subset_variance_contrast"],1)

    def test_pure_init_negative_contrast_retained(self):
        v=variance_components([[0,1,2]]*3)
        self.assertEqual(v["between_subset_variance"],0)
        self.assertAlmostEqual(v["within_subset_variance"],2/3)
        self.assertAlmostEqual(v["signed_subset_variance_contrast"],-1/3)

    def test_exhaustive_variance_identity(self):
        rng=np.random.Generator(np.random.PCG64(22))
        for _ in range(30):
            x=rng.integers(-3,5,(3,3))/10;v=variance_components(x)
            mu=sum(float(t) for t in x.ravel())/9
            b=sum((sum(x[s])/3-mu)**2 for s in range(3))/3
            w=sum((x[s,i]-sum(x[s])/3)**2 for s in range(3) for i in range(3))/9
            self.assertAlmostEqual(v["total_model_variability"],b+w)
            self.assertAlmostEqual(v["between_subset_variance"],b)
            self.assertAlmostEqual(v["within_subset_variance"],w)

    def test_classification_and_majorities(self):
        for n in range(10):
            x=np.zeros(9);x[:n]=.1;r=classify(x.reshape(3,3))
            self.assertEqual(r["harmful_count_all"],n)
            self.assertEqual(r["category"],"Robust" if n<=2 else "Model-specific" if n<=5 else "Usually harmful" if n<=8 else "Universally harmful")
        self.assertTrue(classify([[1,1,0]]*3)["subset_persistent"])
        self.assertFalse(classify([[1,1,1],[1,1,1],[0,0,0]])["subset_persistent"])

    def test_signed_loss_not_harmful(self):
        r=classify([[-.1,0,.1],[-.2,0,0],[0,0,0]])
        self.assertEqual(r["harmful_count_all"],1)

    def test_pair_subset_counts(self):
        p=np.tile([0.,1.,2.,3.],(9,1))
        p[[0,1,3,6],1]=4
        pairs,counts=inversion_union(range(4),[0.,1.,2.,3.],p,2)
        self.assertEqual(len(pairs),2)
        for r in pairs:
            self.assertEqual(r["model_count"],4);self.assertEqual(r["subset_frequency"],3)
            self.assertEqual(r["subset_init_counts"],[2,1,1])
            self.assertEqual(r["subsets_with_init_majority"],1)

    def test_inversions_exhaustive_and_single_reuse(self):
        rng=np.random.Generator(np.random.PCG64(421))
        ids=np.array([5,3,7,2,6,4,1,0]);d=rng.integers(0,10,8).astype(float)
        p=rng.integers(0,10,(9,8)).astype(float);truth=[5,7,0]
        row,repl,pairs,random=measure_factorial(ids,d,p,truth,k=3,samples=16)
        A=np.argsort(d,kind="stable")[:3];expected={}
        for m in range(9):
            for i in A:
                for j in range(8):
                    if j not in A and d[i]<d[j] and p[m,i]>p[m,j]:expected.setdefault((int(ids[i]),int(ids[j])),[False]*9)[m]=True
            reference,_,_,r=single(ids,d,p[m],truth,3,1,16,1e-12)
            for key,v in reference.items():self.assertEqual(row["models"][m][key],v)
            np.testing.assert_array_equal(random,r)
        self.assertEqual({(r["inside_id"],r["outside_id"]):r["model_mask"] for r in pairs},expected)

    def test_exact_ties_and_alignment(self):
        ids=[0,1,2];d=[1.,1.,2.];p=np.tile(d,(9,1))
        row,_,pairs,_=measure_factorial(ids,d,p,[0],k=1,samples=8)
        self.assertEqual(pairs,[]);self.assertEqual(row["exact_topk_ids"],[0])
        with self.assertRaises(ValueError):measure_factorial([0,0,2],d,p,[0],k=1)
        with self.assertRaises(ValueError):variance_components(np.zeros((4,3)))
        with self.assertRaises(ValueError):inversion_union(ids,d,p[:8],1)

    def test_equal_model_frequency_different_subset_frequency(self):
        for models,expected in (([0,1,2],1),([0,3,6],3)):
            p=np.tile([0.,1.,2.],(9,1));p[models,1]=3
            pairs,_=inversion_union(range(3),[0.,1.,2.],p,2)
            self.assertEqual(len(pairs),1)
            self.assertEqual(pairs[0]["model_count"],3)
            self.assertEqual(pairs[0]["subset_frequency"],expected)

    def test_pairwise_balanced_and_secondary_counts(self):
        rng=np.random.Generator(np.random.PCG64(95));loss=rng.integers(0,4,(14,30))*.1
        inv=rng.integers(0,20,(14,30))
        primary=pairwise(NAMES,loss[:9],inv[:9])
        secondary=pairwise(NAMES+OLD_NAMES,loss,inv)
        self.assertEqual(sum(r["relation"]=="within" for r in primary),9)
        self.assertEqual(sum(r["relation"]=="between" for r in primary),27)
        self.assertEqual(sum(r["relation"]=="within" for r in secondary),19)
        self.assertEqual(len(secondary),91)

    def test_measurement_does_not_mutate_frozen_arrays(self):
        ids=np.array([3,2,1,0]);d=np.array([1.,2.,3.,4.]);p=np.tile(d,(9,1))
        saved=[v.copy() for v in (ids,d,p)]
        measure_factorial(ids,d,p,[3,2],k=2,samples=8)
        for before,after in zip(saved,(ids,d,p)):np.testing.assert_array_equal(before,after)


if __name__=="__main__":unittest.main()
