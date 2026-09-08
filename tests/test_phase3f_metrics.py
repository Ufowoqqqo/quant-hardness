import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from phase3f_rotation_validation import orthogonal, rotate, exhaustive, rank_gate, close_distances
from phase3f_metrics import factorial, measure_rotations
from phase3d_metrics import single
from analyze_phase3f_rotation_stability import pairwise


class RotationValidationTests(unittest.TestCase):
    def test_orthogonal_and_deterministic(self):
        r=orthogonal(8,19)
        np.testing.assert_array_equal(r,orthogonal(8,19))
        np.testing.assert_allclose(r.T@r,np.eye(8),atol=1e-12)
        self.assertFalse(np.array_equal(r,orthogonal(8,20)))

    def test_row_convention_and_identity(self):
        x=np.arange(24,dtype=np.float32).reshape(3,8);r=orthogonal(8,21)
        np.testing.assert_array_equal(rotate(x,np.eye(8)),x)
        np.testing.assert_array_equal(rotate(x,r)[0],(r@x[0].astype(float)).astype(np.float32))
        close_distances(exhaustive(x,x),exhaustive(rotate(x,r),rotate(x,r)),.005,5e-7)

    def test_exhaustive_reference(self):
        x=np.arange(24).reshape(6,4);q=np.arange(12).reshape(3,4)
        np.testing.assert_array_equal(exhaustive(x,q),((q[:,None,:]-x[None,:,:])**2).sum(2))

    def test_rank_gate_ties(self):
        r,old,new=rank_gate([1.,2.,2.,4.],[1.,2.00001,1.99999,4.],2,.005,5e-7)
        self.assertTrue(r['raw_topk_set_changed']);self.assertTrue(r['reference_boundary_tie'])
        self.assertEqual(r['strict_reference_reversals'],0)

    def test_reject_broken_invariance(self):
        with self.assertRaises(AssertionError):rank_gate([1.,2.,3.],[1.,3.,2.],2,.005,5e-7)
        with self.assertRaises(AssertionError):close_distances([1.],[float('nan')],.005,5e-7)


class FactorialTests(unittest.TestCase):
    def test_constant(self):
        v=factorial(np.full((5,3),.1))
        self.assertEqual(v['total_model_variance'],0)
        self.assertEqual(v['harmful_rotation_frequency'],1)
        self.assertEqual(v['category'],'rotation-persistent')

    def test_only_rotation(self):
        x=np.repeat(np.arange(5)[:,None]/10,3,axis=1);v=factorial(x)
        self.assertEqual(v['within_rotation_initialization_variance'],0)
        self.assertAlmostEqual(v['between_rotation_variance'],np.var(x))

    def test_only_initialization(self):
        x=np.tile([0.,.1,.2],(5,1));v=factorial(x)
        self.assertEqual(v['between_rotation_variance'],0)
        self.assertAlmostEqual(v['two_way_interaction_variance'],0)
        self.assertLess(v['signed_rotation_variance_contrast'],0)

    def test_decomposition_reference(self):
        x=np.random.default_rng(9).normal(size=(5,3));v=factorial(x)
        self.assertAlmostEqual(v['total_model_variance'],np.mean((x-x.mean())**2))
        self.assertAlmostEqual(v['between_rotation_variance'],np.mean((x.mean(1)-x.mean())**2))
        self.assertAlmostEqual(v['within_rotation_initialization_variance'],np.mean((x-x.mean(1)[:,None])**2))
        self.assertAlmostEqual(v['total_model_variance'],sum(v[k] for k in ('two_way_init_main_variance','two_way_rotation_main_variance','two_way_interaction_variance')))

    def test_frequency_bins_signed(self):
        for count,category in ((0,'rotation-robust'),(2,'occasionally vulnerable'),(4,'usually vulnerable'),(5,'rotation-persistent')):
            x=np.full((5,3),-.1);x[:count,:2]=.1
            v=factorial(x);self.assertEqual(v['category'],category);self.assertEqual(v['harmful_rotation_count'],count)

    def test_inversion_union_scalar_reference(self):
        rng=np.random.default_rng(19);ids=np.arange(24);d=np.arange(24,dtype=float)+1;p=d+rng.normal(size=(15,24))*5
        row,replacements,pairs,random=measure_rotations(ids,d,p,ids[:10].tolist())
        expected={}
        for i in range(10):
            for j in range(10,24):
                mask=[bool(p[m,i]>p[m,j]) for m in range(15)]
                if any(mask):expected[i,j]=mask
        self.assertEqual(len(pairs),len(expected))
        for pair in pairs:
            self.assertEqual(pair['model_mask'],expected[pair['inside_id'],pair['outside_id']])
            self.assertEqual(pair['rotation_frequency'],sum(any(pair['model_mask'][r*3:r*3+3]) for r in range(5)))
        for m in range(15):
            metric,_,_,_=single(ids,d,p[m],ids[:10].tolist(),10,1,512,1e-12)
            for k,v in metric.items():self.assertEqual(v,row['models'][m][k])

    def test_perfect_scores_and_uniqueness(self):
        ids=np.arange(24);d=np.arange(24,dtype=float)+1;p=np.tile(d,(15,1))
        row,_,pairs,_=measure_rotations(ids,d,p,ids[:10].tolist())
        self.assertEqual(pairs,[]);self.assertEqual(row['category'],'rotation-robust')
        self.assertEqual(row['oracle_recall'],1)
        with self.assertRaises(ValueError):measure_rotations(np.zeros(24),d,p,ids[:10].tolist())
        with self.assertRaises(ValueError):factorial(np.zeros((3,3)))

    def test_pairwise_design_counts(self):
        rows=[{'models':[{'ranking_recall_loss':(m+q)%3/10,'cross_boundary_inversion_count':(m+2*q)%7} for m in range(15)]} for q in range(7)]
        pairs=pairwise(rows)
        self.assertEqual(len(pairs),105)
        self.assertEqual(sum(p['relation']=='within' for p in pairs),15)
        self.assertEqual(sum(p['relation']=='between' for p in pairs),90)
        self.assertEqual(sum(p['basis_relation']=='identity_to_random' for p in pairs),36)
        self.assertEqual(sum(p['basis_relation']=='random_to_random' for p in pairs),54)
        self.assertEqual(sum(p['relation']=='between' and p['same_init'] for p in pairs),30)

    def test_measurement_preserves_input_arrays(self):
        ids=np.arange(24);d=np.arange(24,dtype=float)+1;p=np.tile(d,(15,1));before=[v.copy() for v in (ids,d,p)]
        measure_rotations(ids,d,p,ids[:10].tolist())
        for old,current in zip(before,(ids,d,p)):np.testing.assert_array_equal(old,current)


if __name__=='__main__':unittest.main()
