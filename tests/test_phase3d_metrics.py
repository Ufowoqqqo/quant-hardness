import itertools
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from phase3d_metrics import measure_seeds, single, category, pair_class, severity_overlap


class SeedMetricsTest(unittest.TestCase):
    def test_exhaustive_five_seed_reference(self):
        ids=np.array([8,3,9,1,7,6]);d=np.array([3.,1.,4.,2.,6.,5.])
        scores=np.array([[1,3,0,2,7,6],[3,0,1,2,7,6],[3,1,4,2,6,5],
                         [0,1,4,2,6,5],[4,2,1,3,6,5]],dtype=float)
        row,repl,pairs,random=measure_seeds(ids,d,scores,[3,1],k=2,samples=17)
        exact=sorted(range(len(ids)),key=lambda i:(d[i],i))[:2]
        expected={}
        for s,p in enumerate(scores):
            top=sorted(range(len(ids)),key=lambda i:(p[i],i))[:2]
            inverted=[(int(ids[i]),int(ids[j])) for i in exact for j in range(len(ids))
                      if j not in exact and d[i]<d[j] and p[i]>p[j]]
            m=row["models"][s]
            self.assertEqual(m["topk_ids"],ids[top].tolist())
            self.assertEqual(m["cross_boundary_inversion_count"],len(inverted))
            self.assertAlmostEqual(m["ranking_recall_loss"],1-len(set(ids[top])&{3,1})/2)
            self.assertAlmostEqual(sum(m["loss_attribution"].values()),m["ranking_recall_loss"])
            for pair in inverted:expected.setdefault(pair,[False]*5)[s]=True
        self.assertEqual({(p["inside_id"],p["outside_id"]):p["mask"] for p in pairs},expected)
        for p in pairs:self.assertEqual(p["frequency_count"],sum(p["mask"]))
        self.assertEqual(len(random),17)
        self.assertEqual(len(repl),sum(m["displaced_count"]*m["intruder_count"] for m in row["models"]))

    def test_same_models_persistent_pairs(self):
        p=np.tile([0.,3.,1.,4.],(5,1))
        row,_,pairs,_=measure_seeds([0,1,2,3],[0.,1.,2.,3.],p,[0,1],k=2)
        self.assertEqual(row["category"],"Persistently harmful")
        self.assertTrue(all(r["frequency_count"]==5 for r in pairs))
        self.assertEqual(row["std_ranking_loss"],0)

    def test_exact_control_and_order_ties(self):
        d=[1.,2.,2.,3.];ids=[10,11,12,13]
        row,_,pairs,_=measure_seeds(ids,d,np.tile(d,(5,1)),[10,11],k=2)
        self.assertEqual(row["category"],"Robust");self.assertEqual(pairs,[])
        self.assertEqual(row["exact_topk_ids"],[10,11])
        self.assertTrue(row["exact_boundary_tie"])

    def test_signed_lucky_tie_credit(self):
        row,_,pairs,_=measure_seeds([0,1,2],[1.,1.,2.],np.tile([2.,0.,3.],(5,1)),[1],k=1)
        self.assertEqual(pairs,[])
        for m in row["models"]:
            self.assertEqual(m["ranking_recall_loss"],-1)
            self.assertEqual(m["loss_attribution"]["gt_intruder_credit"],-1)

    def test_tie_debit_unassigned(self):
        row,_,pairs,_=measure_seeds([0,1,2],[1.,1.,2.],np.tile([2.,0.,3.],(5,1)),[0],k=1)
        self.assertEqual(pairs,[])
        self.assertEqual(row["models"][0]["loss_attribution"]["unassigned_tie_or_no_strict_overtake"],1)

    def test_uniqueness_and_score_alignment(self):
        with self.assertRaises(ValueError):measure_seeds([0,0,1],[0,1,2],np.zeros((5,3)),[0],k=1)
        with self.assertRaises(ValueError):measure_seeds([0,1,2],[0,1,2],np.zeros((4,3)),[0],k=1)

    def test_category_boundaries(self):
        self.assertEqual([category(i) for i in range(6)],
            ["Robust","Occasionally harmful","Occasionally harmful","Usually harmful","Usually harmful","Persistently harmful"])
        self.assertEqual([pair_class(i) for i in range(1,6)],["one-off","minority","majority","majority","persistent"])

    def test_severity_tie_sensitivity_reference(self):
        a=[1,1,0,0];b=[1,0,1,0]
        m=severity_overlap(a,b,.25)
        self.assertEqual(m["overlap_fraction"],1)
        self.assertEqual(m["independent_tie_resolution_expected_overlap"],.25)
        self.assertEqual(m["tie_inclusive_jaccard"],1/3)

    def test_random_samples_fixed_across_scores(self):
        a=single(range(4),[1.,2.,3.,4.],[4.,3.,2.,1.],[0],1,67,20,1e-12)
        b=single(range(4),[1.,2.,3.,4.],[1.,2.,3.,4.],[0],1,67,20,1e-12)
        np.testing.assert_array_equal(a[3],b[3])

    def test_random_bruteforce_and_signed_accounting(self):
        rng=np.random.Generator(np.random.PCG64(345))
        for _ in range(25):
            d=rng.integers(0,10,8).astype(float);p=rng.integers(0,10,(5,8)).astype(float)
            truth=rng.choice(8,3,replace=False).tolist()
            row,_,pairs,_=measure_seeds(range(8),d,p,truth,k=3,samples=20)
            A=set(sorted(range(8),key=lambda i:(d[i],i))[:3])
            for s in range(5):
                inv=sum(d[i]<d[j] and p[s,i]>p[s,j] for i in A for j in set(range(8))-A)
                self.assertEqual(row["models"][s]["cross_boundary_inversion_count"],inv)
                self.assertAlmostEqual(sum(row["models"][s]["loss_attribution"].values()),row["models"][s]["ranking_recall_loss"])
            self.assertEqual(sum(row["inversion_pair_frequency_counts"]),len(pairs))


if __name__=="__main__":unittest.main()
