import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from phase3g_metrics import permutations, evaluate, scalar_reference, empirical_rank, pair_diagnostics
from phase3d_metrics import single


class ResidualTests(unittest.TestCase):
    def test_scalar_and_prior_reference(self):
        rng = np.random.default_rng(21)
        for n in (21, 50, 100):
            ids = rng.permutation(n); d = rng.integers(0, 100, n).astype(float)
            scores = rng.integers(-10, 120, (7, n)).astype(float); truth = list(range(10))
            batch = evaluate(ids, d, scores, truth)
            for p, s in enumerate(scores):
                ref = scalar_reference(ids, d, s, truth)
                for key in ref: np.testing.assert_array_equal(batch[key][p], ref[key])
                prior, _, _, _ = single(ids, d, s, truth, 10, 3, 10, 1e-12)
                self.assertEqual(batch['loss_hits'][p], prior['loss_hits'])
                self.assertEqual(batch['inversions'][p], prior['cross_boundary_inversion_count'])
                self.assertEqual(batch['topk'][p].tolist(), prior['topk_ids'])

    def test_deterministic_independent_streams_and_multiset(self):
        p = permutations(30, 8, 7, 1, 2)
        np.testing.assert_array_equal(p, permutations(30, 8, 7, 1, 2))
        for other in (permutations(30, 8, 7, 2, 2), permutations(30, 8, 7, 1, 3)):
            self.assertFalse(np.array_equal(p, other))
        e = np.arange(30, dtype=float)**2
        for row in p: np.testing.assert_array_equal(np.sort(e[row]), np.sort(e))

    def test_reconstruction_centering_constant_and_signed_loss(self):
        d = np.arange(30, dtype=np.float32).astype(float); pq = (d*.8+1).astype(np.float32).astype(float)
        e = pq-d; np.testing.assert_array_equal(d+e, pq)
        idx = permutations(30, 8, 9, 0, 0); sh = e[idx]
        a = evaluate(np.arange(30), d, d+sh, list(range(10)))
        b = evaluate(np.arange(30), d, d+sh-e.mean(), list(range(10)))
        np.testing.assert_array_equal(a['topk'], b['topk'])
        c = evaluate(np.arange(30), d, d+100, list(range(10)))
        self.assertEqual(c['loss_hits'][0], 0)
        lucky = evaluate(np.arange(30), d, -d, list(range(20, 30)))
        self.assertEqual(lucky['loss_hits'][0], -10)

    def test_tie_rank(self):
        self.assertEqual(empirical_rank(1, [0, 1, 1, 2]),
                         {'below': 1, 'equal': 2, 'above': 1, 'lower': .25, 'upper': .75, 'midrank': .5})
        self.assertEqual(empirical_rank(0, [0, 0])['midrank'], .5)

    def test_strict_ties_and_order(self):
        ids = np.arange(30)[::-1]; d = np.ones(30); scores = np.zeros(30)
        v = evaluate(ids, d, scores, list(range(10)))
        np.testing.assert_array_equal(v['topk'][0], ids[:10]); self.assertEqual(v['inversions'][0], 0)

    def test_pair_identity_control(self):
        ids = np.arange(21); d = ids.astype(float); e = np.zeros(21); e[9] = 3
        sh = np.stack([e, e[::-1]]); obs = evaluate(ids, d, d+e, ids[:10]); null = evaluate(ids, d, d+sh, ids[:10])
        rows, selected = pair_diagnostics(ids, d, e, sh, obs['topk'][0], null['topk'])
        self.assertEqual(len(rows), 1); self.assertEqual(rows[0]['exact_margin'], 1.)
        self.assertEqual(rows[0]['observed_error_difference'], 3.)
        self.assertEqual(rows[0]['null_same_pair_strict_inversion_fraction'], .5)
        self.assertEqual(selected.shape, (2, 3))

    def test_invalid_input(self):
        with self.assertRaises(ValueError): evaluate([1]*21, np.arange(21), np.arange(21), [1])
        with self.assertRaises(ValueError): evaluate(np.arange(21), np.arange(21), [np.nan]*21, [1])


if __name__ == '__main__': unittest.main()
