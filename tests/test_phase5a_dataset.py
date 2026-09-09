import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from phase5a_dataset import split_ids, normalize_fp32_once


class FrozenDatasetTest(unittest.TestCase):
    def test_split(self):
        a = split_ids(1000, 10, 65, 20260909, 95000011)
        b = split_ids(1000, 10, 65, 20260909, 95000011)
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)
        base, query, train = a
        self.assertEqual(len(np.unique(np.r_[base, query])), 1000)
        self.assertTrue(np.isin(train, base).all())
        self.assertFalse(np.isin(train, query).any())
        self.assertEqual(len(np.unique(train)), 65)

    def test_normalization_reference(self):
        source = np.array([[1.00000001, 2, -3], [7, 8, 0]], dtype=np.float64)
        actual = normalize_fp32_once(source)
        expected = []
        for row in source.astype(np.float32):
            norm = sum(float(v)**2 for v in row)**0.5
            expected.append([float(v)/norm for v in row])
        np.testing.assert_array_equal(actual, np.array(expected, dtype=np.float32))
        self.assertEqual(actual.dtype, np.dtype('float32'))

    def test_bad_vectors(self):
        for x in [[[0, 0]], [[np.nan, 1]], [[np.inf, 1]]]:
            with self.assertRaises(ValueError):
                normalize_fp32_once(x)


if __name__ == '__main__':
    unittest.main()
