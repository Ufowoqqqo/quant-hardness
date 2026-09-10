"""Small deterministic offline rank/tie and signed-ratio controls."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('analysis',Path(__file__).resolve().parents[1]/'scripts/analyze_phase5b.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class AnalysisTests(unittest.TestCase):
    def test_signed_recovery(self):
        self.assertEqual(module.ratio(2,1),2)
        self.assertEqual(module.ratio(-1,2),-.5)
        self.assertIsNone(module.ratio(1,0))

    def test_offline_rank_ties_and_membership(self):
        with tempfile.TemporaryDirectory(prefix='phase5b-rank-test-') as directory:
            root=Path(directory)
            gt=np.tile(np.arange(11,dtype='<i8'),(10000,1));gt.tofile(root/'gt_ids.i64')
            # Rank tie at score0: ID9 arrived before ID1, so9 must rank first.
            ids=np.array([9,1,2,3,4,5,6,7,8,0,10,11,12,13,14,15],dtype='<i4')
            scores=np.arange(16,dtype='<f4');scores[1]=0
            for ef in (32,64,128):
                path=root/f'recall_ef{ef}';path.mkdir()
                np.arange(0,160001,16,dtype='<i8').tofile(path/'pq_offsets.i64')
                np.tile(ids,10000).tofile(path/'pq_candidates.i32')
                np.tile(scores,10000).tofile(path/'pq_scores.f32')
                np.tile(np.arange(10,dtype='<i8'),(10000,1)).tofile(path/'oracle_ids.i64')
            iterator=module.rank_records(root)
            rows=[next(iterator) for _ in range(10)]
            self.assertEqual(rows[9]['pq_rank'],1)
            self.assertEqual(rows[1]['pq_rank'],2)
            self.assertEqual(rows[0]['pq_rank'],10)
            self.assertTrue(all(r['gt_relevant']==1 for r in rows))
            self.assertTrue(all(r['ef']==32 and r['query_id']==0 for r in rows))


if __name__=='__main__':unittest.main()
