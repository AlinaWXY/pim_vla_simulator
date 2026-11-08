"""Behavioral checks for the user-selected cross-window scheduling rule."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replay_idle_service import replay


def block(kind, start, end, windows=()):
    return dict(kind=kind, name=kind, begin_us=start, end_us=end, windows=windows)


class IdleServiceTests(unittest.TestCase):
    def run_case(self, blocks, times):
        return replay(blocks, [dict(us=t, kind='kernel') for t in times], lambda b:b['windows'])

    def test_kernel_larger_than_every_window_crosses_then_delays_AE(self):
        r = self.run_case([block('pim',0,10,[(1,3),(6,9)]),block('gpu',10,12)], [8,4])
        self.assertAlmostEqual(r['VLM_progress_during_PIM_ms'], .005)
        self.assertAlmostEqual(r['AE_gpu_wait_ms'], .003)
        self.assertAlmostEqual(r['AE_with_interference_ms'], .015)
        self.assertAlmostEqual(r['total_ms'], .019)
        self.assertEqual(r['cross_window_units'],1)
        self.assertEqual(r['started_VLM_units'],1)

    def test_order_prevents_short_kernel_overtaking_long_one(self):
        blocks=[block('pim',0,5,[(0,2),(3,4)]),block('gpu',5,6),
                block('pim',6,8,[(0,2)]),block('gpu',8,9)]
        r=self.run_case(blocks,[10,1,1])
        self.assertEqual([x['ordinal'] for x in r['placements']],[0,1,2])
        self.assertAlmostEqual(r['waits'][0]['wait_us'],7)
        self.assertAlmostEqual(r['placements'][1]['start_us'],13)

    def test_consecutive_PIM_blocks_do_not_drain_kernel(self):
        r=self.run_case([block('pim',0,4,[(1,3)]),block('pim',4,8,[(0,3)]),block('gpu',8,9)],[7])
        self.assertAlmostEqual(r['AE_gpu_wait_ms'],.002)
        self.assertEqual(len(r['waits']),1)

    def test_no_idle_means_no_launch_and_no_AE_wait(self):
        r=self.run_case([block('pim',0,10),block('gpu',10,13)],[7])
        self.assertEqual(r['started_VLM_units'],0)
        self.assertEqual(r['AE_gpu_wait_ms'],0)
        self.assertAlmostEqual(r['total_ms'],.020)

    def test_finished_VLM_leaves_later_windows_unused(self):
        r=self.run_case([block('pim',0,10,[(0,10)]),block('gpu',10,12)],[2,1])
        self.assertAlmostEqual(r['unused_window_ms'],.007)
        self.assertAlmostEqual(r['total_ms'],.012)

    def test_sub_microsecond_windows_are_usable(self):
        r=self.run_case([block('pim',0,1,[(.1,.100938)]),block('gpu',1,2)],[.01])
        self.assertAlmostEqual(r['VLM_progress_during_PIM_ms'],.000000938)
        self.assertAlmostEqual(r['AE_gpu_wait_ms'],.000009062)

    def test_final_PIM_tail_does_not_delay_already_finished_AE(self):
        r=self.run_case([block('pim',0,3,[(0,1)])],[5])
        self.assertAlmostEqual(r['AE_with_interference_ms'],.003)
        self.assertAlmostEqual(r['total_ms'],.007)

    def test_full_availability_does_not_restore_whole_kernel_fitting(self):
        r=self.run_case([block('pim',0,2,[(0,2)]),block('gpu',2,5)],[100])
        self.assertEqual(r['started_VLM_units'],1)
        self.assertAlmostEqual(r['AE_gpu_wait_ms'],.098)


if __name__=='__main__':
    unittest.main()
