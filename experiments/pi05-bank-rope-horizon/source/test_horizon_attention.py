"""Geometric and trace checks independent of native timing implementation."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import horizon_attention as A
import horizon_attention_layout as L


class AttentionTests(unittest.TestCase):
    def test_layout_sizes_and_work_rows(self):
        for h, kv, cols, end in ((16,992,28,83968),(20,992,32,83968),
                                  (30,1024,58,83968),(40,1024,68,86016)):
            with self.subTest(h=h):
                layout=A.layout_for(h)
                self.assertEqual(layout['shape']['padded_sequence'],kv)
                self.assertEqual(layout['regions']['work']['used_bytes_per_bank'],cols*32)
                self.assertEqual(layout['end'],end)
                self.assertEqual(layout['buffers']['qk_query_per_bank'],h*32)
                self.assertEqual(layout['buffers']['projection_accumulator_per_bank'],h*64)
                self.assertEqual(layout['initialization_bytes']['coefficients'],h*4096)
                self.assertEqual(layout['qk_weight_system_bytes'],16*1024*1024)
                self.assertLessEqual(layout['work_columns']['k_suffix'][1],64)

    def test_layout_snapshot_does_not_mutate_configured_horizon(self):
        A.configure(20)
        first=A.layout_for(40)
        second=A.layout_for(30)
        self.assertEqual(L.QUERIES,20)
        self.assertEqual(first['work_columns']['q_local'],(20,40))
        self.assertEqual(second['work_columns']['q_local'],(15,30))

    def test_query_ownership_and_buffer_slots(self):
        for h in (16,20,30,40):
            A.configure(h)
            for query in range(h):
                for bank in range(16):
                    for lane in (0,15):
                        feature=L.feature_from_lane(bank,lane)
                        c,r,bg,b,offset=L.query_location(3,query,feature)
                        self.assertEqual((c,r,bg*4+b),(6+query%2,1,bank))
                        self.assertEqual(offset,81920+(h//2+query//2)*32+2*lane)
                        self.assertEqual(L.query_buffer_slot(query,feature),query*32+2*lane)

    def test_k_coverage_once_including_padding(self):
        for h in (20,30,40):
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'trace'
                m=A.generate(path,horizon=h)
                kv=((968+h+31)//32)*32
                counts=m['counts']
                self.assertEqual(counts['PIM_MAC_AB'],64*32*16)
                self.assertEqual(counts['PIM_WR_GB'],64*(h//2)*16)
                self.assertEqual(counts['PIM_MAC_QK'],kv//2*16)
                self.assertEqual(counts['PIM_RD_BG']*32,8*h*kv*4)
                self.assertEqual(counts['PIM_Q_REMOTE'],h//2*256)
                self.assertEqual(counts['PIM_ROPE_EXEC'],h*256)
                addresses={int(line.split()[1]) for line in path.read_text().splitlines()
                           if line.startswith('PIM_MAC_QK ')}
                self.assertEqual(len(addresses),counts['PIM_MAC_QK'])
                self.assertTrue(all(A.decode_address(a)[1]==1 for a in addresses))

    def test_h16_trace_is_identical_to_frozen_generator(self):
        old=Path(__file__).resolve().parents[2]/'pi05-bank-rope-h16/source'
        def load(name,path):
            spec=importlib.util.spec_from_file_location(name,path)
            module=importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        frozen_layout=load('frozen_h16_attention_layout',old/'attention_layout.py')
        previous=sys.modules.get('attention_layout')
        sys.modules['attention_layout']=frozen_layout
        try:
            frozen=load('frozen_h16_attention_workload',old/'attention_workload.py')
        finally:
            if previous is None:
                sys.modules.pop('attention_layout',None)
            else:
                sys.modules['attention_layout']=previous
        with tempfile.TemporaryDirectory() as directory:
            first,second=Path(directory)/'old',Path(directory)/'new'
            frozen.generate(first)
            A.generate(second,horizon=16)
            self.assertEqual(first.read_bytes(),second.read_bytes())

    def test_invalid_horizons(self):
        for invalid in (0,1,15,65,True,20.0):
            with self.assertRaises(ValueError):
                A.configure(invalid)


if __name__=='__main__':
    unittest.main()
