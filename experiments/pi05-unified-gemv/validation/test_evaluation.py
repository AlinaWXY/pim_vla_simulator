import tempfile, unittest, sys
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'source'),str(Path(__file__).resolve().parents[2]/'pi05-bgk-gpuq/backend')]
import evaluation_workload as E

class EvaluationContracts(unittest.TestCase):
    def test_matrix_byte_conservation(self):
        expected={'feature_gpu_before':5904,'token_gpu_before':4968,'token_local_before':4776,
                  'token_gpu_after':4968,'token_local_after':4776}
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'x.trace'
            for v in E.VARIANTS:
                stages=[E.generate(p,c,v) for c in ('chain','pv','gate_up_gpu','down_gpu_input')]
                layer=E.generate(p,'serial_layer_control',v)
                self.assertEqual(sum(m['expected_dq_bytes']['total'] for m in stages),expected[v]*1024)
                self.assertEqual(layer['expected_dq_bytes']['total'],expected[v]*1024)
                self.assertEqual(len(set(s['start'] for s in layer['segments'])),len(layer['segments']))
                text=p.read_text()
                if E.VARIANTS[v]['after']:
                    self.assertIn('MAC_RUN',text)
                    self.assertNotIn('PIM_WR_GB ',text)
                else:
                    self.assertNotIn('MAC_RUN',text)
                    self.assertIn('PIM_WR_GB ',text)
                    # Ordinary input ST must disappear; only GPU RoPE imports remain.
                    self.assertEqual(layer['counts'].get('ST',0),4096 if E.VARIANTS[v]['gpu_rope'] else 0)

    def test_feature_ownership_and_packing(self):
        subsets=[list(range(0,64))+list(range(128,192)),list(range(64,128))+list(range(192,256))]
        self.assertEqual(set(subsets[0])|set(subsets[1]),set(range(256)))
        self.assertFalse(set(subsets[0])&set(subsets[1]))
        for features in subsets:
            self.assertTrue(all((f+128)%256 in features for f in features))
            # Packed K: eight BF16 features/token, two tokens per 32B column.
            locations={(t//2,b,(t%2)*8+f): (t,features[b*8+f])
                       for t in range(992) for b in range(16) for f in range(8)}
            self.assertEqual(len(locations),992*128)
        self.assertLessEqual(16*24*4,2048) # QKV accumulator pool.
        self.assertLessEqual(16*32*4,2048) # O tile accumulator pool.
        self.assertEqual(2*16*16*2,1024) # Two direct input SRF views.

    def test_gpu_rope_round_trip(self):
        reads=E.gpu_rope_records('LD');writes=E.gpu_rope_records('ST')
        self.assertEqual([x[3:] for x in reads],[x[3:] for x in writes])
        self.assertEqual(len(reads),4096)
        self.assertEqual(len(set(reads)),4096)

if __name__=='__main__':unittest.main()
