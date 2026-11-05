import unittest
from analyze import classify, exclusive, stats, union


class AnalysisTests(unittest.TestCase):
    def test_projection_cast_not_linear(self):
        self.assertEqual(classify('aten::_to_copy',[],['q_projection']),('copies_casts','data_movement'))
        self.assertEqual(classify('aten::mm',[],['q_projection']),('q_projection','linear'))

    def test_attention_shapes(self):
        self.assertEqual(classify('aten::bmm',[[8,16,256],[8,256,984]],['attention']),('attention_qkt','linear'))
        self.assertEqual(classify('aten::bmm',[[8,16,984],[8,984,256]],['attention']),('attention_sv','linear'))

    def test_normalization_split(self):
        self.assertEqual(classify('aten::rsqrt',[],['rms_norm']),('rms_rsqrt','nonlinear_core'))
        self.assertEqual(classify('aten::mul',[],['rms_norm']),('norm_mul','elementwise'))

    def test_ffn_and_softmax(self):
        self.assertEqual(classify('aten::mul',[],['ffn']),('ffn_gate_multiply','elementwise'))
        self.assertEqual(classify('aten::_softmax',[],['attention']),('softmax','nonlinear_core'))

    def test_union(self):
        self.assertEqual(union([(1,4),(2,3),(4,5),(7,8)]),[[1,5],[7,8]])

    def test_exclusive_overlap(self):
        items=[dict(start=0,end=3000000,category='a'),dict(start=2000000,end=4000000,category='b')]
        self.assertEqual(exclusive(items,'category'),{'a':2.,'overlap':1.,'b':1.})

    def test_same_category_overlap(self):
        items=[dict(start=0,end=3000000,category='a'),dict(start=2000000,end=4000000,category='a')]
        self.assertEqual(exclusive(items,'category'),{'a':4.})

    def test_stats(self):
        self.assertEqual(stats([1,2,3])['p50'],2)
        self.assertAlmostEqual(stats([1,2,3])['p95'],2.9)


if __name__=='__main__':
    unittest.main()
