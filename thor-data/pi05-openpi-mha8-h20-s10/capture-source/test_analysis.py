import unittest
from analyze import family, LINEAR, NONLINEAR


class CategoriesTest(unittest.TestCase):
    def test_disjoint_categories(self):
        self.assertFalse(LINEAR & NONLINEAR)
        for key in LINEAR:
            self.assertEqual(family(key),'Matrix / linear')
        for key in NONLINEAR:
            self.assertEqual(family(key),'Nonlinear / normalization')

    def test_auxiliary_is_not_nonlinear(self):
        for key in ('kv_concat','copy_or_cast','rope_apply','attention_mul','attention_add','residual_gate'):
            self.assertEqual(family(key),'Movement / pointwise / setup')

    def test_rope_is_split(self):
        self.assertEqual(family('rope_frequency_matmul'),'Matrix / linear')
        self.assertEqual(family('rope_cos'),'Nonlinear / normalization')
        self.assertEqual(family('shared_overlap'),'Shared overlap')


if __name__=='__main__':
    unittest.main()
