import unittest
import numpy as np


class MHA8Test(unittest.TestCase):
    def test_kv_expansion(self):
        k=np.arange(4*3).reshape(1,4,3)
        expanded=np.repeat(k,8,axis=0)
        self.assertEqual(expanded.shape,(8,4,3))
        self.assertFalse(np.shares_memory(k,expanded))
        for head in expanded:
            np.testing.assert_array_equal(head,k[0])

    def test_attention_equivalence(self):
        rng=np.random.default_rng(17)
        q=rng.normal(size=(8,3,4));k=rng.normal(size=(1,5,4));v=rng.normal(size=(1,5,4))
        def attn(key,value):
            scores=q@key.transpose(0,2,1)/2
            weights=np.exp(scores-scores.max(-1,keepdims=True))
            weights/=weights.sum(-1,keepdims=True)
            return weights@value
        np.testing.assert_allclose(attn(k,v),attn(np.repeat(k,8,axis=0),np.repeat(v,8,axis=0)))

    def test_geometry(self):
        self.assertEqual(8*256,2048)
        self.assertEqual(18*2*8*968*256*2/1024**2,136.125)
        self.assertEqual(18*(4*1024*2048+3*1024*4096),377487360)


if __name__=='__main__':
    unittest.main()
