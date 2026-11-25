import unittest

import numpy as np

from proofs import collapse, encoded, inputs, shared_carry
from research import capacity,metrics,sample


class IntegerReference(unittest.TestCase):
    def test_capacity_endpoint(self):
        for base in (16,22,26,30,38,64,128):
            for limit in (63,127):
                cap=capacity(base,limit)
                h,g=encoded(np.arange(-cap-1,cap+2),base)
                valid=(np.abs(h)<=limit)&(np.abs(g)<=limit)
                self.assertTrue(valid[1:-1].all())
                self.assertTrue(not valid[0] or not valid[-1])
                self.assertLessEqual(4096*int(np.max((g[1:-1]-h[1:-1])**2)),2**31-1)

    def test_strassen_and_exact_reconstruction(self):
        rng=np.random.default_rng(815); base=64; cap=capacity(base,63)
        a,b=rng.integers(-cap,cap+1,(2,64,64),dtype=np.int64)
        ha,ga=encoded(a,base);hb,gb=encoded(b,base)
        dots=lambda x,y:collapse([u@v for u,v in inputs(x,y)])
        effective=base*((base-1)*dots(ha,hb)+dots(ga,gb))-(base-1)*dots(ga-ha,gb-hb)
        np.testing.assert_array_equal(effective,a@b)

    def test_carry_legal_and_illegal(self):
        x=np.arange(-254,255)[:,None];y=x+np.arange(-44,45)
        for step in (128,192):
            h,g,c=shared_carry(x,y,step)
            np.testing.assert_array_equal(g-h,y-x)
            self.assertTrue(np.all(np.abs(h)<=127) and np.all(np.abs(g)<=127))
            np.testing.assert_array_equal(h+step*c,np.broadcast_to(x,y.shape))
        with self.assertRaises(ValueError):
            shared_carry(np.array([-200]),np.array([200]),128)

    def test_metric_signed_zero_and_half_overflow(self):
        fp32=np.array([[0.,-0.,4e7]],dtype=np.float32)
        report=metrics(fp32,np.array([[0.,0.,4e7]],dtype=np.float64))
        self.assertEqual(report['relative_frobenius'],0.)
        self.assertEqual(report['fp16_finite_count'],2)
        self.assertEqual(report['fp16_inf'],1)
        self.assertEqual(report['fp16_bitmatch'],2/3)
        a,b=sample((32,32,32),'finite_range')
        self.assertTrue(np.isfinite(a).all() and np.isfinite(b).all())
        self.assertEqual(np.max(np.abs(a)),65504)


if __name__=='__main__':unittest.main()
