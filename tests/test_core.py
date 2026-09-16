import unittest
import numpy as np
from raft_consistency_vfi.core import bilinear_sample, consistency, endpoint_valid, forward_splat, grid, interpolate
from raft_consistency_vfi.metrics import quality


class GeometryTests(unittest.TestCase):
    def test_zero_flow_identity(self):
        a=np.random.default_rng(5).random((15,20,3),dtype=np.float32)
        f=np.zeros((15,20,2),np.float32)
        out,holes=interpolate(a,a,f,f)
        np.testing.assert_allclose(out,a,atol=1e-7)
        self.assertFalse(holes.any())

    def test_signed_translation_midpoint(self):
        a=np.zeros((20,30,3),np.float32); b=a.copy()
        a[4:16,4:10]=1; b[4:16,12:18]=1
        f=np.zeros((20,30,2),np.float32); f[...,0]=8
        out,_=interpolate(a,b,f,-f)
        expected=np.zeros_like(a); expected[4:16,8:14]=1
        np.testing.assert_allclose(out,expected,atol=1e-7)

    def test_subpixel_mass_conservation(self):
        a=np.zeros((8,9,3),np.float32); a[3,4]=1
        f=np.zeros((8,9,2),np.float32); f[...,0]=0.25; f[...,1]=0.5
        weights=np.zeros((8,9),np.float32); weights[3,4]=1
        numerator,denominator=forward_splat(a,f,weights)
        self.assertAlmostEqual(float(denominator.sum()),1)
        np.testing.assert_allclose(numerator.sum((0,1)),np.ones(3))
        self.assertAlmostEqual(float(denominator[3,4]),0.375)

    def test_reverse_flow_sampled_at_mapped_location(self):
        h,w=8,12
        f=np.zeros((h,w,2),np.float32); f[...,0]=1
        reverse=np.zeros_like(f); reverse[:,1:,0]=-1
        c=consistency(f,reverse)
        self.assertEqual(c["hard"][3,0],1)
        self.assertEqual(c["hard"][3,-1],0)

    def test_inconsistent_flow_is_rejected(self):
        f=np.ones((12,12,2),np.float32)*2
        c=consistency(f,f)
        self.assertFalse(c["hard"].any())
        self.assertLess(c["soft"][2,2],0.001)

    def test_geometric_and_consistency_validity_agree(self):
        f=np.zeros((8,12,2),np.float32); f[...,0]=2.5
        np.testing.assert_array_equal(endpoint_valid(f),consistency(f,-f)["valid"])

    def test_collision_uses_weighted_sum(self):
        a=np.zeros((4,4,3),np.float32); a[1,1]=1
        f=np.zeros((4,4,2),np.float32); f[1,2,0]=-1
        weight=np.zeros((4,4),np.float32); weight[1,1]=3; weight[1,2]=1
        n,d=forward_splat(a,f,weight)
        np.testing.assert_allclose(n[1,1]/d[1,1],0.75)

    def test_all_rejected_uses_same_fallback(self):
        a=np.ones((12,12,3),np.float32); b=np.zeros_like(a)
        f=np.zeros((12,12,2),np.float32); weight=np.zeros((12,12),np.float32)
        result,holes=interpolate(a,b,f,f,weight,weight)
        np.testing.assert_allclose(result,0.5)
        self.assertTrue(holes.all())

    def test_outside_contributions_do_not_wrap(self):
        a=np.ones((3,3,3),np.float32)
        f=np.ones((3,3,2),np.float32)*100
        n,d=forward_splat(a,f,np.ones((3,3)))
        self.assertEqual(float(n.sum()+d.sum()),0)

    def test_endpoints_are_exact(self):
        a=np.ones((4,4,3),np.float32); b=np.zeros_like(a); f=np.ones((4,4,2),np.float32)
        np.testing.assert_array_equal(interpolate(a,b,f,-f,t=0)[0],a)
        np.testing.assert_array_equal(interpolate(a,b,f,-f,t=1)[0],b)

    def test_nonfinite_flow_rejected(self):
        f=np.zeros((3,3,2)); f[0,0]=np.nan
        with self.assertRaises(ValueError): consistency(f,f)


class MetricTests(unittest.TestCase):
    def test_identical_images(self):
        a=np.random.default_rng(2).random((25,30,3))
        m=quality(a,a)
        self.assertEqual(m["psnr"],float("inf"))
        self.assertAlmostEqual(m["ssim"],1,places=12)

    def test_known_constant_difference(self):
        a=np.full((20,20,3),0.2); b=np.full_like(a,0.3)
        m=quality(a,b)
        self.assertAlmostEqual(m["psnr"],20,places=10)
        self.assertAlmostEqual(m["ssim"],(2*0.2*0.3+0.0001)/(0.2**2+0.3**2+0.0001),places=9)


if __name__=="__main__": unittest.main()
