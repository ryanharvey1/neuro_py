import unittest

import numpy as np

from neuro_py.ensemble.dynamics import (
    potential_landscape,
    potential_landscape_nd,
)


class TestPotentialLandscape(unittest.TestCase):
    def test_basic_case(self):
        X_dyn = np.array([[0.1, 0.2, 0.4], [0.0, 0.3, 0.6]])
        projbins = 3
        domainbins = 3
        potential, gradient, hist, latentedges, domainedges = potential_landscape(X_dyn, projbins, domainbins)
        
        expected_potential = np.array([
            [ 0. ,  0. ,   np.nan],
            [-0.1 ,  0.,   np.nan],
            [np.nan,  0., -0.25]
        ])
        expected_gradient = np.array([
            [0.3 ,  np.nan,  np.nan],
            [0.1 ,  np.nan,  np.nan],
            [np.nan,  np.nan, 0.25]
        ])
        expected_hist = np.array([
            [1., 0., 0.],
            [1., 0., 0.],
            [0., 0., 2.]
        ])
        np.testing.assert_array_almost_equal(potential, expected_potential)
        np.testing.assert_array_almost_equal(gradient, expected_gradient)
        np.testing.assert_array_equal(hist, expected_hist)

    def test_no_domainbins(self):
        X_dyn = np.array([[0.1, 0.2, 0.4], [0.0, 0.3, 0.6]])
        projbins = 3
        potential, gradient, hist, latentedges, domainedges = potential_landscape(X_dyn, projbins)
        self.assertEqual(hist.shape[1], X_dyn.shape[1] - 1)

    def test_empty_input(self):
        X_dyn = np.array([[]])
        projbins = 3
        with self.assertRaises(ValueError):
            potential_landscape(X_dyn, projbins)

    def test_high_dimensional_input(self):
        X_dyn = np.random.random((10, 50))
        projbins = 20
        domainbins = 15
        potential, gradient, hist, latentedges, domainedges = potential_landscape(X_dyn, projbins, domainbins)
        self.assertEqual(hist.shape, (20, 15))


class TestPotentialLandscapeND(unittest.TestCase):
    def test_basic_case(self):
        X_dyn = np.array([[0.1, 0.2, 0.4], [0.0, 0.3, 0.6]])
        projbins = 3
        domainbins = 3
        potential, gradient, hist, latentedges, domainedges = potential_landscape(X_dyn, projbins, domainbins)
        
        potential_nd, potential_pos_t_nd, gradient_nd, hist_nd, latentedges_nd, domainedges_nd = potential_landscape_nd(X_dyn.reshape(2, 3, 1), projbins, domainbins)
        np.testing.assert_array_equal(potential.shape, potential_pos_t_nd[:, :, 0].shape)
        np.testing.assert_array_almost_equal(gradient, gradient_nd[:, :, 0])
        np.testing.assert_array_equal(hist, hist_nd[:, :, 0])
        np.testing.assert_array_equal(latentedges, latentedges_nd[:, 0])
        np.testing.assert_array_equal(domainedges, domainedges_nd[:, 0])

    def test_output_shapes(self):
        nnrns = 3
        X_dyn = np.random.random((5, 10, nnrns))
        projbins = 5
        domainbins = 5
        potential, _, _, hist, latentedges, domainedges = potential_landscape_nd(X_dyn, projbins, domainbins)

        self.assertEqual(potential.shape, tuple([projbins for _ in range(nnrns)]))
        self.assertEqual(hist.shape, (*[projbins for _ in range(nnrns)], domainbins, nnrns))
        self.assertEqual(len(latentedges), projbins+1)
        self.assertEqual(len(domainedges), domainbins+1)

    def test_no_domainbins(self):
        X_dyn = np.random.random((5, 10, 3))
        projbins = 5
        _, potential_pos_t, _, _, _, _ = potential_landscape_nd(X_dyn, projbins)
        self.assertGreater(potential_pos_t.shape[-2], 0) 

    def test_empty_input(self):
        X_dyn = np.empty((0, 0, 0))
        projbins = 5
        with self.assertRaises(ValueError):
            potential_landscape_nd(X_dyn, projbins)

    def test_nanborder_handling(self):
        X_dyn = np.random.random((5, 10, 3))
        projbins = 5
        _, potential, _, _, _, _ = potential_landscape_nd(X_dyn, projbins, nanborderempty=True)
        self.assertTrue(np.isnan(potential).any())

