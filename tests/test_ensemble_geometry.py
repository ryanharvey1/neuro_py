import unittest
import numpy as np

from neuro_py.ensemble.geometry import (
    proximity,
)


class TestProximity(unittest.TestCase):
    def setUp(self):
        N_POINTS = 20
        x = np.sin(np.linspace(0, 2 * np.pi, N_POINTS))
        y = np.cos(np.linspace(0, 2 * np.pi, N_POINTS))
        z1 = np.linspace(0, 1, N_POINTS)
        z2 = np.linspace(0, 1, N_POINTS) + np.sin(np.linspace(0.25, 0.75, N_POINTS))
        self.prox_traj1 = np.array([x, y, z1]).T
        self.prox_traj2 = np.array([x, y, z2]).T

    def test_proximity_output_shape(self):
        result = proximity(self.prox_traj1, self.prox_traj2)
        self.assertEqual(result.shape, (20,))

    def test_proximity_sum_positive(self):
        result = proximity(self.prox_traj1, self.prox_traj2)
        self.assertGreater(np.sum(result), 0)


if __name__ == '__main__':
    unittest.main()
