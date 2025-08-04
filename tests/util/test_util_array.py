import unittest

import numpy as np

from neuro_py.util.array import (
    find_terminal_masked_indices,
    replace_border_zeros_with_nan,
    circular_interp,
)


class TestFindTerminalMaskedIndices(unittest.TestCase):
    def test_1d_array(self):
        mask = np.array([0, 0, 1, 1, 0])
        first_idx, last_idx = find_terminal_masked_indices(mask, axis=0)
        self.assertEqual(first_idx, 2)
        self.assertEqual(last_idx, 3)

    def test_2d_array_axis_0(self):
        mask = np.array([[0, 0, 1], [1, 1, 0], [0, 0, 0]])
        first_idx, last_idx = find_terminal_masked_indices(mask, axis=0)
        np.testing.assert_array_equal(first_idx, [1, 1, 0])
        np.testing.assert_array_equal(last_idx, [1, 1, 0])

    def test_2d_array_axis_1(self):
        mask = np.array([[0, 0, 1], [1, 1, 0], [0, 0, 0]])
        first_idx, last_idx = find_terminal_masked_indices(mask, axis=1)
        np.testing.assert_array_equal(first_idx, [2, 0, 0])
        np.testing.assert_array_equal(last_idx, [2, 1, 2])

    def test_empty_mask(self):
        mask = np.array([[0, 0, 0], [0, 0, 0]])
        first_idx, last_idx = find_terminal_masked_indices(mask, axis=0)
        np.testing.assert_array_equal(
            first_idx, [0, 0, 0]
        )  # argmax defaults to 0 for all-0 inputs
        np.testing.assert_array_equal(
            last_idx, [1, 1, 1]
        )  # Adjusted correctly for flipped argmax

    def test_full_mask(self):
        mask = np.array([[1, 1, 1], [1, 1, 1]])
        first_idx, last_idx = find_terminal_masked_indices(mask, axis=1)
        np.testing.assert_array_equal(first_idx, [0, 0])
        np.testing.assert_array_equal(last_idx, [2, 2])


class TestReplaceBorderZerosWithNaN(unittest.TestCase):
    def test_1d_array(self):
        arr = np.array([0, 1, 2, 0, 0])
        result = replace_border_zeros_with_nan(arr)
        expected = np.array([np.nan, 1, 2, np.nan, np.nan])
        np.testing.assert_array_equal(result, expected)

    def test_2d_array(self):
        arr = np.array([[0, 1, 2], [3, 0, 5], [0, 0, 0]])
        result = replace_border_zeros_with_nan(arr)
        expected = np.array([[np.nan, 1, 2], [3, np.nan, 5], [np.nan, np.nan, np.nan]])
        np.testing.assert_array_equal(result, expected)

    def test_3d_array(self):
        arr = np.arange(27).reshape(3, 3, 3)
        arr[0, 2] = arr[2, 2] = arr[2, 0, 0] = arr[1, 1, 1] = 0
        result = replace_border_zeros_with_nan(arr)
        expected = np.array(
            [
                [[np.nan, 1.0, 2.0], [3.0, 4.0, 5.0], [np.nan, np.nan, np.nan]],
                [[9.0, 10.0, 11.0], [12.0, 0.0, 14.0], [15.0, 16.0, 17.0]],
                [[np.nan, 19.0, 20.0], [21.0, 22.0, 23.0], [np.nan, np.nan, np.nan]],
            ]
        )
        np.testing.assert_array_equal(result, expected)

    def test_empty_array(self):
        arr = np.array([])
        result = replace_border_zeros_with_nan(arr)
        expected = np.array([])
        np.testing.assert_array_equal(result, expected)

    def test_all_zeros(self):
        arr = np.zeros((3, 3))
        result = replace_border_zeros_with_nan(arr)
        expected = np.full((3, 3), np.nan)
        np.testing.assert_array_equal(result, expected)

    def test_no_zeros(self):
        arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = replace_border_zeros_with_nan(arr)
        np.testing.assert_array_equal(result, arr)  # No change expected

    def test_single_element_array(self):
        arr = np.array([0])
        result = replace_border_zeros_with_nan(arr)
        expected = np.array([np.nan])
        np.testing.assert_array_equal(result, expected)

    def test_high_dimensional_array(self):
        arr = np.zeros((2, 2, 2, 2))
        arr[0, 0, 0, 0] = 1  # Add non-zero in one corner
        result = replace_border_zeros_with_nan(arr)
        expected = np.full((2, 2, 2, 2), np.nan)
        expected[0, 0, 0, 0] = 1  # Preserve the non-zero value
        np.testing.assert_array_equal(result, expected)


class TestCircularInterp(unittest.TestCase):
    def test_basic_interpolation_neg_pi_to_pi(self):
        """Test basic interpolation with data in [-π, π] range."""
        xp = np.array([0, 1, 2])
        fp = np.array([-np.pi / 2, 0, np.pi / 2])
        x = np.array([0.5, 1.5])

        result = circular_interp(x, xp, fp)

        # Should stay in [-π, π] range
        self.assertTrue(np.all(result >= -np.pi))
        self.assertTrue(np.all(result <= np.pi))

        # Check approximate values
        np.testing.assert_allclose(result[0], -np.pi / 4, rtol=1e-2)
        np.testing.assert_allclose(result[1], np.pi / 4, rtol=1e-2)

    def test_basic_interpolation_zero_to_2pi(self):
        """Test basic interpolation with data in [0, 2π] range."""
        xp = np.array([0, 1, 2])
        fp = np.array([0, np.pi / 2, np.pi])
        x = np.array([0.5, 1.5])

        result = circular_interp(x, xp, fp)

        # Should stay in [0, 2π] range
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 2 * np.pi))

        # Check approximate values
        np.testing.assert_allclose(result[0], np.pi / 4, rtol=1e-2)
        np.testing.assert_allclose(result[1], 3 * np.pi / 4, rtol=1e-2)

    def test_range_preservation_zero_to_2pi(self):
        """Test that [0, 2π] range is preserved when all fp >= 0."""
        xp = np.array([0, 1, 2, 3])
        fp = np.array([0.1, np.pi, 1.5 * np.pi, 0.2])  # All positive, spans [0, 2π]
        x = np.array([0.5, 1.5, 2.5])

        result = circular_interp(x, xp, fp)

        # All results should be in [0, 2π]
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 2 * np.pi))

    def test_range_preservation_neg_pi_to_pi(self):
        """Test that [-π, π] range is preserved when fp contains negative values."""
        xp = np.array([0, 1, 2, 3])
        fp = np.array(
            [-np.pi / 2, 0, np.pi / 2, -np.pi / 4]
        )  # Contains negative values
        x = np.array([0.5, 1.5, 2.5])

        result = circular_interp(x, xp, fp)

        # Results should be in [-π, π]
        self.assertTrue(np.all(result >= -np.pi))
        self.assertTrue(np.all(result <= np.pi))

    def test_circular_boundary_crossing(self):
        """Test interpolation across circular boundaries."""
        # Test near 0/2π boundary for [0, 2π] data
        xp = np.array([0, 1, 2])
        fp = np.array([0.1, np.pi, 2 * np.pi - 0.1])  # Near 0 and 2π
        x = np.array([0.5, 1.5])

        result = circular_interp(x, xp, fp)

        # Should handle circular interpolation correctly
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 2 * np.pi))

        # Test near -π/π boundary for [-π, π] data
        xp = np.array([0, 1, 2])
        fp = np.array([-np.pi + 0.1, 0, np.pi - 0.1])
        x = np.array([0.5, 1.5])

        result = circular_interp(x, xp, fp)

        self.assertTrue(np.all(result >= -np.pi))
        self.assertTrue(np.all(result <= np.pi))

    def test_input_validation(self):
        """Test input validation and error handling."""
        xp = np.array([0, 1, 2])
        fp = np.array([0, np.pi / 2, np.pi])

        # Test mismatched lengths
        with self.assertRaises(ValueError):
            circular_interp(np.array([0.5]), np.array([0, 1]), np.array([0]))

        # Test insufficient points
        with self.assertRaises(ValueError):
            circular_interp(np.array([0.5]), np.array([0]), np.array([0]))

        # Test non-increasing xp
        with self.assertRaises(ValueError):
            circular_interp(np.array([0.5]), np.array([1, 0, 2]), np.array([0, 1, 2]))

    def test_edge_cases(self):
        """Test edge cases and special values."""
        # Test with exactly 2 points (minimum)
        xp = np.array([0, 1])
        fp = np.array([0, np.pi])
        x = np.array([0.5])

        result = circular_interp(x, xp, fp)
        self.assertEqual(len(result), 1)
        self.assertTrue(0 <= result[0] <= 2 * np.pi)

        # Test interpolation at exact data points
        xp = np.array([0, 1, 2])
        fp = np.array([0, np.pi / 2, np.pi])
        x = np.array([0, 1, 2])  # Exact matches

        result = circular_interp(x, xp, fp)
        np.testing.assert_allclose(result, fp, rtol=1e-10)

    def test_mixed_ranges(self):
        """Test behavior with edge case ranges."""
        # Test data that spans exactly [0, π] (ambiguous case)
        xp = np.array([0, 1, 2])
        fp = np.array([0, np.pi / 2, np.pi])
        x = np.array([0.5, 1.5])

        result = circular_interp(x, xp, fp)

        # Since all fp >= 0, should be treated as [0, 2π] range
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 2 * np.pi))

    def test_full_circle_interpolation(self):
        """Test interpolation over a full circle."""
        # Test complete [0, 2π] cycle
        xp = np.array([0, 1, 2, 3, 4])
        fp = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
        x = np.array([0.5, 1.5, 2.5, 3.5])

        result = circular_interp(x, xp, fp)

        # Verify all results are in [0, 2π]
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 2 * np.pi))

        # Check that interpolation is reasonable
        expected_approx = np.array(
            [np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4, 7 * np.pi / 4]
        )
        np.testing.assert_allclose(result, expected_approx, rtol=0.1)