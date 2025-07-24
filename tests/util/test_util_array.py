import unittest

import numpy as np

from neuro_py.util.array import (
    find_terminal_masked_indices,
    replace_border_zeros_with_nan,
)


class TestFindTerminalMaskedIndices(unittest.TestCase):
    def test_1d_array(self):
        mask = np.array([0, 0, 1, 1, 0])
        first_idx, last_idx = find_terminal_masked_indices(mask, axis=0)
        self.assertEqual(first_idx, 2)
        self.assertEqual(last_idx, 3)

    def test_2d_array_axis_0(self):
        mask = np.array([[0, 0, 1],
                         [1, 1, 0],
                         [0, 0, 0]])
        first_idx, last_idx = find_terminal_masked_indices(mask, axis=0)
        np.testing.assert_array_equal(first_idx, [1, 1, 0])
        np.testing.assert_array_equal(last_idx, [1, 1, 0])

    def test_2d_array_axis_1(self):
        mask = np.array([[0, 0, 1],
                         [1, 1, 0],
                         [0, 0, 0]])
        first_idx, last_idx = find_terminal_masked_indices(mask, axis=1)
        np.testing.assert_array_equal(first_idx, [2, 0, 0])
        np.testing.assert_array_equal(last_idx, [2, 1, 2])

    def test_empty_mask(self):
        mask = np.array([[0, 0, 0],
                         [0, 0, 0]])
        first_idx, last_idx = find_terminal_masked_indices(mask, axis=0)
        np.testing.assert_array_equal(first_idx, [0, 0, 0])  # argmax defaults to 0 for all-0 inputs
        np.testing.assert_array_equal(last_idx, [1, 1, 1])  # Adjusted correctly for flipped argmax

    def test_full_mask(self):
        mask = np.array([[1, 1, 1],
                         [1, 1, 1]])
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
        arr = np.array([
            [0, 1, 2],
            [3, 0, 5],
            [0, 0, 0]
        ])
        result = replace_border_zeros_with_nan(arr)
        expected = np.array([
            [np.nan, 1, 2],
            [3, np.nan, 5],
            [np.nan, np.nan, np.nan]
        ])
        np.testing.assert_array_equal(result, expected)

    def test_3d_array(self):
        arr = np.arange(27).reshape(3, 3, 3)
        arr[0, 2] = arr[2, 2] = arr[2, 0, 0] = arr[1, 1, 1] = 0
        result = replace_border_zeros_with_nan(arr)
        expected = np.array([
            [[np.nan,  1.,  2.],
             [ 3.,  4.,  5.],
             [np.nan, np.nan, np.nan]],
    
            [[ 9., 10., 11.],
             [12.,  0., 14.],
             [15., 16., 17.]],
    
            [[np.nan, 19., 20.],
             [21., 22., 23.],
             [np.nan, np.nan, np.nan]]
        ])
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
        arr = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])
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
