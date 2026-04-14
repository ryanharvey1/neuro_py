import numpy as np
import pandas as pd
import pytest
from scipy.ndimage import gaussian_filter1d

from neuro_py.util.array import (
    circular_interp,
    find_terminal_masked_indices,
    replace_border_zeros_with_nan,
    shrink,
    smooth_peth,
    zscore_columns,
)


class TestFindTerminalMaskedIndices:
    def test_1d_array(self):
        mask = np.array([0, 0, 1, 1, 0])
        first_idx, last_idx = find_terminal_masked_indices(mask, axis=0)
        assert first_idx == 2
        assert last_idx == 3

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


class TestReplaceBorderZerosWithNaN:
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


class TestCircularInterp:
    def test_basic_interpolation_neg_pi_to_pi(self):
        """Test basic interpolation with data in [-π, π] range."""
        xp = np.array([0, 1, 2])
        fp = np.array([-np.pi / 2, 0, np.pi / 2])
        x = np.array([0.5, 1.5])

        result = circular_interp(x, xp, fp)

        # Should stay in [-π, π] range
        assert np.all(result >= -np.pi)
        assert np.all(result <= np.pi)

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
        assert np.all(result >= 0)
        assert np.all(result <= 2 * np.pi)

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
        assert np.all(result >= 0)
        assert np.all(result <= 2 * np.pi)

    def test_range_preservation_neg_pi_to_pi(self):
        """Test that [-π, π] range is preserved when fp contains negative values."""
        xp = np.array([0, 1, 2, 3])
        fp = np.array(
            [-np.pi / 2, 0, np.pi / 2, -np.pi / 4]
        )  # Contains negative values
        x = np.array([0.5, 1.5, 2.5])

        result = circular_interp(x, xp, fp)

        # Results should be in [-π, π]
        assert np.all(result >= -np.pi)
        assert np.all(result <= np.pi)

    def test_circular_boundary_crossing(self):
        """Test interpolation across circular boundaries."""
        # Test near 0/2π boundary for [0, 2π] data
        xp = np.array([0, 1, 2])
        fp = np.array([0.1, np.pi, 2 * np.pi - 0.1])  # Near 0 and 2π
        x = np.array([0.5, 1.5])

        result = circular_interp(x, xp, fp)

        # Should handle circular interpolation correctly
        assert np.all(result >= 0)
        assert np.all(result <= 2 * np.pi)

        # Test near -π/π boundary for [-π, π] data
        xp = np.array([0, 1, 2])
        fp = np.array([-np.pi + 0.1, 0, np.pi - 0.1])
        x = np.array([0.5, 1.5])

        result = circular_interp(x, xp, fp)

        assert np.all(result >= -np.pi)
        assert np.all(result <= np.pi)

    def test_input_validation(self):
        """Test input validation and error handling."""
        xp = np.array([0, 1, 2])
        fp = np.array([0, np.pi / 2, np.pi])

        # Test mismatched lengths
        with pytest.raises(ValueError):
            circular_interp(np.array([0.5]), np.array([0, 1]), np.array([0]))

        # Test insufficient points
        with pytest.raises(ValueError):
            circular_interp(np.array([0.5]), np.array([0]), np.array([0]))

        # Test non-increasing xp
        with pytest.raises(ValueError):
            circular_interp(np.array([0.5]), np.array([1, 0, 2]), np.array([0, 1, 2]))

    def test_edge_cases(self):
        """Test edge cases and special values."""
        # Test with exactly 2 points (minimum)
        xp = np.array([0, 1])
        fp = np.array([0, np.pi])
        x = np.array([0.5])

        result = circular_interp(x, xp, fp)
        assert len(result) == 1
        assert 0 <= result[0] <= 2 * np.pi

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
        assert np.all(result >= 0)
        assert np.all(result <= 2 * np.pi)

    def test_full_circle_interpolation(self):
        """Test interpolation over a full circle."""
        # Test complete [0, 2π] cycle
        xp = np.array([0, 1, 2, 3, 4])
        fp = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
        x = np.array([0.5, 1.5, 2.5, 3.5])

        result = circular_interp(x, xp, fp)

        # Verify all results are in [0, 2π]
        assert np.all(result >= 0)
        assert np.all(result <= 2 * np.pi)

        # Check that interpolation is reasonable
        expected_approx = np.array(
            [np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4, 7 * np.pi / 4]
        )
        np.testing.assert_allclose(result, expected_approx, rtol=0.1)


class TestInterpMaxGap:
    def test_interp_max_gap_basic(self):
        """Basic behavior: values inside small gaps are interpolated, large gaps return fallback."""
        from neuro_py.util.array import interp_max_gap

        xp = np.array([0.0, 1.0, 3.0, 4.0])
        fp = np.array([0.0, 10.0, 30.0, 40.0])
        x = np.array([0.5, 1.5, 2.5, 3.5])

        # with max_gap = 1.5, gap between 1.0 and 3.0 is too large -> x in (1,3) masked
        out = interp_max_gap(x, xp, fp, max_gap=1.5, fallback=np.nan)
        # 0.5 -> interpolated between 0 and 1 -> 5.0
        assert np.isclose(out[0], 5.0)
        # 1.5 and 2.5 fall inside the large gap -> fallback
        assert np.isnan(out[1])
        assert np.isnan(out[2])
        # 3.5 -> interpolated between 3 and 4 -> 35.0
        assert np.isclose(out[3], 35.0)

    def test_interp_max_gap_no_gaps(self):
        """If all gaps are within max_gap, behaves like numpy.interp."""
        from neuro_py.util.array import interp_max_gap

        xp = np.array([0.0, 1.0, 2.0])
        fp = np.array([0.0, 10.0, 20.0])
        x = np.array([0.25, 0.75, 1.5])

        out = interp_max_gap(x, xp, fp, max_gap=2.0, fallback=-999)
        expected = np.interp(x, xp, fp, left=-999, right=-999)
        np.testing.assert_allclose(out, expected)

    def test_interp_max_gap_boundaries(self):
        """Points outside xp should receive the fallback value (left/right)."""
        from neuro_py.util.array import interp_max_gap

        xp = np.array([0.0, 1.0])
        fp = np.array([0.0, 10.0])
        x = np.array([-0.5, 0.5, 1.5])

        out = interp_max_gap(x, xp, fp, max_gap=1.0, fallback=-999)
        # left and right should be fallback
        assert out[0] == -999
        assert out[2] == -999
        # middle is interpolated
        assert np.isclose(out[1], 5.0)


def test_shrink_basic_block_mean():
    """Shrink should correctly average non-overlapping blocks."""
    mat = np.arange(16).reshape(4, 4)
    shrunk = shrink(mat, 2, 2)
    expected = np.array([[2.5, 4.5], [10.5, 12.5]])
    np.testing.assert_allclose(shrunk, expected)


def test_shrink_with_nans():
    """Shrink should ignore NaNs when averaging."""
    mat = np.array([[1, 2, np.nan, 4], [5, 6, 7, 8]])
    shrunk = shrink(mat, 1, 2)
    # Each block: [1,2], [nan,4], [5,6], [7,8]
    expected = np.array([[1.5, 4.0], [5.5, 7.5]])
    np.testing.assert_allclose(shrunk, expected, equal_nan=True)


def test_shrink_nondivisible_padding():
    """Shrink should pad with NaNs when shape not divisible by bin size, then average blocks."""
    mat = np.arange(10).reshape(2, 5)
    shrunk = shrink(mat, 1, 3)
    # Expected shape after padding: ceil(2/1)=2, ceil(5/3)=2 -> (2, 2)
    assert shrunk.shape == (2, 2)
    # The function uses nanmean, so NaNs are ignored in averaging
    # Verify the values are reasonable (not all zeros or all NaNs)
    assert not np.isnan(shrunk).all()
    assert np.all(
        shrunk >= 0
    )  # All values should be non-negative since input is arange(10)


def test_shrink_single_element_blocks():
    """Shrink(.,1,1) should return the matrix unchanged."""
    mat = np.random.rand(5, 7)
    shrunk = shrink(mat, 1, 1)
    np.testing.assert_allclose(mat, shrunk)


def test_zscore_columns_mean_std():
    """Each column should have mean 0 and std 1."""
    df = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [10, 20, 30, 40, 50],
        }
    )

    z = zscore_columns(df, ddof=0)

    # Means should be ~0
    assert np.allclose(z.mean(axis=0), 0.0)

    # Std should be ~1
    assert np.allclose(z.std(axis=0, ddof=0), 1.0)


def test_zscore_preserves_structure():
    """Index and columns should be preserved."""
    df = pd.DataFrame(
        np.random.randn(5, 3),
        index=["t1", "t2", "t3", "t4", "t5"],
        columns=["unit1", "unit2", "unit3"],
    )

    z = zscore_columns(df)

    assert list(z.index) == list(df.index)
    assert list(z.columns) == list(df.columns)
    assert z.shape == df.shape


def test_zscore_does_not_modify_input():
    """Original dataframe should not be modified."""
    df = pd.DataFrame(np.random.randn(5, 2))
    df_copy = df.copy()

    _ = zscore_columns(df)

    pd.testing.assert_frame_equal(df, df_copy)


def test_zscore_zero_variance_column():
    """Columns with zero variance should become NaN."""
    df = pd.DataFrame(
        {
            "constant": [1, 1, 1, 1],
            "varying": [1, 2, 3, 4],
        }
    )

    z = zscore_columns(df, ddof=0)

    # Constant column should be all NaN
    assert z["constant"].isna().all()
    # Varying column should have mean 0
    assert np.isclose(z["varying"].mean(), 0.0)


class TestSmoothPeth:
    def test_numpy_array_matches_gaussian_filter(self):
        time = np.linspace(-0.5, 0.5, 11)
        dt = time[1] - time[0]
        peth = np.arange(22, dtype=float).reshape(11, 2)

        result = smooth_peth(peth, smooth_window=0.2, smooth_std=0.1, dt=dt)

        expected = gaussian_filter1d(
            peth,
            sigma=0.1 / dt,
            axis=0,
            mode="nearest",
            truncate=(0.2 / dt) / (2 * (0.1 / dt)),
        )

        np.testing.assert_allclose(result, expected)
        assert result.shape == peth.shape

    def test_dataframe_preserves_structure(self):
        time = np.linspace(-0.5, 0.5, 11)
        dt = time[1] - time[0]
        peth = pd.DataFrame(
            np.arange(22, dtype=float).reshape(11, 2),
            index=time,
            columns=["unit1", "unit2"],
        )

        result = smooth_peth(peth, smooth_window=0.2, smooth_std=0.1)

        expected = gaussian_filter1d(
            peth.values,
            sigma=0.1 / dt,
            axis=0,
            mode="nearest",
            truncate=(0.2 / dt) / (2 * (0.1 / dt)),
        )

        assert isinstance(result, pd.DataFrame)
        assert result.index.equals(peth.index)
        assert result.columns.equals(peth.columns)
        np.testing.assert_allclose(result.values, expected)

    def test_one_dimensional_input_is_squeezed(self):
        time = np.linspace(-0.5, 0.5, 11)
        dt = time[1] - time[0]
        peth = np.arange(11, dtype=float)

        result = smooth_peth(peth, smooth_window=0.2, smooth_std=0.1, dt=dt)

        expected = gaussian_filter1d(
            peth[:, None],
            sigma=0.1 / dt,
            axis=0,
            mode="nearest",
            truncate=(0.2 / dt) / (2 * (0.1 / dt)),
        )[:, 0]

        assert result.shape == peth.shape
        np.testing.assert_allclose(result, expected)

    def test_numpy_array_requires_dt(self):
        peth = np.arange(11, dtype=float)

        with pytest.raises(ValueError, match="dt must be provided"):
            smooth_peth(peth, smooth_window=0.2, smooth_std=0.1)
