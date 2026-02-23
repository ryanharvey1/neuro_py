"""
Tests for neuro_py.process.correlations functions that use crossCorr.
Tests coverage for compute_AutoCorrs and pairwise_cross_corr.
"""

import numpy as np
import pandas as pd
import pytest

from neuro_py.process.correlations import compute_AutoCorrs, pairwise_cross_corr
from neuro_py.process.peri_event import crossCorr

# Pre-warm the JIT compiler with proper types to avoid typing issues
_prewarm_t1 = np.array([0.1, 0.2], dtype=np.float64)
_prewarm_t2 = np.array([0.15, 0.25], dtype=np.float64)
try:
    crossCorr(_prewarm_t1, _prewarm_t2, 0.01, 10)
except:
    pass


class TestComputeAutoCorrs:
    """Test suite for compute_AutoCorrs function."""

    def test_compute_autocorrs_returns_dataframe(self):
        """Test that compute_AutoCorrs returns a pandas DataFrame."""
        spks = np.array(
            [
                np.array([0.1, 0.2, 0.3], dtype=np.float64),
                np.array([0.15, 0.25, 0.35], dtype=np.float64),
            ],
            dtype=object,
        )
        result = compute_AutoCorrs(spks, binsize=0.01, nbins=50)

        assert isinstance(result, pd.DataFrame)

    def test_compute_autocorrs_output_shape(self):
        """Test that output has correct shape."""
        spks = np.array(
            [
                np.array([0.1, 0.2, 0.3], dtype=np.float64),
                np.array([0.15, 0.25, 0.35], dtype=np.float64),
            ],
            dtype=object,
        )
        nbins = 50
        result = compute_AutoCorrs(spks, binsize=0.01, nbins=nbins)

        # Output should have nbins+1 rows (adjusted to odd) and len(spks) columns
        assert result.shape[0] == 51  # 50 adjusted to odd = 51
        assert result.shape[1] == 2

    def test_compute_autocorrs_index_symmetric(self):
        """Test that index is symmetric around zero."""
        spks = np.array([np.array([0.1, 0.2, 0.3], dtype=np.float64)], dtype=object)
        binsize = 0.01
        result = compute_AutoCorrs(spks, binsize=binsize, nbins=50)

        # Check that index is centered around 0
        index = result.index.values
        assert np.isclose(index[len(index) // 2], 0.0, atol=1e-10)

    def test_compute_autocorrs_zero_lag_is_zero(self):
        """Test that at exactly zero lag, autocorrelation is set to 0."""
        spks = np.array(
            [np.array([0.1, 0.2, 0.3, 0.5], dtype=np.float64)], dtype=object
        )
        result = compute_AutoCorrs(spks, binsize=0.01, nbins=50)

        # Find row where index is closest to 0
        zero_idx = np.argmin(np.abs(result.index.values))
        assert result.iloc[zero_idx, 0] == 0.0

    def test_compute_autocorrs_empty_spike_trains(self):
        """Test handling of empty spike trains."""
        spks = np.array(
            [
                np.array([], dtype=np.float64),
                np.array([0.1, 0.2], dtype=np.float64),
                np.array([], dtype=np.float64),
            ],
            dtype=object,
        )
        result = compute_AutoCorrs(spks, binsize=0.01, nbins=50)

        # Should still output DataFrame with correct shape
        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == 3
        # Columns with empty spike trains should be NaN or 0
        assert result[0].isna().any() or (result[0] == 0).all()

    def test_compute_autocorrs_single_neuron(self):
        """Test with a single neuron."""
        spks = np.array(
            [np.array([0.1, 0.2, 0.3, 0.5, 0.7], dtype=np.float64)], dtype=object
        )
        result = compute_AutoCorrs(spks, binsize=0.01, nbins=50)

        assert result.shape == (51, 1)
        assert isinstance(result, pd.DataFrame)

    def test_compute_autocorrs_multiple_neurons(self):
        """Test with multiple neurons."""
        n_neurons = 5
        spks = np.array(
            [np.random.rand(20).astype(np.float64) for _ in range(n_neurons)],
            dtype=object,
        )
        result = compute_AutoCorrs(spks, binsize=0.001, nbins=100)

        assert result.shape[1] == n_neurons
        assert isinstance(result, pd.DataFrame)

    def test_compute_autocorrs_positive_values(self):
        """Test that autocorrelation values are non-negative."""
        spks = np.array(
            [np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)], dtype=object
        )
        result = compute_AutoCorrs(spks, binsize=0.01, nbins=50)

        # Autocorrelation should be non-negative (rates in Hz)
        assert (result >= 0).all().all() or result.isna().all().all()

    def test_compute_autocorrs_different_binsize(self):
        """Test with different bin sizes."""
        spks = np.array([np.array([0.1, 0.2, 0.3], dtype=np.float64)], dtype=object)

        result1 = compute_AutoCorrs(spks, binsize=0.001, nbins=100)
        result2 = compute_AutoCorrs(spks, binsize=0.01, nbins=100)

        # Different bin sizes should give different indices
        assert not np.allclose(result1.index.values, result2.index.values)

    def test_compute_autocorrs_column_indexing(self):
        """Test that columns are properly indexed by neuron number."""
        spks = np.array(
            [
                np.array([0.1, 0.2], dtype=np.float64),
                np.array([0.15, 0.25], dtype=np.float64),
                np.array([0.12, 0.22], dtype=np.float64),
            ],
            dtype=object,
        )
        result = compute_AutoCorrs(spks, binsize=0.01, nbins=50)

        # Columns should be indexed 0, 1, 2
        assert list(result.columns) == [0, 1, 2]


class TestPairwiseCrossCorr:
    """Test suite for pairwise_cross_corr function."""

    def test_pairwise_cross_corr_returns_dataframe(self):
        """Test that pairwise_cross_corr returns a DataFrame."""
        spks = np.array(
            [
                np.array([0.1, 0.2, 0.3], dtype=np.float64),
                np.array([0.15, 0.25, 0.35], dtype=np.float64),
            ],
            dtype=object,
        )
        result = pairwise_cross_corr(spks, binsize=0.01, nbins=50)

        assert isinstance(result, pd.DataFrame)

    def test_pairwise_cross_corr_output_shape_two_neurons(self):
        """Test output shape with two neurons (one pair)."""
        spks = np.array(
            [
                np.array([0.1, 0.2, 0.3], dtype=np.float64),
                np.array([0.15, 0.25, 0.35], dtype=np.float64),
            ],
            dtype=object,
        )
        nbins = 50
        result = pairwise_cross_corr(spks, binsize=0.01, nbins=nbins)

        # Should have nbins+1 (adjusted to odd) rows and 1 column (one pair)
        assert result.shape[0] == 51
        assert result.shape[1] == 1

    def test_pairwise_cross_corr_output_shape_three_neurons(self):
        """Test output shape with three neurons (three pairs)."""
        spks = np.array(
            [
                np.array([0.1, 0.2, 0.3], dtype=np.float64),
                np.array([0.15, 0.25, 0.35], dtype=np.float64),
                np.array([0.12, 0.22, 0.32], dtype=np.float64),
            ],
            dtype=object,
        )
        result = pairwise_cross_corr(spks, binsize=0.01, nbins=50)

        # Should have 3 pairs: (0,1), (0,2), (1,2)
        assert result.shape == (51, 3)

    def test_pairwise_cross_corr_with_return_index(self):
        """Test that return_index=True returns both crosscorrs and pairs."""
        spks = np.array(
            [
                np.array([0.1, 0.2, 0.3], dtype=np.float64),
                np.array([0.15, 0.25, 0.35], dtype=np.float64),
                np.array([0.12, 0.22, 0.32], dtype=np.float64),
            ],
            dtype=object,
        )
        result, pairs = pairwise_cross_corr(
            spks, binsize=0.01, nbins=50, return_index=True
        )

        assert isinstance(result, pd.DataFrame)
        assert isinstance(pairs, np.ndarray)
        assert result.shape[1] == len(pairs)
        # Check pairs are unique combinations
        np.testing.assert_array_equal(pairs, np.array([(0, 1), (0, 2), (1, 2)]))

    def test_pairwise_cross_corr_without_return_index(self):
        """Test that return_index=False returns only DataFrame."""
        spks = np.array(
            [
                np.array([0.1, 0.2, 0.3], dtype=np.float64),
                np.array([0.15, 0.25, 0.35], dtype=np.float64),
            ],
            dtype=object,
        )
        result = pairwise_cross_corr(spks, binsize=0.01, nbins=50, return_index=False)

        assert isinstance(result, pd.DataFrame)

    def test_pairwise_cross_corr_custom_pairs(self):
        """Test with custom pairs specification."""
        spks = np.array(
            [
                np.array([0.1, 0.2, 0.3], dtype=np.float64),
                np.array([0.15, 0.25, 0.35], dtype=np.float64),
                np.array([0.12, 0.22, 0.32], dtype=np.float64),
            ],
            dtype=object,
        )
        custom_pairs = np.array([(0, 1)])  # Only compute one pair
        result = pairwise_cross_corr(spks, binsize=0.01, nbins=50, pairs=custom_pairs)

        assert result.shape[1] == 1

    def test_pairwise_cross_corr_deconvolve_false(self):
        """Test with deconvolve=False (standard cross-correlation)."""
        spks = np.array(
            [
                np.array([0.1, 0.2, 0.3], dtype=np.float64),
                np.array([0.15, 0.25, 0.35], dtype=np.float64),
            ],
            dtype=object,
        )
        result = pairwise_cross_corr(spks, binsize=0.01, nbins=50, deconvolve=False)

        assert isinstance(result, pd.DataFrame)
        assert not result.isna().all().all()

    def test_pairwise_cross_corr_deconvolve_true(self):
        """Test with deconvolve=True."""
        spks = np.array(
            [
                np.array([0.1, 0.2, 0.3], dtype=np.float64),
                np.array([0.15, 0.25, 0.35], dtype=np.float64),
            ],
            dtype=object,
        )
        result = pairwise_cross_corr(spks, binsize=0.01, nbins=50, deconvolve=True)

        assert isinstance(result, pd.DataFrame)
        # Result should have shape (nbins+1, 1)
        assert result.shape[1] == 1

    def test_pairwise_cross_corr_symmetric_around_zero(self):
        """Test that time axis is symmetric around zero."""
        spks = np.array(
            [
                np.array([0.1, 0.2, 0.3], dtype=np.float64),
                np.array([0.15, 0.25, 0.35], dtype=np.float64),
            ],
            dtype=object,
        )
        result = pairwise_cross_corr(spks, binsize=0.01, nbins=50)

        index = result.index.values
        # Check that zero is in the middle
        zero_idx = np.argmin(np.abs(index))
        assert np.isclose(index[zero_idx], 0.0, atol=1e-10)

    def test_pairwise_cross_corr_positive_values(self):
        """Test that cross-correlogram values are non-negative."""
        spks = np.array(
            [
                np.array([0.1, 0.2, 0.3], dtype=np.float64),
                np.array([0.15, 0.25, 0.35], dtype=np.float64),
            ],
            dtype=object,
        )
        result = pairwise_cross_corr(spks, binsize=0.01, nbins=50)

        # Rates should be non-negative
        assert (result >= 0).all().all()

    def test_pairwise_cross_corr_many_pairs(self):
        """Test with many neurons and pairs."""
        n_neurons = 10
        spks = np.array(
            [np.random.rand(30).astype(np.float64) for _ in range(n_neurons)],
            dtype=object,
        )
        result = pairwise_cross_corr(spks, binsize=0.001, nbins=100)

        # Should have n_choose_2 pairs
        n_pairs = n_neurons * (n_neurons - 1) // 2
        assert result.shape[1] == n_pairs

    def test_pairwise_cross_corr_different_binsize(self):
        """Test with different bin sizes gives different time indices."""
        spks = np.array(
            [
                np.array([0.1, 0.2, 0.3], dtype=np.float64),
                np.array([0.15, 0.25, 0.35], dtype=np.float64),
            ],
            dtype=object,
        )

        result1 = pairwise_cross_corr(spks, binsize=0.001, nbins=100)
        result2 = pairwise_cross_corr(spks, binsize=0.01, nbins=100)

        # Different bin sizes should give different indices
        assert not np.allclose(result1.index.values, result2.index.values)

    def test_pairwise_cross_corr_reproducible(self):
        """Test that results are reproducible."""
        spks = np.array(
            [
                np.array([0.1, 0.2, 0.3], dtype=np.float64),
                np.array([0.15, 0.25, 0.35], dtype=np.float64),
            ],
            dtype=object,
        )

        result1 = pairwise_cross_corr(spks, binsize=0.01, nbins=50)
        result2 = pairwise_cross_corr(spks, binsize=0.01, nbins=50)

        pd.testing.assert_frame_equal(result1, result2)

    def test_pairwise_cross_corr_single_pair_same_as_single_crosscorr(self):
        """Test that single pair gives same result as direct crossCorr."""
        from neuro_py.process.peri_event import crossCorr

        t1 = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        t2 = np.array([0.15, 0.25, 0.35], dtype=np.float64)
        spks = np.array([t1, t2], dtype=object)

        pairwise_result = pairwise_cross_corr(spks, binsize=0.01, nbins=50)

        # Compute direct crossCorr for comparison
        direct_result = crossCorr(t1, t2, 0.01, 50)

        # Results should be very close (allowing for small numerical differences)
        np.testing.assert_allclose(
            pairwise_result.iloc[:, 0].values, direct_result, rtol=1e-10
        )
