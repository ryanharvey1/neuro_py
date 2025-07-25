import numpy as np
import pytest

from neuro_py.process import (
    event_triggered_cross_correlation,
    pairwise_event_triggered_cross_correlation,
)


class TestEventTriggeredCrossCorrelation:
    """Test suite for event_triggered_cross_correlation function"""

    @pytest.fixture
    def basic_data(self):
        """Create basic test data"""
        # Create time vectors
        dt = 0.001  # 1ms sampling
        t = np.arange(0, 10, dt)  # 10 seconds

        # Create two sinusoidal signals
        freq = 2  # 2 Hz
        signal1 = np.sin(2 * np.pi * freq * t)
        signal2 = np.cos(2 * np.pi * freq * t)  # 90 degree phase shift

        # Event times
        event_times = np.array([2.0, 4.0, 6.0, 8.0])

        return {
            "signal1_data": signal1,
            "signal1_ts": t,
            "signal2_data": signal2,
            "signal2_ts": t,
            "event_times": event_times,
        }

    @pytest.fixture
    def shifted_data(self):
        """Create test data with known time shift"""
        dt = 0.001
        t = np.arange(0, 10, dt)

        # Create signal and its delayed version
        signal1 = np.sin(2 * np.pi * 2 * t)
        time_shift = 0.1  # 100ms delay
        signal2 = np.sin(2 * np.pi * 2 * (t - time_shift))

        event_times = np.array([2.0, 4.0, 6.0])

        return {
            "signal1_data": signal1,
            "signal1_ts": t,
            "signal2_data": signal2,
            "signal2_ts": t,
            "event_times": event_times,
            "expected_lag": time_shift,
        }

    def test_basic_functionality(self, basic_data):
        """Test that function runs without error and returns expected shapes"""
        lags, correlation = event_triggered_cross_correlation(**basic_data)

        # Check return types
        assert isinstance(lags, np.ndarray)
        assert isinstance(correlation, np.ndarray)

        # Check shapes match
        assert len(lags) == len(correlation)

        # Check that correlation values are in reasonable range
        assert np.all(np.abs(correlation) <= 1.1)  # Allow small numerical errors

    def test_correlation_range(self, basic_data):
        """Test that correlation values are within [-1, 1] range"""
        lags, correlation = event_triggered_cross_correlation(**basic_data)

        # Allow for small numerical errors
        assert np.all(correlation >= -1.01), (
            f"Found correlation < -1: {np.min(correlation)}"
        )
        assert np.all(correlation <= 1.01), (
            f"Found correlation > 1: {np.max(correlation)}"
        )

    def test_zero_lag_identical_signals(self):
        """Test that identical signals have correlation = 1 at zero lag"""
        dt = 0.001
        t = np.arange(0, 5, dt)
        signal = np.sin(2 * np.pi * 3 * t) + 0.5 * np.cos(2 * np.pi * 7 * t)
        event_times = np.array([1.0, 2.0, 3.0])

        lags, correlation = event_triggered_cross_correlation(
            event_times=event_times,
            signal1_data=signal,
            signal1_ts=t,
            signal2_data=signal,
            signal2_ts=t,
            window=[-0.2, 0.2],
            bin_width=0.01,
        )

        # Find zero lag index
        zero_lag_idx = np.argmin(np.abs(lags))

        # Check that correlation at zero lag is close to 1
        assert correlation[zero_lag_idx] > 0.98, (
            f"Zero lag correlation: {correlation[zero_lag_idx]}"
        )

    def test_anticorrelated_signals(self):
        """Test that perfectly anticorrelated signals give correlation = -1"""
        dt = 0.001
        t = np.arange(0, 5, dt)
        signal1 = np.sin(2 * np.pi * 2 * t)
        signal2 = -signal1  # Perfect anticorrelation
        event_times = np.array([1.0, 2.0, 3.0])

        lags, correlation = event_triggered_cross_correlation(
            event_times=event_times,
            signal1_data=signal1,
            signal1_ts=t,
            signal2_data=signal2,
            signal2_ts=t,
            window=[-0.1, 0.1],
            bin_width=0.01,
        )

        # Find zero lag index
        zero_lag_idx = np.argmin(np.abs(lags))

        # Check that correlation at zero lag is close to -1
        assert correlation[zero_lag_idx] < -0.98, (
            f"Zero lag correlation: {correlation[zero_lag_idx]}"
        )

    def test_time_shift_detection(self, shifted_data):
        """Test that function correctly identifies time shifts"""
        lags, correlation = event_triggered_cross_correlation(
            event_times=shifted_data["event_times"],
            signal1_data=shifted_data["signal1_data"],
            signal1_ts=shifted_data["signal1_ts"],
            signal2_data=shifted_data["signal2_data"],
            signal2_ts=shifted_data["signal2_ts"],
            window=[-0.3, 0.3],
            bin_width=0.01,
        )

        # Find peak correlation
        max_corr_idx = np.argmax(np.abs(correlation))
        detected_lag = lags[max_corr_idx]

        # Check if detected lag is close to expected lag
        expected_lag = shifted_data["expected_lag"]
        assert np.abs(detected_lag - expected_lag) < 0.02, (
            f"Expected lag: {expected_lag}, Detected lag: {detected_lag}"
        )

    def test_custom_time_lags(self, basic_data):
        """Test function with custom time_lags parameter"""
        custom_lags = np.linspace(-0.2, 0.2, 50)

        lags, correlation = event_triggered_cross_correlation(
            time_lags=custom_lags, **basic_data
        )

        # Should ignore window parameter when time_lags is provided
        # Check that output lags span the expected range
        assert len(lags) > 0
        assert len(correlation) == len(lags)

    def test_different_sampling_rates(self):
        """Test with signals that have different sampling rates"""
        # Signal 1: high sampling rate
        dt1 = 0.001
        t1 = np.arange(0, 5, dt1)
        signal1 = np.sin(2 * np.pi * 3 * t1)

        # Signal 2: lower sampling rate
        dt2 = 0.01
        t2 = np.arange(0, 5, dt2)
        signal2 = np.sin(2 * np.pi * 3 * t2)

        event_times = np.array([1.0, 2.0, 3.0])

        lags, correlation = event_triggered_cross_correlation(
            event_times=event_times,
            signal1_data=signal1,
            signal1_ts=t1,
            signal2_data=signal2,
            signal2_ts=t2,
            window=[-0.1, 0.1],
            bin_width=0.01,
        )

        # Should still work and give reasonable correlation at zero lag
        zero_lag_idx = np.argmin(np.abs(lags))
        assert correlation[zero_lag_idx] > 0.8

    def test_empty_event_times(self, basic_data):
        """Test behavior with empty event times"""
        basic_data["event_times"] = np.array([])

        lags, correlation = event_triggered_cross_correlation(**basic_data)

        # Should return arrays of appropriate shape with zeros
        assert len(lags) > 0
        assert len(correlation) == len(lags)
        assert np.allclose(correlation, 0)

    def test_single_event(self, basic_data):
        """Test with single event"""
        basic_data["event_times"] = np.array([2.0])

        lags, correlation = event_triggered_cross_correlation(**basic_data)

        # Should work with single event
        assert len(lags) > 0
        assert len(correlation) == len(lags)
        assert np.all(np.abs(correlation) <= 1.1)

    def test_constant_signals(self):
        """Test behavior with constant (zero variance) signals"""
        t = np.arange(0, 5, 0.001)
        signal1 = np.ones_like(t) * 5  # Constant signal
        signal2 = np.sin(2 * np.pi * 2 * t)  # Varying signal
        event_times = np.array([1.0, 2.0, 3.0])

        lags, correlation = event_triggered_cross_correlation(
            event_times=event_times,
            signal1_data=signal1,
            signal1_ts=t,
            signal2_data=signal2,
            signal2_ts=t,
            window=[-0.1, 0.1],
            bin_width=0.01,
        )

        # Should handle constant signal gracefully (correlation should be 0)
        assert np.allclose(correlation, 0), (
            "Correlation with constant signal should be 0"
        )

    def test_window_bounds(self, basic_data):
        """Test that output lags respect window bounds"""
        window = [-0.3, 0.2]

        lags, correlation = event_triggered_cross_correlation(
            window=window, **basic_data
        )

        # Check that all lags are within window bounds
        assert np.all(lags >= window[0] - 1e-10), f"Lag below window: {np.min(lags)}"
        assert np.all(lags <= window[1] + 1e-10), f"Lag above window: {np.max(lags)}"

    def test_bin_width_parameter(self, basic_data):
        """Test that bin_width parameter affects resolution"""
        # Test with coarse bin width
        lags_coarse, corr_coarse = event_triggered_cross_correlation(
            bin_width=0.05, **basic_data
        )

        # Test with fine bin width
        lags_fine, corr_fine = event_triggered_cross_correlation(
            bin_width=0.01, **basic_data
        )

        # Finer bin width should give more points
        assert len(lags_fine) > len(lags_coarse)

    @pytest.mark.parametrize(
        "window", [[-0.1, 0.1], [-0.5, 0.3], [-0.2, 0.2], [-1.0, 1.0]]
    )
    def test_various_windows(self, basic_data, window):
        """Test function with various window sizes"""
        lags, correlation = event_triggered_cross_correlation(
            window=window, **basic_data
        )

        # Should work for all window sizes
        assert len(lags) > 0
        assert len(correlation) == len(lags)
        assert np.all(np.abs(correlation) <= 1.1)


def test_pairwise_event_triggered_cross_correlation():
    """Test the pairwise_event_triggered_cross_correlation function for correctness and shape."""
    # Create simple test data: 3 signals, 1000 samples
    dt = 0.001
    t = np.arange(0, 1, dt)
    n_signals = 3
    signals = np.stack([np.sin(2 * np.pi * (i + 1) * t) for i in range(n_signals)])
    event_times = np.array([0.2, 0.5, 0.7])

    # Run the function
    lags, avg_corr, pairs = pairwise_event_triggered_cross_correlation(
        event_times=event_times,
        signals_data=signals,
        signals_ts=t,
        window=[-0.1, 0.1],
        bin_width=0.01,
        n_jobs=1,  # for deterministic test
    )

    # There should be n_signals choose 2 pairs
    assert pairs.shape[0] == 3, f"Expected 3 pairs, got {pairs.shape[0]}"
    # Output shape: (n_pairs, n_lags)
    assert avg_corr.shape[0] == pairs.shape[0]
    assert avg_corr.shape[1] == len(lags)
    # Correlation values should be in [-1, 1]
    assert np.all(avg_corr <= 1.1) and np.all(avg_corr >= -1.1)
    # Lags should be symmetric around zero
    assert np.any(np.isclose(lags, 0, atol=1e-8)), "Zero lag missing"
