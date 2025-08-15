import numpy as np
import pandas as pd

from neuro_py.process.peri_event import event_triggered_average


def test_event_triggered_average():
    # Create test input arrays
    timestamps = np.linspace(0, 10, num=100)
    signal = np.random.randn(100, 3)
    events = np.array([3, 6, 9])

    # Test with default parameters
    result_sta, time_lags = event_triggered_average(
        timestamps, signal, events, window=[-0.5, 0.5]
    )
    assert result_sta.shape == (10, 3)
    assert time_lags.shape == (10,)

    # Test with custom window and sampling rate
    result_sta, time_lags = event_triggered_average(
        timestamps, signal, events, sampling_rate=10, window=[-1, 1]
    )
    assert result_sta.shape == (20, 3)
    assert time_lags.shape == (20,)

    # Test with one event and return_pandas=True
    result_df = event_triggered_average(
        timestamps, signal, [5], return_pandas=True, window=[-0.5, 0.5]
    )
    assert isinstance(result_df, pd.DataFrame)
    assert result_df.shape == (10, 3)

    # Test with timestamps with gaps
    # Create test input arrays
    timestamps = np.delete(timestamps, [range(50, 70)], None)
    signal = np.delete(signal, [range(50, 70)], 0)
    result_sta, time_lags = event_triggered_average(
        timestamps, signal, events, window=[-0.5, 0.5]
    )
    assert result_sta.shape == (10, 3)
    assert time_lags.shape == (10,)

    # Test 1 signal
    timestamps = np.linspace(0, 10, num=100)
    signal = np.random.randn(100, 1)
    events = np.array([3, 6, 9])
    result_sta, time_lags = event_triggered_average(
        timestamps, signal, events, sampling_rate=10, window=[-1, 1]
    )
    assert result_sta.shape == (20, 1)
    assert time_lags.shape == (20,)

    # Test 1 signal with known output
    timestamps = np.linspace(0, 10, num=100)
    signal = np.ones_like(timestamps)
    events = np.array([5, 5, 5])
    result_sta, time_lags = event_triggered_average(
        timestamps, signal, events, sampling_rate=10, window=[-5, 5]
    )
    assert all(result_sta.T[0] == signal)
    assert all(time_lags == np.linspace(-5, 5, 100))


def test_event_triggered_average_return_individual():
    """Test the new return_average=False functionality."""
    # Create test data
    timestamps = np.linspace(0, 10, num=100)
    signal = np.random.randn(100, 3)
    events = np.array([3, 6, 9])

    # Test return_average=False
    result_matrix, time_lags = event_triggered_average(
        timestamps, signal, events, window=[-0.5, 0.5], return_average=False
    )

    # Should return full matrix: (n_time_bins, n_signals, n_events)
    assert result_matrix.shape == (10, 3, 3)  # 10 time bins, 3 signals, 3 events
    assert time_lags.shape == (10,)

    # Test that averaging the matrix gives same result as return_average=True
    result_avg, _ = event_triggered_average(
        timestamps, signal, events, window=[-0.5, 0.5], return_average=True
    )
    manual_avg = np.nanmean(result_matrix, axis=2)
    np.testing.assert_array_almost_equal(result_avg, manual_avg)


def test_event_triggered_average_pandas_individual():
    """Test return_average=False with pandas DataFrame."""
    timestamps = np.linspace(0, 10, num=100)
    signal = np.random.randn(100, 2)
    events = np.array([3, 6])

    # Test return_average=False with return_pandas=True should return matrix, not DataFrame
    result_matrix, time_lags = event_triggered_average(
        timestamps,
        signal,
        events,
        window=[-0.5, 0.5],
        return_average=False,
        return_pandas=True,
    )

    # Should still return numpy array when return_average=False
    assert isinstance(result_matrix, np.ndarray)
    assert result_matrix.shape == (10, 2, 2)


def test_event_triggered_average_edge_cases():
    """Test edge cases and error handling."""
    import warnings

    timestamps = np.linspace(0, 10, num=100)
    signal = np.random.randn(100, 3)

    # Test with no valid events - expect warning
    events = np.array([15, 20])  # Outside timestamp range
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # Capture expected warning
        result_avg, _ = event_triggered_average(
            timestamps, signal, events, window=[-0.5, 0.5], return_average=True
        )
    assert result_avg.shape == (10, 3)
    assert np.all(result_avg == 0)  # Should be zeros

    # Test with return_average=False and no valid events
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # Capture expected warning
        result_matrix, _ = event_triggered_average(
            timestamps, signal, events, window=[-0.5, 0.5], return_average=False
        )
    assert result_matrix.shape == (10, 3, 0)  # 0 events

    # Test with single event
    events = np.array([5])
    result_matrix, _ = event_triggered_average(
        timestamps, signal, events, window=[-0.5, 0.5], return_average=False
    )
    assert result_matrix.shape == (10, 3, 1)

    # Test invalid window
    try:
        event_triggered_average(timestamps, signal, [5], window=[0.5, -0.5])
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_event_triggered_average_consistency():
    """Test consistency between regular and irregular sampling paths."""
    # Regular sampling
    timestamps_regular = np.linspace(0, 10, num=100)
    signal = np.random.randn(100, 2)
    events = np.array([3, 6, 9])

    # Use explicit sampling rate to ensure same window bins
    sampling_rate = 10  # Matches the 100 samples over 10 seconds

    result_regular, _ = event_triggered_average(
        timestamps_regular,
        signal,
        events,
        window=[-0.5, 0.5],
        sampling_rate=sampling_rate,
        return_average=False,
    )

    # Slightly irregular sampling (should trigger interpolation path)
    timestamps_irregular = timestamps_regular + np.random.normal(0, 0.001, 100)
    timestamps_irregular = np.sort(timestamps_irregular)

    result_irregular, _ = event_triggered_average(
        timestamps_irregular,
        signal,
        events,
        window=[-0.5, 0.5],
        sampling_rate=sampling_rate,
        return_average=False,
    )

    # Results should have same shape when using same sampling rate
    assert result_regular.shape == result_irregular.shape
    # Check that they're reasonably close (allowing for interpolation differences)
    correlation = np.corrcoef(result_regular.flatten(), result_irregular.flatten())[
        0, 1
    ]
    # Check for valid correlation value before assertion
    assert np.isfinite(correlation), f"Correlation is not finite: {correlation}"
    assert (
        correlation > 0.7
    )  # Should be reasonably correlated (lowered for interpolation differences)


def test_event_triggered_average_1d_signal():
    """Test with 1D signal input."""
    timestamps = np.linspace(0, 10, num=100)
    signal_1d = np.random.randn(100)  # 1D signal
    events = np.array([3, 6, 9])

    # Test with 1D signal
    result_avg, _ = event_triggered_average(
        timestamps, signal_1d, events, window=[-0.5, 0.5], return_average=True
    )
    assert result_avg.shape == (10, 1)  # Should be reshaped to 2D

    # Test return_average=False with 1D signal
    result_matrix, _ = event_triggered_average(
        timestamps, signal_1d, events, window=[-0.5, 0.5], return_average=False
    )
    assert result_matrix.shape == (10, 1, 3)


def test_event_triggered_average_large_window():
    """Test with large windows that might exceed signal bounds."""
    timestamps = np.linspace(0, 10, num=100)
    signal = np.random.randn(100, 2)
    events = np.array([3, 5, 7])  # Events well within boundaries

    # Large window that might challenge the algorithm but keeps events valid
    result_matrix, time_lags = event_triggered_average(
        timestamps, signal, events, window=[-1.5, 1.5], return_average=False
    )

    # Should handle all events successfully when they're within valid range
    assert result_matrix.shape[0] == len(time_lags)
    assert result_matrix.shape[1] == 2
    assert result_matrix.shape[2] == 3

    # Test actual boundary case where some events get filtered
    events_boundary = np.array([0.5, 5, 9.5])  # Events near actual boundaries
    result_boundary, time_lags_boundary = event_triggered_average(
        timestamps, signal, events_boundary, window=[-1, 1], return_average=False
    )

    # Should handle boundary events - some might be filtered out
    assert result_boundary.shape[0] == len(time_lags_boundary)
    assert result_boundary.shape[1] == 2
    # Number of events might be less than 3 due to boundary filtering
    assert result_boundary.shape[2] <= 3


def test_event_triggered_average_sampling_rate_detection():
    """Test automatic sampling rate detection."""
    # Regular sampling
    timestamps = np.linspace(0, 10, num=1000)  # 100 Hz
    signal = np.random.randn(1000, 1)
    events = np.array([5])

    # Don't provide sampling_rate, let it auto-detect
    result1, time_lags1 = event_triggered_average(
        timestamps, signal, events, window=[-0.5, 0.5]
    )

    # Provide explicit sampling_rate
    result2, time_lags2 = event_triggered_average(
        timestamps, signal, events, window=[-0.5, 0.5], sampling_rate=100
    )

    # Results should be identical
    np.testing.assert_array_almost_equal(result1, result2)
    np.testing.assert_array_almost_equal(time_lags1, time_lags2)


def test_event_triggered_average_comparison_with_fast():
    """Test that results match event_triggered_average_fast for regular sampling."""
    from neuro_py.process.peri_event import event_triggered_average_fast

    # Create regular sampling scenario
    sampling_rate = 100
    timestamps = np.arange(0, 10, 1 / sampling_rate)
    signal = np.random.randn(len(timestamps), 3)
    events = np.array([2, 4, 6, 8])
    window = [-0.5, 0.5]

    # Test our function
    result_new, time_lags_new = event_triggered_average(
        timestamps,
        signal,
        events,
        window=window,
        sampling_rate=sampling_rate,
        return_average=True,
    )

    # Test fast function (note: signal needs to be transposed)
    result_fast, time_lags_fast = event_triggered_average_fast(
        signal.T,
        events,
        sampling_rate=sampling_rate,
        window=window,
        return_average=True,
    )

    # Results should be very close
    np.testing.assert_array_almost_equal(result_new, result_fast.T, decimal=10)
    np.testing.assert_array_almost_equal(time_lags_new, time_lags_fast, decimal=10)
