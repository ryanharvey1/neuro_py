import numpy as np
import pandas as pd
from neuro_py.process.peri_event import event_triggered_average

def test_event_triggered_average():
    # Create test input arrays
    timestamps = np.linspace(0, 10, num=100)
    signal = np.random.randn(100, 3)
    events = np.array([3, 6, 9])

    # Test with default parameters
    result_sta, time_lags = event_triggered_average(timestamps, signal, events, window=[-0.5, 0.5])
    assert result_sta.shape == (10, 3)
    assert time_lags.shape == (10,)

    # Test with custom window and sampling rate
    result_sta, time_lags = event_triggered_average(timestamps, signal, events, sampling_rate=10, window=[-1, 1])
    assert result_sta.shape == (20, 3)
    assert time_lags.shape == (20,)

    # Test with one event and return_pandas=True
    result_df = event_triggered_average(timestamps, signal, [5], return_pandas=True, window=[-0.5, 0.5])
    assert isinstance(result_df, pd.DataFrame)
    assert result_df.shape == (10, 3)

    # Test with timestamps with gaps
    # Create test input arrays
    timestamps = np.delete(timestamps, [range(50,70)], None)
    signal = np.delete(signal, [range(50,70)], 0)
    result_sta, time_lags = event_triggered_average(timestamps, signal, events, window=[-0.5, 0.5])
    assert result_sta.shape == (10, 3)
    assert time_lags.shape == (10,)

    # Test 1 signal
    timestamps = np.linspace(0, 10, num=100)
    signal = np.random.randn(100, 1)
    events = np.array([3, 6, 9])
    result_sta, time_lags = event_triggered_average(timestamps, signal, events, sampling_rate=10, window=[-1, 1])
    assert result_sta.shape == (20, 1)
    assert time_lags.shape == (20,)

    # Test 1 signal with known output
    timestamps = np.linspace(0, 10, num=100)
    signal = np.ones_like(timestamps)
    events = np.array([5, 5, 5])
    result_sta, time_lags = event_triggered_average(timestamps, signal, events, sampling_rate=10, window=[-5, 5])
    assert all(result_sta.T[0] == signal)
    assert all(time_lags == np.linspace(-5,5,100))
