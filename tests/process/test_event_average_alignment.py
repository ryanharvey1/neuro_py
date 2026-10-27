import numpy as np
import pytest

from neuro_py.process.peri_event import (
    event_triggered_average,
    event_triggered_average_fast,
)


@pytest.mark.parametrize("fast", [False, True])
@pytest.mark.parametrize("window", [[0, 0.4], [-0.4, 0], [-0.15, 0.25], [-0.1, 0.2]])
@pytest.mark.parametrize("event", [2.0, 2.035])
def test_event_windows_match_exact_lags_on_linear_signals(fast, window, event):
    timestamps = np.arange(51) / 10
    signal = np.column_stack((timestamps, 3 + 2 * timestamps))
    original = signal.copy()
    if fast:
        response, lags = event_triggered_average_fast(
            signal.T, np.array([event]), 10, window
        )
        response = response.T
    else:
        response, lags = event_triggered_average(
            timestamps, signal, [event], 10, window
        )
    count = int(np.ceil(np.round((window[1] - window[0]) * 10, 12)))
    np.testing.assert_allclose(lags, window[0] + np.arange(count) / 10, atol=1e-12)
    np.testing.assert_allclose(response[:, 0], event + lags, atol=1e-12)
    np.testing.assert_allclose(response[:, 1], 3 + 2 * (event + lags), atol=1e-12)
    np.testing.assert_array_equal(signal, original)


@pytest.mark.parametrize("dtype", [np.int16, np.int64, np.float32, np.float64])
def test_invalid_events_stay_missing_for_every_input_dtype(dtype):
    signal = np.full((2, 51), 7, dtype=dtype)
    events = np.array([0, 2, 20, np.nan])
    individual, lags = event_triggered_average_fast(
        signal, events, 10, return_average=False
    )
    assert np.issubdtype(individual.dtype, np.floating)
    assert individual.shape == (2, len(lags), 4)
    assert np.isnan(individual[:, :, [0, 2, 3]]).all()
    np.testing.assert_array_equal(individual[:, :, 1], 7)
    average, _ = event_triggered_average_fast(signal, events, 10)
    np.testing.assert_array_equal(average, 7)


@pytest.mark.parametrize("irregular", [False, True])
def test_nonzero_origin_and_irregular_sampling_use_actual_timestamps(irregular):
    timestamps = 1000 + np.arange(100) / 10
    if irregular:
        timestamps[1:-1] += 0.02 * np.sin(np.arange(98))
    signal = 2 * (timestamps - 1000) + 3
    result, lags = event_triggered_average(
        timestamps, signal, [1005.035], 10, [0, 0.4], irregular_sampling=irregular
    )
    np.testing.assert_allclose(result[:, 0], 2 * (5.035 + lags) + 3, atol=1e-10)


@pytest.mark.parametrize(
    "event, window, expected", [(0, [0, 0.2], [0, 1]), (5, [-0.2, 0], [48, 49])]
)
def test_half_open_windows_accept_boundary_samples(event, window, expected):
    signal = np.arange(51)[None, :]
    result, _ = event_triggered_average_fast(signal, np.array([event]), 10, window)
    np.testing.assert_array_equal(result[0], expected)


@pytest.mark.parametrize("fast", [False, True])
def test_pandas_output_and_empty_events_preserve_contract(fast):
    import pandas as pd

    times = np.arange(51) / 10
    if fast:
        frame = event_triggered_average_fast(
            times[None, :], np.array([2]), 10, return_pandas=True
        )
        individual, lags = event_triggered_average_fast(
            times[None, :], np.array([]), 10, return_average=False
        )
    else:
        frame = event_triggered_average(times, times, [2], 10, return_pandas=True)
        with pytest.warns(UserWarning, match="No events"):
            individual, lags = event_triggered_average(
                times, times, [], 10, return_average=False
            )
    assert isinstance(frame, pd.DataFrame)
    assert individual.size == 0
    assert 0 in frame.index
    np.testing.assert_allclose(np.diff(frame.index), 0.1)


@pytest.mark.parametrize("window", [[0, 0], [1, -1], [np.nan, 1], [0, np.inf]])
def test_invalid_windows_are_rejected(window):
    with pytest.raises(ValueError, match="window"):
        event_triggered_average_fast(np.ones((1, 10)), np.array([0.5]), 10, window)


@pytest.mark.parametrize("rate", [0, -1, np.inf, np.nan])
def test_invalid_sample_rates_are_rejected(rate):
    with pytest.raises(ValueError, match="sampling_rate"):
        event_triggered_average_fast(np.ones((1, 10)), np.array([0.5]), rate)


@pytest.mark.parametrize("budget", [1, 12, 500_000])
def test_streaming_means_match_individual_windows_with_missing_samples(
    monkeypatch, budget
):
    import neuro_py.process.peri_event as module

    monkeypatch.setattr(module, "_EVENT_MAX_CHUNK_VALUES", budget)
    times = np.arange(101) / 10
    signal = np.column_stack((times, 2 * times))
    signal[11, 0] = np.nan
    events = np.array([1.0, 1.05, 2, 3.025, -1, np.nan])
    individual, lags = event_triggered_average_fast(
        signal.T, events, 10, [0, 0.4], return_average=False
    )
    expected = np.nanmean(individual, axis=2)
    average, _ = event_triggered_average_fast(signal.T, events, 10, [0, 0.4])
    np.testing.assert_allclose(average, expected, atol=1e-12)
    # Exact-grid finite sample is retained even next to a missing sample.
    assert individual[0, 0, 0] == 1.0
    general, _ = event_triggered_average(times, signal, events, 10, [0, 0.4])
    np.testing.assert_allclose(general.T, expected, atol=1e-12)


def test_average_only_path_does_not_allocate_full_event_cube(monkeypatch):
    import neuro_py.process.peri_event as module

    original = module.np.full
    shapes = []

    def record(shape, *args, **kwargs):
        shapes.append(shape)
        return original(shape, *args, **kwargs)

    monkeypatch.setattr(module.np, "full", record)
    result, _ = event_triggered_average_fast(
        np.ones((8, 10_000)), np.linspace(1, 8, 1000), 1000, [-0.1, 0.1]
    )
    np.testing.assert_array_equal(result, 1)
    assert all(len(shape) <= 2 for shape in shapes)


@pytest.mark.parametrize("fortran", [False, True])
def test_batched_extraction_matches_direct_interpolation(fortran):
    rng = np.random.default_rng(4)
    times = np.arange(1000) / 100
    signal = rng.normal(size=(1000, 5))
    if fortran:
        signal = np.asfortranarray(signal)
    events = np.array([2, 3.005, 4.003, 7])
    actual, lags = event_triggered_average_fast(
        signal.T, events, 100, [-0.13, 0.27], return_average=False
    )
    expected = np.stack(
        [
            np.stack([np.interp(event + lags, times, col) for col in signal.T])
            for event in events
        ],
        axis=2,
    )
    np.testing.assert_allclose(actual, expected, atol=1e-12)
