import numpy as np
import pytest

from neuro_py.lfp.spectral import compute_wavelet_transform, event_triggered_wavelet


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("event", [0.5, 0.505])
def test_sample_grid_and_off_grid_events_match_direct_transform(parallel, event):
    timestamps = np.arange(201) / 100
    values = np.sin(2 * np.pi * 10 * timestamps)
    power, raw, lags, frequencies = event_triggered_wavelet(
        values,
        timestamps,
        np.array([event]),
        max_lag=0.2,
        freq_min=10,
        freq_max=20,
        freq_step=10,
        fs=100,
        whiten=False,
        parallel=parallel,
    )
    np.testing.assert_allclose(lags, -0.2 + np.arange(40) / 100)
    expected_segment = np.interp(event + lags, timestamps, values)
    expected = abs(compute_wavelet_transform(expected_segment, 100, frequencies))
    np.testing.assert_allclose(raw, expected_segment, atol=1e-12)
    np.testing.assert_allclose(power, expected, atol=1e-12)
    assert np.isfinite(power).all()


def test_serial_parallel_averages_ignore_the_same_invalid_events():
    timestamps = np.arange(201) / 100
    values = 3 + 2 * timestamps
    options = dict(
        max_lag=0.2, freq_min=10, freq_max=20, freq_step=10, fs=100, whiten=False
    )
    expected = event_triggered_wavelet(
        values, timestamps, np.array([0.5, 1.0]), parallel=False, **options
    )
    for parallel in (False, True):
        result = event_triggered_wavelet(
            values,
            timestamps,
            np.array([-1, 0.5, 1.0, 10, np.nan]),
            parallel=parallel,
            **options,
        )
        for actual, reference in zip(result, expected):
            np.testing.assert_allclose(actual, reference)


@pytest.mark.parametrize("parallel", [False, True])
def test_no_valid_events_returns_documented_missing_arrays(parallel):
    timestamps = np.arange(101) / 100
    with pytest.warns(UserWarning, match="No events"):
        power, raw, lags, frequencies = event_triggered_wavelet(
            np.ones(101),
            timestamps,
            np.array([-1, 10]),
            max_lag=0.2,
            freq_min=10,
            freq_max=20,
            freq_step=10,
            fs=100,
            whiten=False,
            parallel=parallel,
        )
    assert power.shape == (len(frequencies), len(lags))
    assert np.isnan(power).all() and np.isnan(raw).all()


def test_irregular_timestamps_and_nonzero_origin_use_event_relative_grid():
    timestamps = 100 + np.arange(201) / 100
    timestamps[1:-1] += 0.001 * np.sin(np.arange(199))
    values = 3 + 2 * (timestamps - 100)
    frame, raw = event_triggered_wavelet(
        values,
        timestamps,
        np.array([100.5]),
        max_lag=0.2,
        freq_min=10,
        freq_max=20,
        freq_step=10,
        fs=100,
        whiten=False,
        return_pandas=True,
    )
    np.testing.assert_allclose(raw, 3 + 2 * (0.5 + frame.index), atol=1e-10)
    assert frame.shape == (40, 1)


@pytest.mark.parametrize("origin", [0, 10000, 1000000])
@pytest.mark.parametrize("sample_count", [12500, 12501])
@pytest.mark.parametrize("max_lag, count", [(0.5, 1250), (0.50004, 1251)])
def test_inferred_regular_rate_keeps_lag_count_independent_of_clock_origin(
    origin, max_lag, count, sample_count
):
    times = origin + np.arange(sample_count) / 1250
    values = np.sin(2 * np.pi * 10 * np.arange(sample_count) / 1250)
    power, raw, lags, frequencies = event_triggered_wavelet(
        values,
        times,
        np.array([origin + 5.0001]),
        max_lag=max_lag,
        freq_min=10,
        freq_max=20,
        freq_step=10,
        whiten=False,
        parallel=False,
    )
    assert power.shape == (len(frequencies), count)
    assert raw.shape == (count,)
    # An inferred rate cannot recover time precision lost at the clock origin.
    np.testing.assert_allclose(
        lags,
        -max_lag + np.arange(count) / 1250,
        atol=max(1e-12, np.spacing(float(origin + 10))),
    )
    np.testing.assert_allclose(raw, np.interp(origin + 5.0001 + lags, times, values))
