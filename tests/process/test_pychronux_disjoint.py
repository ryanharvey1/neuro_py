import nelpy as nel
import numpy as np
import pandas as pd
import pytest

from neuro_py.process import pychronux as px


def test_running_power_matches_independent_windows_and_reuses_tapers(
    monkeypatch,
) -> None:
    fs = 128
    band = [1, 40]
    starts = [0, 2, 100_000]
    # Overlap, touching intervals, and short or leftover stretches.
    support = np.array([[2, 5], [100_000, 100_002], [1, 3], [0, 1], [800, 801]])
    units = [
        np.array([100_001.5, 2, 0.5, 0, 1.25, 4, 800.5, 100_002]),
        np.array([0.1, 2.2, 2.8, 99_999.2]),
        np.array([]),
    ]
    estimates = [
        px.mtspectrumpt(
            [unit[(unit >= t) & (unit < t + 2)] - t for unit in units],
            fs,
            band,
            time_support=[0, 2],
        )
        for t in starts
    ]
    expected = sum(estimates) / len(estimates)

    calls = []
    dpss = px.dpss

    def record_taper_length(length, *args, **kwargs):
        calls.append(length)
        return dpss(length, *args, **kwargs)

    monkeypatch.setattr(px, "dpss", record_taper_length)
    actual = px.mtspectrumpt(units, fs, band, time_support=support, window_duration=2)
    pd.testing.assert_frame_equal(actual, expected, rtol=1e-12, atol=1e-14)
    assert calls == [2 * fs + 3]  # once, independent of unit count and wall-clock span

    power, frequencies = px.point_spectra(
        units[0], fs, band, [2.5, 4], time_support=support, window_duration=2
    )
    np.testing.assert_array_equal(frequencies, actual.index)
    np.testing.assert_allclose(power, actual[0], atol=1e-14)
    gram = px.mtspecgrampt(units[1], fs, band, time_support=support, movingwin=(2, 2))
    np.testing.assert_array_equal(gram.columns, [1, 3, 100_001])
    np.testing.assert_array_equal(gram.iloc[:, -1], 0)
    np.testing.assert_allclose(gram.mean(axis=1), actual[1], atol=1e-14)


def test_native_spike_train_uses_the_same_support_as_explicit_call() -> None:
    base = nel.EpochArray([[0, 30]])
    running = nel.EpochArray([[1, 5], [12, 16]])
    st = nel.SpikeTrainArray(
        timestamps=[[1, 1.5, 6, 13, 16], [2, 2.5, 12, 12.5]],
        fs=30_000,
        support=base,
    )
    restricted = st[base & running]
    options = dict(Fs=128, fpass=[1, 40])
    explicit = px.mtspectrumpt(restricted.data, time_support=running.data, **options)
    native = px.mtspectrumpt(restricted, **options)
    pd.testing.assert_frame_equal(native, explicit)

    single_bout = running[0]
    pd.testing.assert_frame_equal(
        px.mtspectrumpt(st[single_bout], **options),
        px.mtspectrumpt(st[single_bout].data, time_support=single_bout.data, **options),
    )
    silent = st[nel.EpochArray([[17, 21], [24, 28]])]
    silent_power = px.mtspectrumpt(silent, **options)
    assert silent_power.shape == explicit.shape
    np.testing.assert_array_equal(silent_power, 0)
    empty_array_power = px.mtspectrumpt(
        np.array([]), time_support=running.data, **options
    )
    assert empty_array_power.shape == (len(explicit), 1)
    np.testing.assert_array_equal(empty_array_power, 0)


def test_point_pair_pools_complex_cross_and_both_auto_spectra() -> None:
    fs = 128
    band = [1, 20]
    support = np.array([[0, 2], [100, 102], [200, 202]])
    t = np.arange(0.1, 1.95, 0.125)
    first = np.r_[t, 100 + t, 200 + t]
    second = np.r_[t + 0.02, 100 + t - 0.02, 30.5]
    cross, first_power, second_power, window_coherence = [], [], [], []
    for start, stop in support:
        x = first[(first >= start) & (first < stop)] - start
        y = second[(second >= start) & (second < stop)] - start
        cross.append(
            px.mtcsdpt(x, y, fs, band, time_support=[0, 2], return_complex=True)["CSD"]
        )
        first_power.append(px.mtspectrumpt([x], fs, band, time_support=[0, 2])[0])
        second_power.append(px.mtspectrumpt([y], fs, band, time_support=[0, 2])[0])
        window_coherence.append(
            px.mtcoherencept(x, y, fs, band, time_support=[0, 2])["Coherence"]
        )
    expected_cross = np.mean(cross, axis=0)
    expected_coherence = np.abs(expected_cross) ** 2 / (
        np.mean(first_power, axis=0) * np.mean(second_power, axis=0)
    )

    opts = dict(time_support=support, window_duration=2)
    actual_cross = px.mtcsdpt(first, second, fs, band, return_complex=True, **opts)
    actual_real = px.mtcsdpt(first, second, fs, band, **opts)
    actual_coherence = px.mtcoherencept(first, second, fs, band, **opts)
    np.testing.assert_allclose(actual_cross["CSD"], expected_cross, atol=1e-13)
    np.testing.assert_allclose(actual_real["CSD"], expected_cross.real, atol=1e-13)
    np.testing.assert_allclose(
        actual_coherence["Coherence"], expected_coherence, atol=1e-13
    )
    assert not np.allclose(
        actual_coherence["Coherence"], np.nanmean(window_coherence, axis=0)
    )


def test_continuous_estimators_ignore_gap_samples_and_pool_spectra() -> None:
    fs = 128
    band = [1, 40]
    tapers = (2.5, 4)
    support = np.array([[0, 2], [5, 7]])
    t = np.arange(2 * fs) / fs
    signal = np.sin(2 * np.pi * 8 * t)
    other_signal = np.cos(2 * np.pi * 8 * t)
    missing = np.full(3 * fs, np.nan)
    first = np.r_[signal, missing, signal]
    second = np.r_[other_signal, missing, -other_signal]
    expected_power = px.mtspectrumc(signal, fs, band, tapers)
    expected_coherence = px.mtcoherencec(signal, other_signal, fs, band, tapers)
    opts = dict(time_support=support, window_duration=2)
    pd.testing.assert_series_equal(
        px.mtspectrumc(first, fs, band, tapers, **opts), expected_power
    )
    pooled = px.mtcoherencec(first, second, fs, band, tapers, **opts)
    np.testing.assert_allclose(pooled, 0, atol=1e-13)  # opposite cross-spectral phases

    opts = dict(time_support=support, movingwin=(2, 1))
    spec = px.mtspecgramc(first, fs, band, tapers, **opts)
    coh = px.mtcohgramc(first, second, fs, band, tapers, **opts)
    np.testing.assert_array_equal(spec.columns, [1, 6])
    np.testing.assert_array_equal(coh.columns, spec.columns)
    np.testing.assert_allclose(spec[1], expected_power)
    np.testing.assert_allclose(spec[6], expected_power)
    np.testing.assert_allclose(coh[1], expected_coherence)
    np.testing.assert_allclose(coh[6], expected_coherence)


def test_continuous_fractional_bounds_use_original_sample_grid() -> None:
    fs = 100
    signal = np.random.default_rng(20).normal(size=500)
    opts = (fs, [1, 40], (2.5, 4))
    support = [[0.105, 1.115], [3.205, 4.215]]
    gram = px.mtspecgramc(signal, *opts, movingwin=(1, 1), time_support=support)
    np.testing.assert_allclose(gram.columns, [0.61, 3.71])
    for i, first_sample in enumerate([11, 321]):
        expected = px.mtspectrumc(signal[first_sample : first_sample + 100], *opts)
        np.testing.assert_allclose(gram.iloc[:, i], expected)
    pd.testing.assert_series_equal(
        px.mtspectrumc(signal, *opts, time_support=support[0]),
        px.mtspectrumc(signal[11:112], *opts),
    )


@pytest.mark.parametrize("support", [[], [[0, np.nan]], [[2, 1]], [[0, 1, 2]]])
def test_invalid_observation_support_is_rejected(support) -> None:
    with pytest.raises(ValueError, match="time_support"):
        px.mtspectrumpt(np.array([0.1]), 128, [1, 40], time_support=support)


def test_window_and_support_validation() -> None:
    spans = [[0, 1], [2, 3]]
    x = np.array([0.1, 0.2])
    with pytest.raises(ValueError, match="No complete analysis windows"):
        px.mtspectrumpt(x, 128, [1, 40], time_support=spans)
    with pytest.raises(ValueError, match="tapers_ts is not supported"):
        px.mtspectrumpt(x, 128, [1, 40], time_support=spans, tapers_ts=np.arange(10))
    with pytest.raises(ValueError, match="must lie within"):
        px.mtspecgramc(
            np.ones(256), 128, [1, 40], (2, 3), movingwin=(1, 1), time_support=spans
        )


def test_pooled_spectra_warn_when_most_observation_time_is_excluded() -> None:
    support = [[0, 4], [5, 8], [10, 13]]
    with pytest.warns(
        UserWarning, match="1 complete analysis windows cover only 40.0%"
    ):
        px.mtspectrumpt(np.array([0.1, 0.2]), 128, [1, 40], time_support=support)
    with pytest.warns(UserWarning, match="40.0% of time_support"):
        px.mtspectrumc(np.ones(13 * 128), 128, [1, 40], (2, 3), time_support=support)
