import nelpy as nel
import numpy as np
import pandas as pd
import pytest

from neuro_py.process import pychronux as px


def _unit(times: np.ndarray) -> nel.SpikeTrainArray:
    return nel.SpikeTrainArray(
        timestamps=times, fs=30_000, support=nel.EpochArray([[0, 40]])
    )


def test_single_unit_methods_use_native_support_on_the_recording_clock() -> None:
    fs = 128
    freq = [1, 40]
    x = np.arange(0.07, 40, 0.125)
    bouts = nel.EpochArray([[4, 8], [24, 28]])
    unit = _unit(x)[bouts]
    expected_gram = px.mtspecgrampt(
        x, fs, freq, movingwin=(2, 2), time_support=bouts.data
    )
    gram = px.mtspecgrampt(unit, fs, freq, movingwin=(2, 2))
    pd.testing.assert_frame_equal(gram, expected_gram)
    np.testing.assert_array_equal(gram.columns, [5, 7, 25, 27])

    expected_power, expected_f = px.point_spectra(x, fs, freq, time_support=bouts.data)
    power, f = px.point_spectra(unit, fs, freq)
    np.testing.assert_allclose(power, expected_power)
    np.testing.assert_array_equal(f, expected_f)

    # Explicit broad support cannot make the gap between native bouts observed.
    anchor = [1, 32]
    native_power = px.mtspectrumpt(unit, fs, freq, time_support=anchor)
    expected_native_power = px.mtspectrumpt(unit.data, fs, freq, time_support=bouts)
    pd.testing.assert_frame_equal(native_power, expected_native_power)
    actual_lag, actual_exposure = px.point_lag_spectrum(
        unit, fs, [4, 12], time_support=anchor
    )
    expected_lag, expected_exposure = px.point_lag_spectrum(
        x, fs, [4, 12], time_support=bouts
    )
    pd.testing.assert_frame_equal(actual_lag, expected_lag)
    pd.testing.assert_frame_equal(actual_exposure, expected_exposure)


@pytest.mark.parametrize("method", [px.mtcsdpt, px.mtcoherencept])
def test_point_pairs_intersect_native_support_and_allow_numpy_input(method) -> None:
    x = np.arange(0.06, 40, 0.125)
    y = x + 0.02
    first = _unit(x)[nel.EpochArray([[4, 10], [24, 30]])]
    second = _unit(y)[nel.EpochArray([[5, 15], [20, 29]])]
    request = nel.EpochArray([[2, 7], [25, 35]])
    args = (128, [1, 40])
    expected = method(x, y, *args, time_support=[[5, 7], [25, 29]], window_duration=2)
    actual = method(first, second, *args, time_support=request, window_duration=2)
    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(
        method(first, y, *args, window_duration=2),
        method(x, y, *args, time_support=first.support.data, window_duration=2),
    )
    with pytest.raises(ValueError, match="No shared observation time"):
        method(first, second, *args, time_support=[11, 14], window_duration=2)
    span = [5, 7]
    # A raw partner's spike exactly at the excluded stop isn't observed.
    pd.testing.assert_frame_equal(
        method(first, np.r_[y, 7], *args, time_support=span),
        method(x, y[y < 7], *args, time_support=span),
    )


def _lfp(
    signal: np.ndarray, time: np.ndarray, fs: float = 128
) -> nel.AnalogSignalArray:
    return nel.AnalogSignalArray(signal, fs=fs, abscissa_vals=time)


def test_contiguous_native_lfp_matches_numpy_and_preserves_absolute_gram_times() -> (
    None
):
    fs, t0 = 128, 30.5
    power_opts = dict(fpass=[1, 40], tapers=(3, 5))
    # Native metadata is used even when the recording doesn't start at zero.
    t = np.arange(512) / fs
    x = np.sin(2 * np.pi * 8 * t)
    native = _lfp(x, t0 + t)
    pd.testing.assert_series_equal(
        px.mtspectrumc(native, **power_opts), px.mtspectrumc(x, fs, **power_opts)
    )
    native_gram = px.mtspecgramc(native, **power_opts, movingwin=(1, 0.5))
    expected = px.mtspecgramc(x, fs, **power_opts, movingwin=(1, 0.5))
    expected.columns += t0
    pd.testing.assert_frame_equal(native_gram, expected)
    with pytest.raises(ValueError, match="sampling rate"):
        px.mtspectrumc(native, Fs=100, **power_opts)
    with pytest.raises(ValueError, match="Fs is required"):
        px.mtspectrumc(x, **power_opts)


def test_native_lfp_drops_gaps_even_with_a_broad_explicit_interval() -> None:
    fs, t0 = 128, 100
    samples = np.arange(10 * fs) / fs
    x = np.sin(2 * np.pi * 7 * samples)
    native = _lfp(x, samples + t0)[nel.EpochArray([[101, 103], [106, 108]])]
    band = [1, 40]
    bank = (2.5, 4)
    expected = px.mtspectrumc(
        x, fs, band, bank, time_support=[[1, 3], [6, 8]], window_duration=2
    )
    actual = px.mtspectrumc(
        native, fpass=band, tapers=bank, time_support=[100, 110], window_duration=2
    )
    pd.testing.assert_series_equal(actual, expected)
    gram = px.mtspecgramc(native, fpass=band, tapers=bank, movingwin=(1, 1))
    np.testing.assert_array_equal(gram.columns, [101.5, 102.5, 106.5, 107.5])


def test_native_coherence_aligns_different_slices_and_pools_on_shared_support() -> None:
    fs, t0 = 128, 100
    t = np.arange(10 * fs) / fs
    x = np.sin(2 * np.pi * 8 * t) + np.cos(2 * np.pi * 13 * t)
    y = np.cos(2 * np.pi * 8 * t)
    clock = t0 + t
    a = _lfp(x, clock)[nel.EpochArray([[102, 104], [106, 108]])]
    b = _lfp(y[fs:], clock[fs:])[nel.EpochArray([[101, 103], [106, 110]])]
    support = [[2, 3], [6, 8]]
    options = dict(fpass=[1, 40], tapers=(2.5, 4))
    expected = px.mtcoherencec(
        x, y, fs, **options, time_support=support, window_duration=1
    )
    pd.testing.assert_series_equal(
        px.mtcoherencec(a, b, **options, window_duration=1), expected
    )
    expected_gram = px.mtcohgramc(
        x, y, fs, **options, time_support=support, movingwin=(1, 1)
    )
    expected_gram.columns += t0
    pd.testing.assert_frame_equal(
        px.mtcohgramc(a, b, **options, movingwin=(1, 1)), expected_gram
    )

    # A mixed NumPy input is on the native clock, including after restriction.
    same_y = np.cos(2 * np.pi * 8 * (a.abscissa_vals - t0))
    paired = _lfp(y, clock)[a.support]
    pd.testing.assert_series_equal(
        px.mtcoherencec(same_y, a, **options, window_duration=1),
        px.mtcoherencec(paired, a, **options, window_duration=1),
    )


def test_native_continuous_input_rejects_invalid_sampling_or_shape() -> None:
    fs = 128
    t = np.arange(512) / fs
    x = np.sin(2 * np.pi * 8 * t)
    native = _lfp(x, t)
    with pytest.raises(ValueError, match="aligned sample times"):
        px.mtcoherencec(native, _lfp(x, t + 0.5 / fs), fpass=[1, 40])
    with pytest.raises(ValueError, match="sampling rate"):
        px.mtcoherencec(native, _lfp(x, 2 * t, fs / 2), fpass=[1, 40])
    with pytest.raises(ValueError, match="Select exactly one"):
        px.mtspectrumc(_lfp(np.vstack([x, x]), t), fpass=[1, 40])
    with pytest.raises(ValueError, match="equal length"):
        px.mtcoherencec(native, x[2:], fpass=[1, 40])


def test_low_level_transforms_accept_one_contiguous_native_input() -> None:
    fs = 128
    origin = 4
    x = np.arange(0.06, 40, 0.125)
    unit = _unit(x)[nel.EpochArray([[origin, origin + 1]])]
    t = origin + np.arange(-1, fs + 2) / fs
    f, mask = px.getfgrid(fs, 256, [1, 20])
    bank = px.dpsschk((2, 3), len(t), fs)
    actual, *counts = px.mtfftpt(unit, bank, 256, t, f, mask)
    expected, *expected_counts = px.mtfftpt(unit.data[0], bank, 256, t, f, mask)
    np.testing.assert_allclose(actual, expected)
    assert counts == expected_counts

    t = origin + np.arange(fs) / fs
    lfp = np.sin(2 * np.pi * 8 * t)
    bank = px.dpsschk((2, 3), len(lfp), fs)
    np.testing.assert_allclose(
        px.mtfftc(_lfp(lfp, t), bank, 256), px.mtfftc(lfp, bank, 256, fs)
    )
    native_gaps = _lfp(lfp, t)[nel.EpochArray([[4, 4.25], [4.5, 4.75]])]
    with pytest.raises(ValueError, match="single contiguous"):
        px.mtfftc(native_gaps, bank, 256)
    with pytest.raises(ValueError, match="single contiguous"):
        px.mtfftpt(_unit(x)[nel.EpochArray([[4, 5], [10, 11]])], bank, 256, t, f, mask)
