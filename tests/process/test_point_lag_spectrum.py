from itertools import product

import nelpy as nel
import numpy as np
import pandas as pd
import pytest
from scipy.signal import find_peaks

from neuro_py import process
from neuro_py.process import pychronux as px


def test_lag_spectrum_matches_direct_same_bout_pair_counts() -> None:
    fs = 8
    max_offset = 3
    bounds = [[0, 0.5], [100, 100.875], [200, 200.25]]
    # Each row below is the bin counts in one bout. Include repeated spikes
    # within a single bin, a silent short bout, and a spike on a stop boundary.
    counts = [np.array([1, 2, 0, 1]), np.array([0, 1, 0, 1, 1, 0, 0]), np.zeros(2)]
    spikes = [*np.repeat(np.arange(4) / fs, counts[0]), 0.5, 50.0]
    spikes.extend(100 + np.repeat(np.arange(7) / fs, counts[1]))
    lags = np.arange(max_offset + 1)
    eligible = np.zeros_like(lags, dtype=float)
    centered = np.zeros_like(eligible)
    for x in counts:
        n, m = x.sum(), len(x)
        for k in range(min(m, max_offset + 1)):
            pairs = x[: m - k] @ x[k:]
            if k == 0:
                pairs -= n
            centered[k] += pairs - n * (n - 1) * (m - k) / m**2
            eligible[k] += m - k
    excess = centered / eligible
    window = 0.5 * (1 + np.cos(np.pi * lags / max_offset))
    f = np.arange(9) * fs / 16
    expected = fs * (
        excess[0]
        + 2 * (np.cos(2 * np.pi * np.outer(f, lags[1:] / fs)) @ (excess * window)[1:])
    )

    actual, exposure = process.point_lag_spectrum(
        np.array(spikes[::-1]),
        fs,
        [0, fs / 2],
        time_support=bounds,
        max_lag=max_offset / fs,
        nfft=16,
    )
    np.testing.assert_array_equal(actual.index, f)
    np.testing.assert_allclose(actual[0], expected, atol=1e-14)
    np.testing.assert_array_equal(exposure["pair_exposure_s"], eligible / fs)
    np.testing.assert_array_equal(exposure["n_bouts"], [3, 3, 2, 2])


def test_count_matched_uniform_spikes_have_zero_expected_excess() -> None:
    # Enumerate both spike positions in the exact count-matched uniform null.
    trains = [np.array(pair) / 8 for pair in product(range(4), repeat=2)]
    spectrum, _ = px.point_lag_spectrum(
        trains, 8, [0, 4], max_lag=0.25, time_support=[0, 0.5]
    )
    assert (spectrum.to_numpy() < 0).any()  # do not clip finite-sample estimates
    np.testing.assert_allclose(spectrum.mean(axis=1), 0, atol=1e-15)


def test_seeded_theta_example_resolves_close_peaks_with_many_short_bouts() -> None:
    fs = 128
    rng = np.random.default_rng(64)
    n_short, n_long = 600, 200
    starts = 4 * np.arange(n_short + n_long, dtype=float)
    lengths = np.r_[rng.integers(22, 66, n_short), rng.integers(190, 350, n_long)]
    bounds = np.column_stack((starts, starts + lengths / fs))
    spikes = []
    for start, length in zip(starts, lengths):
        t = (np.arange(length) + 0.5) / fs
        low, high = rng.uniform(0, 2 * np.pi, 2)
        rate = 50 + 20 * np.cos(2 * np.pi * 7.25 * t + low)
        rate += 20 * np.cos(2 * np.pi * 8.75 * t + high)
        spikes.append(start + np.repeat(t, rng.poisson(rate / fs)))
    train = np.concatenate(spikes)
    st = nel.SpikeTrainArray(
        timestamps=train, fs=30_000, support=nel.EpochArray(bounds)
    )
    spectrum, exposure = px.point_lag_spectrum(st, fs, [5, 11])
    direct, _ = px.point_lag_spectrum(train, fs, [5, 11], time_support=bounds)
    pd.testing.assert_frame_equal(spectrum, direct)
    peaks, _ = find_peaks(spectrum[0], prominence=3)
    np.testing.assert_allclose(spectrum.index[peaks], [7.25, 8.75], atol=0.25)
    assert exposure.loc[0.125, "n_bouts"] == n_short + n_long
    assert exposure.loc[1, "n_bouts"] == n_long
    assert lengths.max() / fs < 4


def test_lag_spectrum_rejects_unsupported_lags_and_preserves_silent_units() -> None:
    with pytest.raises(ValueError, match="time_support is required"):
        px.point_lag_spectrum(np.array([0.1]), 128, [4, 12])
    with pytest.raises(ValueError, match="No same-bout bin pairs"):
        px.point_lag_spectrum(
            np.array([0.1]), 128, [4, 12], time_support=[[0, 0.3], [50, 50.4]]
        )
    silent, _ = px.point_lag_spectrum(np.array([]), 128, [4, 12], time_support=[0, 2])
    assert silent.shape[1] == 1
    np.testing.assert_array_equal(silent, 0)
