import numpy as np
import pytest
from joblib import Parallel
from scipy.signal.windows import dpss

from neuro_py import process
from neuro_py.process import pychronux as px


def test_continuous_coherence_matches_tapered_cross_spectrum() -> None:
    fs = 100
    time = np.arange(128) / fs
    first = np.sin(2 * np.pi * 9 * time) + 0.3 * np.cos(2 * np.pi * 21 * time)
    second = np.cos(2 * np.pi * 9 * time) + 0.3 * np.sin(2 * np.pi * 23 * time)
    bank = dpss(len(time), 2.5, 4)
    j1 = np.fft.rfft(bank * first, axis=1)
    j2 = np.fft.rfft(bank * second, axis=1)
    cross = np.mean(j1.conj() * j2, axis=0)
    expected = np.abs(cross) ** 2 / (
        np.mean(np.abs(j1) ** 2, axis=0) * np.mean(np.abs(j2) ** 2, axis=0)
    )

    coherence = process.mtcoherencec(first, second, fs, [0, 50], (2.5, 4))
    np.testing.assert_array_equal(coherence.index, np.fft.rfftfreq(len(time), 1 / fs))
    np.testing.assert_allclose(coherence, expected, atol=1e-12)
    assert px.mtcoherencec(first, np.zeros_like(second), fs, [0, 50], (2.5, 4)).isna().all()


def test_continuous_grams_match_standalone_estimates_per_complete_window() -> None:
    fs = 100
    rng = np.random.default_rng(42)
    first, second = rng.normal(size=(2, 275))
    opts = (fs, [1, 40], (2.5, 4))
    power = process.mtspecgramc(first, *opts, movingwin=(1, 0.5))
    coherence = process.mtcohgramc(first, second, *opts, movingwin=(1, 0.5))
    np.testing.assert_array_equal(power.columns, [0.5, 1, 1.5, 2])
    np.testing.assert_array_equal(coherence.columns, power.columns)

    for i, start in enumerate(range(0, 200, 50)):
        window = slice(start, start + 100)
        expected_power = px.mtspectrumc(first[window], *opts)
        expected_coherence = px.mtcoherencec(first[window], second[window], *opts)
        np.testing.assert_array_equal(power.index, expected_power.index)
        np.testing.assert_allclose(power.iloc[:, i], expected_power)
        np.testing.assert_allclose(coherence.iloc[:, i], expected_coherence)

    assert px.mtspecgramc(first[:50], *opts, movingwin=(1, 0.5)).shape[1] == 0
    with pytest.raises(ValueError, match="positive finite"):
        px.mtspecgramc(first, *opts, movingwin=(1, 0))
    with pytest.raises(ValueError, match="equal length"):
        px.mtcohgramc(first, second[:-1], *opts, movingwin=(1, 0.5))


def test_spike_gram_preserves_empty_windows_and_boundary_assignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(px, "Parallel", lambda n_jobs: Parallel(n_jobs=1))
    fs = 100
    band = [0, 40]
    times = np.array([14.0, 10.2, 11.0, 11.24, 13.18, 13.4])
    gram = process.mtspecgrampt(times, fs, band, movingwin=(1, 1), time_support=[10, 14])
    np.testing.assert_array_equal(gram.columns, [10.5, 11.5, 12.5, 13.5])
    for center in gram.columns:
        start = center - 0.5
        spikes = times[(times >= start) & (times < start + 1)] - start
        expected = px.mtspectrumpt([spikes], fs, band, time_support=[0, 1])
        np.testing.assert_array_equal(gram.index, expected.index)
        np.testing.assert_allclose(gram[center], expected.iloc[:, 0], atol=1e-12)

    np.testing.assert_array_equal(gram[12.5], 0)
    with pytest.raises(ValueError, match="required for an empty spike train"):
        px.mtspecgrampt(np.array([]), fs, band, movingwin=(1, 1))
