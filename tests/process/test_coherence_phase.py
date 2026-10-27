import numpy as np
import pytest
from joblib import Parallel
from scipy.signal.windows import dpss

from neuro_py.process import pychronux as px


@pytest.fixture(autouse=True)
def serial_taper_tasks(monkeypatch):
    monkeypatch.setattr(px, "Parallel", lambda n_jobs: Parallel(n_jobs=1))


def reference_spectra(first, second, support, n_tapers):
    times = np.arange(support[0] - 0.01, support[1] + 0.01, 0.01)
    tapers = dpss(len(times), 2.5, n_tapers).T
    frequencies, mask = px.getfgrid(100, 1000, [10, 10])
    j1, _, _ = px.mtfftpt(first, tapers, 1000, times, frequencies, mask)
    j2, _, _ = px.mtfftpt(second, tapers, 1000, times, frequencies, mask)
    cross = (j1.conj() * j2).mean(axis=1)
    expected = abs(cross) ** 2 / (
        (abs(j1) ** 2).mean(axis=1) * (abs(j2) ** 2).mean(axis=1)
    )
    return times, tapers, cross, expected


@pytest.mark.parametrize("n_tapers", [1, 4])
@pytest.mark.parametrize("shift", [0, 0.025, 0.05])
def test_coherence_uses_complex_cross_spectrum(n_tapers, shift):
    first = np.arange(0.2, 1.81, 0.1)
    second = first + shift
    _, _, _, expected = reference_spectra(first, second, [0, 2], n_tapers)
    actual = px.mtcoherencept(
        first, second, 100, [10, 10], n_tapers=n_tapers, time_support=[0, 2], nfft=1000
    )
    np.testing.assert_allclose(actual.Coherence, expected, atol=1e-12)


@pytest.mark.parametrize("transpose", [False, True])
def test_supplied_tapers_and_time_origin_are_shared(transpose):
    first = np.arange(0.2, 1.81, 0.1) + 100
    second = first + 0.025
    times, tapers, cross, expected = reference_spectra(first, second, [100, 102], 4)
    supplied = tapers.T if transpose else tapers
    original = supplied.copy()
    actual = px.mtcoherencept(
        first, second, 100, [10, 10], tapers=supplied, tapers_ts=times, nfft=1000
    )
    np.testing.assert_allclose(actual.Coherence, expected, atol=1e-12)
    real = px.mtcsdpt(
        first, second, 100, [10, 10], tapers=supplied, tapers_ts=times, nfft=1000
    )
    complex_result = px.mtcsdpt(
        first,
        second,
        100,
        [10, 10],
        tapers=supplied,
        tapers_ts=times,
        nfft=1000,
        return_complex=True,
    )
    assert not np.iscomplexobj(real.CSD.values)
    np.testing.assert_allclose(real.CSD, cross.real)
    np.testing.assert_allclose(complex_result.CSD, cross)
    np.testing.assert_array_equal(supplied, original)


def test_inferred_support_matches_explicit_joint_support():
    first = np.arange(0.2, 1.81, 0.1)
    second = first + 0.025
    expected = px.mtcoherencept(
        first, second, 100, [1, 20], time_support=[first.min(), second.max()]
    )
    actual = px.mtcoherencept(first, second, 100, [1, 20])
    np.testing.assert_allclose(actual.Coherence, expected.Coherence)
    np.testing.assert_array_equal(actual.index, expected.index)


def test_swapping_inputs_conjugates_cross_spectrum_without_changing_coherence():
    first = np.arange(0.2, 1.81, 0.1)
    second = first + 0.025
    kwargs = dict(time_support=[0, 2], nfft=1000)
    forward = px.mtcsdpt(first, second, 100, [10, 10], return_complex=True, **kwargs)
    backward = px.mtcsdpt(second, first, 100, [10, 10], return_complex=True, **kwargs)
    np.testing.assert_allclose(forward.CSD, backward.CSD.to_numpy().conj())
    np.testing.assert_allclose(
        px.mtcoherencept(first, second, 100, [10, 10], **kwargs),
        px.mtcoherencept(second, first, 100, [10, 10], **kwargs),
    )


def test_zero_power_coherence_is_undefined():
    result = px.mtcoherencept(
        np.array([]), np.array([]), 100, [1, 20], time_support=[0, 2]
    )
    assert np.isnan(result.Coherence).all()
