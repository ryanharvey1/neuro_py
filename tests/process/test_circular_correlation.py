import numpy as np
import pytest

from neuro_py.process.precession_utils import (
    corrcc,
    corrcc_uniform,
    spatial_phase_precession,
)


@pytest.mark.parametrize("function", [corrcc, corrcc_uniform])
@pytest.mark.parametrize("axis", [0, 1, 2, -1, -2, -3])
def test_batched_correlations_match_independent_slices(function, axis):
    rng = np.random.default_rng(123)
    first = rng.normal(0, 0.7, size=(4, 5, 30))
    second = first + rng.normal(0, 0.9, size=first.shape)
    actual_rho, actual_p = function(first, second, axis=axis)
    rows_first = np.moveaxis(first, axis, -1)
    rows_second = np.moveaxis(second, axis, -1)
    expected_rho = np.empty(rows_first.shape[:-1])
    expected_p = np.empty_like(expected_rho)
    for index in np.ndindex(expected_rho.shape):
        expected_rho[index], expected_p[index] = function(
            rows_first[index], rows_second[index]
        )
    assert actual_rho.shape == expected_rho.shape
    assert actual_p.shape == expected_p.shape
    np.testing.assert_allclose(actual_rho, expected_rho, atol=1e-12)
    np.testing.assert_allclose(actual_p, expected_p, atol=1e-12)


@pytest.mark.parametrize("function", [corrcc, corrcc_uniform])
def test_flattened_correlation_uses_every_observation(function):
    rng = np.random.default_rng(4)
    first = rng.normal(size=(2, 12))
    second = first + rng.normal(size=first.shape)
    actual = function(first, second)
    expected = function(first.ravel(), second.ravel())
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("function", [corrcc, corrcc_uniform])
def test_one_dimensional_call_preserves_existing_statistics(function):
    first = np.array([0.1, 0.2, 0.4, 0.5])
    second = np.array([0.3, 0.6, 0.2, 0.8])
    # Reference calculation of the existing one-dimensional normal approximation.
    first_centered = first - np.angle(np.exp(1j * first).mean())
    second_centered = second - np.angle(np.exp(1j * second).mean())
    x, y = np.sin(first_centered), np.sin(second_centered)
    if function is corrcc:
        rho = np.sum(x * y) / np.sqrt(np.sum(x**2) * np.sum(y**2))
    else:
        numerator = abs(np.exp(1j * (first - second)).mean()) - abs(
            np.exp(1j * (first + second)).mean()
        )
        rho = first.size * numerator / (2 * np.sqrt(np.sum(x**2) * np.sum(y**2)))
    z = np.sqrt(first.size * np.mean(x**2) * np.mean(y**2) / np.mean(x**2 * y**2)) * rho
    from scipy.stats import norm

    np.testing.assert_allclose(
        function(first, second), [rho, 2 * (1 - norm.cdf(abs(z)))]
    )


@pytest.mark.parametrize("function", [corrcc, corrcc_uniform])
@pytest.mark.parametrize("axis", [-3, 2])
def test_invalid_axes_are_rejected(function, axis):
    with pytest.raises(ValueError, match="axis is out of bounds"):
        function(np.ones((3, 4)), np.ones((3, 4)), axis=axis)


def test_phase_precession_preserves_scalar_correlation_outputs():
    position = np.linspace(0, 1, 60)
    phase = (-3 * position + 0.4 + 0.1 * np.sin(20 * position)) % (2 * np.pi)
    rho, pvalue, slope, offset = spatial_phase_precession(
        phase, position, slope_bounds=[-4, -2]
    )
    assert isinstance(rho, float)
    assert isinstance(pvalue, float)
    assert np.all(np.isfinite([rho, pvalue, slope, offset]))
    assert rho < -0.9
    assert 0 <= pvalue <= 1
    assert -4 <= slope <= -2
