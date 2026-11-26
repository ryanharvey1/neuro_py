import warnings

import numpy as np
import pytest

from neuro_py.stats import circ_stats


@pytest.mark.parametrize("axial_correction", [2, 3, 4])
@pytest.mark.parametrize("spread", [0.3, 1.2])
def test_axial_mean_ci_uses_concentration_of_equivalent_directions(
    axial_correction: int, spread: float
) -> None:
    # Equivalent axial directions cancel on the original circle, but are
    # concentrated after transformation. Exercise both CI approximation regimes.
    offsets = np.linspace(-spread, spread, 40)
    alpha = (
        0.4 + offsets[None, :] + 2 * np.pi * np.arange(axial_correction)[:, None]
    ) / axial_correction
    expected_width = circ_stats.mean_ci_limits(alpha * axial_correction)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mu, ci = circ_stats.mean(alpha, ci=0.95, axial_correction=axial_correction)

    assert not caught
    assert isinstance(ci, circ_stats.CI)
    assert mu == pytest.approx(0.4 / axial_correction)
    assert ci.lower == pytest.approx(mu - expected_width / axial_correction)
    assert ci.upper == pytest.approx(mu + expected_width / axial_correction)
    assert circ_stats.mean(alpha, axial_correction=axial_correction) == pytest.approx(
        mu
    )


@pytest.mark.parametrize("axis", [None, 0, 1, -1])
@pytest.mark.parametrize("d", [None, 0.02])
def test_axial_mean_ci_preserves_weights_axes_and_bin_spacing(
    axis: int | None, d: float | None
) -> None:
    axial_correction = 3
    alpha = np.linspace(0.1, 0.5, 60).reshape(6, 10)
    weights = (np.arange(alpha.size).reshape(alpha.shape) % 4) + 1
    scaled_d = None if d is None else d * axial_correction
    expected_width = (
        circ_stats.mean_ci_limits(
            alpha * axial_correction, ci=0.9, w=weights, d=scaled_d, axis=axis
        )
        / axial_correction
    )

    mu, ci = circ_stats.mean(
        alpha, w=weights, ci=0.9, d=d, axis=axis, axial_correction=axial_correction
    )

    assert np.shape(mu) == np.shape(np.sum(alpha, axis=axis))
    assert np.shape(ci.lower) == np.shape(mu)
    assert np.shape(ci.upper) == np.shape(mu)
    np.testing.assert_allclose(ci.lower, mu - expected_width)
    np.testing.assert_allclose(ci.upper, mu + expected_width)


def test_axial_mean_ci_weights_match_repeated_observations() -> None:
    alpha = np.array([0.1, 0.2, 0.3, np.pi + 0.1, np.pi + 0.2])
    weights = np.array([2, 3, 5, 2, 4])

    weighted = circ_stats.mean(alpha, w=weights, ci=0.95, axial_correction=2)
    repeated = circ_stats.mean(np.repeat(alpha, weights), ci=0.95, axial_correction=2)

    np.testing.assert_allclose(weighted[0], repeated[0])
    np.testing.assert_allclose(weighted[1], repeated[1])


@pytest.mark.parametrize("center", [-0.4, 0.0, np.pi / 2])
def test_axial_mean_ci_wraps_around_existing_mean_representative(center: float) -> None:
    alpha = center + np.tile(np.linspace(-0.1, 0.1, 40), 2)
    alpha[40:] += np.pi
    width = circ_stats.mean_ci_limits(2 * alpha) / 2
    expected_mean = circ_stats.mean(alpha, axial_correction=2)

    mu, ci = circ_stats.mean(alpha, ci=0.95, axial_correction=2)

    assert mu == pytest.approx(expected_mean)
    # Modulo can round a tiny negative mean to exactly 2*pi.
    assert 0 <= mu <= 2 * np.pi
    np.testing.assert_allclose(ci.lower, (mu - width) % (2 * np.pi))
    np.testing.assert_allclose(ci.upper, (mu + width) % (2 * np.pi))


def test_axial_mean_ci_rejects_insufficient_axial_concentration() -> None:
    alpha = np.linspace(0, np.pi, 100, endpoint=False)

    with pytest.raises(UserWarning, match="Requirements for confidence levels not met"):
        circ_stats.mean(alpha, ci=0.95, axial_correction=2)


def test_directional_mean_ci_is_unchanged() -> None:
    alpha = np.linspace(0.1, 0.5, 40)
    expected_width = circ_stats.mean_ci_limits(alpha)

    mu, ci = circ_stats.mean(alpha, ci=0.95)

    assert mu == pytest.approx(0.3)
    assert ci.lower == pytest.approx(0.3 - expected_width)
    assert ci.upper == pytest.approx(0.3 + expected_width)
