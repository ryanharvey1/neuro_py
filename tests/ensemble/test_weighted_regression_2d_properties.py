"""Independent regression and invariance checks for the 2D replay score."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from neuro_py.ensemble.replay import weighted_regression_2d, weighted_regression_2d_jit


def reference_fit(weights, x, y, t):
    """Independent weighted least-squares solve, not the moment formula."""
    xx, yy, tt = np.meshgrid(x, y, t, indexing="ij")
    w = np.nan_to_num(weights.astype(float), nan=0).ravel()
    target = np.column_stack((xx.ravel(), yy.ravel()))
    # Centering time keeps the reference stable for absolute timestamps.
    mt = np.average(tt.ravel(), weights=w)
    design = np.column_stack((np.ones(w.size), tt.ravel() - mt))
    coef = np.linalg.lstsq(
        design * np.sqrt(w[:, None]), target * np.sqrt(w[:, None]), rcond=None
    )[0]
    residual = target - design @ coef
    mean = np.average(target, axis=0, weights=w)
    sse = np.sum(w[:, None] * residual**2)
    sst = np.sum(w[:, None] * (target - mean) ** 2)
    score = max(0, 1 - sse / sst)
    trajectory = np.column_stack((np.ones(len(t)), t - mt)) @ coef
    return score, trajectory, coef[1], mean


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("layout", ["C", "F", "strided", "time_major"])
def test_matches_independent_weighted_least_squares(dtype, layout):
    w = np.random.default_rng(7).random((7, 9, 16)).astype(dtype)
    w[1, 2, 3] = np.nan
    if layout == "F":
        w = np.asfortranarray(w)
    elif layout == "time_major":
        w = np.ascontiguousarray(w.transpose(2, 0, 1)).transpose(1, 2, 0)
    elif layout == "strided":
        w = w[::2, ::2, ::2]
    x = np.linspace(-3, 6, w.shape[0])
    y = np.linspace(2, 18, w.shape[1])
    t = np.linspace(0, 1, w.shape[2]) ** 2
    expected = reference_fit(w, x, y, t)
    actual = weighted_regression_2d(w, x, y, t)
    assert_allclose(actual[0], expected[0], atol=2e-12)
    assert_allclose(np.column_stack(actual[1:3]), expected[1], atol=1e-6)
    assert_allclose(actual[3:5], expected[2], atol=2e-12)
    assert_allclose(actual[5:7], expected[3], atol=2e-12)
    direct = weighted_regression_2d_jit(w, x, y, t)
    for a, b in zip(actual, direct):
        assert_allclose(a, b)


def straight_posterior(dx, dy, spread=1):
    # Integer trajectories and symmetric isotropic uncertainty avoid rotation
    # interpolation or boundary artifacts. (5,0) and (3,4) have equal speed.
    w = np.zeros((65, 65, 9))
    offsets = np.arange(-2, 3)
    mass = np.exp(-(offsets**2) / (2 * spread**2))
    mass /= mass.sum()
    for k in range(9):
        w[np.ix_(10 + dx * k + offsets, 10 + dy * k + offsets, [k])] = (
            mass[:, None] * mass[None, :]
        )[..., None]
    return w


def test_rotation_and_reflection_invariance_with_uncertainty():
    a = straight_posterior(5, 0)
    b = straight_posterior(3, 4)
    scores = [
        weighted_regression_2d(w)[0] for w in (a, b, b[:, ::-1], b.transpose(1, 0, 2))
    ]
    assert_allclose(scores, scores[0], atol=1e-12)


def test_translation_common_spatial_scale_and_time_reversal():
    w = straight_posterior(3, 4).astype(np.float32)
    base = weighted_regression_2d(w)
    # Do not lose millisecond timing when weights are float32.
    t = 1e9 + np.arange(9) * 0.125
    moved = weighted_regression_2d(
        w, 1e6 + 2 * np.arange(65), -1e6 + 2 * np.arange(65), t
    )
    assert_allclose(moved[0], base[0], atol=1e-12)
    assert_allclose(moved[3:5], np.array(base[3:5]) * 16, atol=1e-12)
    rev = weighted_regression_2d(w[..., ::-1])
    assert_allclose(rev[0], base[0], atol=1e-12)
    assert_allclose(rev[3:5], -np.array(base[3:5]), atol=1e-12)


def test_posterior_uncertainty_is_penalized():
    sharp = weighted_regression_2d(straight_posterior(3, 4, 0.3))[0]
    broad = weighted_regression_2d(straight_posterior(3, 4, 2.0))[0]
    assert 0 < broad < sharp <= 1


@pytest.mark.parametrize("shape", [(0, 2, 3), (2, 0, 3), (2, 3, 0), (2, 3, 1)])
def test_empty_or_single_time_is_undefined(shape):
    assert np.isnan(weighted_regression_2d(np.ones(shape))[0])


def test_stationary_diffuse_and_point_posteriors():
    assert_allclose(weighted_regression_2d(np.ones((5, 5, 7)))[0], 0, atol=1e-12)
    w = np.zeros((5, 5, 7))
    w[2, 2, :] = 1
    result = weighted_regression_2d(w)
    assert np.isnan(result[0])
    assert_allclose(result[3:5], 0)
    assert_allclose(result[1], 2)


def test_integer_weights_are_promoted_and_missing_bins_keep_their_times():
    w = np.zeros((7, 1, 3), dtype=int)
    w[[0, 2, 6], 0, np.arange(3)] = 1
    assert_allclose(weighted_regression_2d(w, time_coords=np.array([0, 1, 3]))[0], 1)
    assert weighted_regression_2d(w)[0] < 1


@pytest.mark.parametrize("value", [-1.0, np.inf, -np.inf])
def test_invalid_weights_rejected(value):
    w = np.ones((3, 3, 3))
    w[0, 0, 0] = value
    with pytest.raises(ValueError, match="weights"):
        weighted_regression_2d(w)


def test_invalid_coordinates_rejected():
    with pytest.raises(ValueError, match="coordinate"):
        weighted_regression_2d(np.ones((3, 3, 3)), x_coords=np.arange(2))
    with pytest.raises(ValueError, match="finite"):
        weighted_regression_2d(np.ones((3, 3, 3)), time_coords=[0, np.nan, 2])
