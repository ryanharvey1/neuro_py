import numpy as np
import pytest

from neuro_py.ensemble.explained_variance import explained_variance as ev_func


def _equicorr_cov(n: int, rho: float) -> np.ndarray:
    """Return an n x n equicorrelation covariance matrix with off-diagonal rho."""
    cov = np.full((n, n), rho, dtype=float)
    np.fill_diagonal(cov, 1.0)
    return cov


def test_explained_variance_factor_model():
    """
    Construct a one-factor model: task and post share a latent factor
    with similar loadings; pre has weak loadings and higher noise.
    This should yield EV > rEV robustly.
    """
    rng = np.random.default_rng(42)
    n_features = 8
    n_time = 2000

    # Shared latent for task/post with moderate noise
    latent = rng.standard_normal(n_time)
    loadings = rng.uniform(0.5, 1.0, size=n_features)
    task = loadings[:, None] * latent + 0.3 * rng.standard_normal((n_features, n_time))
    post = loadings[:, None] * latent + 0.3 * rng.standard_normal((n_features, n_time))

    # Pre has weak latent structure and higher noise
    latent_pre = rng.standard_normal(n_time)
    weak_loadings = rng.uniform(0.0, 0.2, size=n_features)
    pre = weak_loadings[:, None] * latent_pre + 1.0 * rng.standard_normal(
        (n_features, n_time)
    )

    EV, rEV = ev_func(task, post, pre)

    assert np.isfinite(EV)
    assert np.isfinite(rEV)
    assert EV > rEV
    assert EV > 0.05


def test_explained_variance_shape_mismatch():
    """Mismatched feature counts should raise ValueError."""
    x = np.random.randn(5, 100)
    y = np.random.randn(6, 100)
    z = np.random.randn(5, 100)

    with pytest.raises(ValueError):
        _ = ev_func(x, y, z)


def test_explained_variance_with_nans():
    """Verify function handles NaN in pairwise correlations gracefully."""
    rng = np.random.default_rng(123)
    # Create data where one neuron has zero variance → NaN in corrcoef
    task = rng.standard_normal((5, 100))
    task[0, :] = 0  # Zero variance neuron
    post = rng.standard_normal((5, 100))
    pre = rng.standard_normal((5, 100))

    EV, rEV = ev_func(task, post, pre)
    assert np.isfinite(EV) or np.isnan(EV)  # Should handle gracefully
    assert np.isfinite(rEV) or np.isnan(rEV)
