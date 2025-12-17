import numpy as np
import pytest

from neuro_py.ensemble.explained_variance import explained_variance as ev_func


def _equicorr_cov(n: int, rho: float) -> np.ndarray:
    """Return an n x n equicorrelation covariance matrix with off-diagonal rho."""
    cov = np.full((n, n), rho, dtype=float)
    np.fill_diagonal(cov, 1.0)
    return cov


def test_explained_variance_mvn_similarity():
    """
    Task and post share the same correlation structure; pre differs.
    Expect EV > rEV and EV reasonably positive.
    """
    rng = np.random.default_rng(42)
    n_features = 6
    n_time = 200

    cov_task = _equicorr_cov(n_features, rho=0.6)
    cov_post = _equicorr_cov(n_features, rho=0.6)
    cov_pre = _equicorr_cov(n_features, rho=0.1)

    # Generate T x N then transpose to (N, T)
    task = rng.multivariate_normal(
        mean=np.zeros(n_features), cov=cov_task, size=n_time
    ).T
    post = rng.multivariate_normal(
        mean=np.zeros(n_features), cov=cov_post, size=n_time
    ).T
    pre = rng.multivariate_normal(mean=np.zeros(n_features), cov=cov_pre, size=n_time).T

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
