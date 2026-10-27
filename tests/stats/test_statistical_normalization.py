import bottleneck as bn
import numpy as np
import pytest
from scipy import stats
from sklearn.metrics import mean_squared_error

from neuro_py.stats.regression import kernelReducedRankRegressor
from neuro_py.stats.stats import confidence_intervals


@pytest.mark.parametrize(
    "estimator", [np.nanmean, bn.nanmean, np.nanmedian, bn.nanmedian]
)
def test_missing_observations_do_not_narrow_confidence_intervals(estimator):
    observed = np.array([[0.0], [1.0]])
    expected = confidence_intervals(observed, estimator=estimator)
    padded = np.vstack((observed, np.full((98, 1), np.nan)))
    actual = confidence_intervals(padded, estimator=estimator)
    np.testing.assert_allclose(actual, expected)


def test_confidence_intervals_use_each_columns_observation_count():
    samples = np.array(
        [
            [0, 3, np.nan, 1],
            [1, 5, np.nan, np.nan],
            [np.nan, 9, np.nan, np.nan],
            [np.nan, 7, np.nan, np.nan],
        ]
    )
    lower, upper = confidence_intervals(samples)
    for column in range(2):
        observed = samples[:, column]
        observed = observed[~np.isnan(observed)]
        expected = stats.t.interval(
            0.95, len(observed) - 1, loc=observed.mean(), scale=stats.sem(observed)
        )
        np.testing.assert_allclose([lower[column], upper[column]], expected)
    assert np.isnan(lower[2:]).all() and np.isnan(upper[2:]).all()


@pytest.mark.parametrize("n_samples,n_targets", [(1, 1), (4, 3), (20, 2)])
def test_mse_is_normalized_once(n_samples, n_targets):
    features = np.arange(n_samples, dtype=float)[:, None]
    model = kernelReducedRankRegressor(rank=1, reg=1)
    model.fit(features, np.zeros((n_samples, n_targets)))
    assert model.mse(features, np.ones((n_samples, n_targets))) == pytest.approx(1)


def test_mse_matches_standard_reference_and_single_target_shapes():
    rng = np.random.default_rng(6)
    features = rng.normal(size=(12, 3))
    targets = rng.normal(size=(12, 1))
    model = kernelReducedRankRegressor(rank=1, reg=1)
    model.fit(features, targets)
    expected = mean_squared_error(targets, model.predict(features).real)
    assert model.mse(features, targets) == pytest.approx(expected)
    assert model.mse(features, targets[:, 0]) == pytest.approx(expected)
    with pytest.raises(ValueError, match="same nonempty shape"):
        model.mse(features, np.ones((12, 2)))
    with pytest.raises(ValueError, match="same nonempty shape"):
        model.mse(features[:0], targets[:0])
