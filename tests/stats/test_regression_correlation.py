import numpy as np
import pytest

from neuro_py.stats.regression import kernelReducedRankRegressor


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("shape", [(20,), (20, 1), (20, 3)])
def test_scorer_correlates_predictions_with_targets(sign, shape):
    target = np.random.default_rng(1).normal(size=shape)
    result = kernelReducedRankRegressor().rrr_scorer(sign * target + 7, target)
    assert result == pytest.approx(sign)


def test_scorer_averages_corresponding_target_correlations():
    rng = np.random.default_rng(3)
    target = rng.normal(size=(100, 3))
    prediction = target + rng.normal(size=target.shape)
    expected = np.mean(
        [np.corrcoef(prediction[:, i], target[:, i])[0, 1] for i in range(3)]
    )
    result = kernelReducedRankRegressor().rrr_scorer(prediction, target)
    assert result == pytest.approx(expected)
    assert result < 0.9


def test_constant_target_correlation_is_undefined():
    assert np.isnan(kernelReducedRankRegressor().rrr_scorer(np.ones(5), np.arange(5)))


@pytest.mark.parametrize(
    "prediction, target",
    [
        (np.ones((5, 2)), np.ones((5, 3))),
        (np.ones((2, 2, 2)), np.ones((2, 2, 2))),
        (np.ones((1, 2)), np.ones((1, 2))),
        (np.empty((5, 0)), np.empty((5, 0))),
    ],
)
def test_invalid_scoring_shapes_are_rejected(prediction, target):
    with pytest.raises(ValueError):
        kernelReducedRankRegressor().rrr_scorer(prediction, target)
