import numpy as np
import pytest

from neuro_py.ensemble.decoding.pipeline import zscore_trial_segs


@pytest.mark.parametrize("dtype", [np.int16, np.int64, np.float32, np.float64])
def test_dense_trials_preserve_fractional_zscores_and_training_parameters(dtype):
    train = np.arange(24, dtype=dtype).reshape(2, 6, 2)
    holdout = train + 10
    original = train.copy()
    actual, rest, params = zscore_trial_segs(train, [holdout])
    mean = train.reshape(-1, 2).mean(axis=0)
    std = train.reshape(-1, 2).std(axis=0)
    assert np.issubdtype(actual.dtype, np.floating)
    assert np.issubdtype(rest[0].dtype, np.floating)
    np.testing.assert_allclose(actual, (train - mean) / std)
    np.testing.assert_allclose(rest[0], (holdout - mean) / std)
    np.testing.assert_array_equal(params["X_train_mean"], mean)
    np.testing.assert_allclose(params["X_train_std"], std)
    np.testing.assert_array_equal(train, original)


def test_ragged_integer_trials_preserve_trial_shapes_and_fractional_values():
    train = np.empty(2, dtype=object)
    train[0] = np.arange(6).reshape(3, 2)
    train[1] = np.arange(10).reshape(5, 2) + 8
    actual, rest, params = zscore_trial_segs(train, [train.copy()])
    assert actual.shape == (2,)
    for source, result, heldout in zip(train, actual, rest[0]):
        expected = (source - params["X_train_mean"]) / params["X_train_std"]
        assert result.shape == source.shape
        np.testing.assert_allclose(result, expected)
        np.testing.assert_allclose(heldout, expected)


def test_normalization_reuses_training_parameters_and_zeroes_constant_features():
    train = np.arange(24).reshape(2, 6, 2)
    train[:, :, 1] = 7
    normed, _, params = zscore_trial_segs(train)
    applied, _, reused = zscore_trial_segs(train + 10, normparams=params)
    np.testing.assert_array_equal(normed[:, :, 1], 0)
    np.testing.assert_array_equal(applied[:, :, 1], 0)
    np.testing.assert_allclose(
        applied[:, :, 0],
        (train[:, :, 0] + 10 - params["X_train_mean"][0]) / params["X_train_std"][0],
    )
    np.testing.assert_array_equal(reused["X_train_mean"], params["X_train_mean"])
