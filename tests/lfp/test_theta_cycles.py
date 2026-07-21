import numpy as np

from neuro_py.lfp.theta_cycles import _detect_bycycle_features


def test_native_theta_cycle_detector_returns_saved_feature_columns() -> None:
    fs = 1_000.0
    times = np.arange(0, 5, 1 / fs)
    signal = np.sin(2 * np.pi * 8 * times)
    thresholds = {
        "amp_fraction": 0.1,
        "amp_consistency": 0.4,
        "period_consistency": 0.5,
        "monotonicity": 0.6,
        "min_n_cycles": 3,
    }

    features = _detect_bycycle_features(signal, fs, (6, 10), thresholds)

    assert {"sample_peak", "sample_last_trough", "band_amp", "is_burst"}.issubset(features.columns)
    assert features["sample_peak"].is_monotonic_increasing
    assert features["is_burst"].any()


def test_native_theta_cycle_detector_matches_bycycle_1_2_golden_features() -> None:
    """Regression fixture generated with Bycycle 1.2.0 on a deterministic trace."""
    fs = 1_000.0
    times = np.arange(0, 2, 1 / fs)
    signal = np.sin(2 * np.pi * 8 * times) + 0.15 * np.sin(2 * np.pi * 17 * times)
    thresholds = {
        "amp_fraction": 0.1,
        "amp_consistency": 0.4,
        "period_consistency": 0.5,
        "monotonicity": 0.6,
        "min_n_cycles": 3,
    }

    features = _detect_bycycle_features(signal, fs, (6, 10), thresholds)

    np.testing.assert_array_equal(
        features["sample_peak"].head(6).to_numpy(), [150, 285, 413, 536, 658, 781]
    )
    np.testing.assert_array_equal(
        features["sample_last_trough"].head(6).to_numpy(),
        [97, 219, 342, 464, 587, 715],
    )
    np.testing.assert_allclose(
        features["band_amp"].head(6).to_numpy(),
        [0.967221, 1.000108, 1.000042, 1.000017, 1.000004, 1.0],
        atol=1e-6,
    )
    np.testing.assert_array_equal(
        features["is_burst"].head(6).to_numpy(),
        [False, False, True, True, True, True],
    )
