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
