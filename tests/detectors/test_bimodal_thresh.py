import numpy as np

from neuro_py.detectors.up_down_state import bimodal_thresh, hartigan_diptest


def test_hartigan_diptest_unimodal_gaussian():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(500)

    dip, p_value = hartigan_diptest(data, n_boot=50, seed=123)

    assert p_value > 0.05, "Unimodal data should not be flagged as bimodal"


def test_hartigan_diptest_bimodal_mixture():
    rng = np.random.default_rng(7)
    data = np.concatenate([
        rng.normal(-3, 0.5, 300),
        rng.normal(3, 0.5, 300),
    ])

    dip, p_value = hartigan_diptest(data, n_boot=50, seed=321)

    assert dip > 0.05, "Dip statistic should increase for bimodal data"
    assert p_value < 0.05, "Bimodal mixture should yield significant dip test"


def test_bimodal_thresh_detects_modes_and_crossings():
    rng = np.random.default_rng(11)
    # Build a time-ordered series that alternates low/high segments multiple times to create crossings
    low_segment = rng.normal(-2, 0.3, 100)
    high_segment = rng.normal(2, 0.3, 100)
    series = np.concatenate([
        low_segment,
        high_segment,
        low_segment,
        high_segment,
    ])

    thresh, cross, bihist, dip_res = bimodal_thresh(series, nboot=50)

    assert not np.isnan(thresh), "Threshold should be found for bimodal data"
    assert dip_res["p"] < 0.05, "Dip test should detect bimodality in synthetic data"
    assert cross["upints"].size > 0 or cross["downints"].size > 0, "Crossings should be identified"
    # Threshold should sit between the two modes
    assert -1.0 < thresh < 1.0, "Threshold should lie between the two mode centers"
