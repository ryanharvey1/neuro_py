import numpy as np

from neuro_py.process import peri_event


def test_deconvolve_peth():
    signal = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    events = np.array([2, 5, 8])
    expected_deconvolved = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    expected_times = np.array(
        [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    )
    result_deconvolved, result_times = peri_event.deconvolve_peth(
        signal, events, n_bins=10, bin_width=0.1
    )

    assert np.allclose(result_deconvolved, expected_deconvolved, atol=1), (
        f"Expected {expected_deconvolved}, but got {result_deconvolved}"
    )
    assert np.allclose(result_times, expected_times, atol=1e-6), (
        f"Expected {expected_times}, but got {result_times}"
    )
