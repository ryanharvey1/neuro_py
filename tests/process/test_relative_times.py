import numpy as np
from neuro_py.process import peri_event


def test_relative_times():
    # Test 1: basic test case
    t = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    intervals = np.array([[1, 3], [4, 6], [7, 9]])
    assert np.allclose(
        peri_event.relative_times(t, intervals),
        (
            np.array([np.nan, 0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.0, 0.5, 1.0]),
            np.array([np.nan, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0]),
        ),
        atol=1e-6,
        equal_nan=True,
    )

    # Test 2: with values assigned to each interval
    t = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    intervals = np.array([[1, 3], [4, 6], [7, 9]])
    values = np.array([10, 20])
    assert np.allclose(
        peri_event.relative_times(t, intervals, values),
        (
            np.array([np.nan, 10, 15, 20, 10, 15, 20, 10, 15, 20]),
            np.array([np.nan, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0]),
        ),
        atol=1e-6,
        equal_nan=True,
    )

    # Test 3: when t is outside of all intervals
    t = np.array([-2, -1])
    intervals = np.array([[1, 3], [4, 6], [7, 9]])
    assert np.allclose(
        peri_event.relative_times(t, intervals),
        (
            np.array([np.nan, np.nan]),
            np.array([np.nan, np.nan]),
        ),
        atol=1e-6,
        equal_nan=True,
    )

    # Test 4: when t is at the start of the first interval
    t = np.array([1])
    intervals = np.array([[1, 3], [4, 6], [7, 9]])
    assert np.allclose(
        peri_event.relative_times(t, intervals),
        (np.array([0.0]), np.array([0.0])),
        atol=1e-6,
    )


# %%
