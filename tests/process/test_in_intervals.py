import numpy as np
from neuro_py.process.intervals import in_intervals


def test_in_intervals():
    # Test 1
    timestamps = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    intervals = np.array([[2, 4], [5, 7]])
    expected_output = np.array([False, True, True, True, True, True, True, False])
    assert np.array_equal(in_intervals(timestamps, intervals), expected_output)

    # Test 2
    timestamps = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    intervals = np.array([[2, 4], [5, 7], [8, 10]])
    expected_output = np.array(
        [False, True, True, True, True, True, True, True, True, True]
    )
    assert np.array_equal(in_intervals(timestamps, intervals), expected_output)

    # Test 3
    timestamps = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    intervals = np.array([[2, 4], [5, 7], [8, 9]])
    expected_output = np.array(
        [False, True, True, True, True, True, True, True, True, False]
    )
    assert np.array_equal(in_intervals(timestamps, intervals), expected_output)
