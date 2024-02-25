import pytest
from neuro_py.process.peri_event import nearest_event_delay
import numpy as np

def test_nearest_event_delay():
    # Test case 1
    ts_1 = [0, 4, 15, 20]
    ts_2 = [1, 5, 8, 10, 21, 40, 90]
    expected_nearest = [1, 5, 10, 21]
    expected_delay = [-1, -1, 5, -1]
    expected_nearest_index = [0, 1, 3, 4]

    close_ts, delay, nearest_index = nearest_event_delay(ts_1, ts_2)
    assert np.array_equal(close_ts, expected_nearest), "Test case 1 failed: nearest timestamps do not match"
    assert np.array_equal(delay, expected_delay), "Test case 1 failed: delays do not match"
    assert np.array_equal(nearest_index, expected_nearest_index), "Test case 1 failed: nearest indices do not match"

    # Test case 2: Edge case with empty input arrays
    ts_1_empty = []
    ts_2_empty = []
    with pytest.raises(ValueError, match="ts_1 is empty"):
        nearest_event_delay(ts_1_empty, ts_2_empty)

    # Test case 3: Edge case with non-monotonic ts_2
    ts_1_non_mono = [2, 5, 10]
    ts_2_non_mono = [15, 10, 5]
    with pytest.raises(ValueError, match="ts_2 must be monotonically increasing"):
        nearest_event_delay(ts_1_non_mono, ts_2_non_mono)