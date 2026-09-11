import nelpy as nel
import numpy as np
import pytest

from neuro_py.process.intervals import truncate_epoch


def test_truncate_epoch():
    # Create a sample epoch with intervals of various lengths
    start_times = [0, 5, 10, 15, 20]
    end_times = [2, 7, 13, 18, 25]
    epoch_data = [(start, end) for start, end in zip(start_times, end_times)]
    epoch = nel.EpochArray(epoch_data)

    # Test case 1: No truncation needed
    truncated = truncate_epoch(epoch, time=30)
    assert np.all(truncated.data == epoch.data)

    # Test case 2: Truncation needed
    truncated = truncate_epoch(epoch, time=10)
    assert truncated.duration == 10

    # Test case 3: Truncation with fractional interval
    truncated = truncate_epoch(epoch, time=8)
    assert truncated.duration == 8

    # Test case 4: Truncation with multiple intervals
    start_times = [0, 5, 10, 15, 20]
    end_times = [2, 7, 10.5, 18, 25]
    epoch_data = [(start, end) for start, end in zip(start_times, end_times)]
    epoch = nel.EpochArray(epoch_data)

    truncated = truncate_epoch(epoch, time=4.25)
    assert truncated.duration == 4.25

    # Test case 5: Truncation with partial interval
    start_times = [0, 5, 10, 15, 20]
    end_times = [2, 7, 10.1, 18, 25]
    epoch_data = [(start, end) for start, end in zip(start_times, end_times)]
    epoch = nel.EpochArray(epoch_data)

    truncated = truncate_epoch(epoch, time=4.25)
    assert truncated.duration == 4.25


def test_truncate_epoch_from_end():
    epoch = nel.EpochArray([(0, 2), (5, 7), (10, 13), (15, 18), (20, 25)])

    truncated = truncate_epoch(epoch, time=6.25, from_end=True)

    expected = np.array([[16.75, 18], [20, 25]])
    np.testing.assert_allclose(truncated.data, expected)
    assert truncated.duration == 6.25


def test_truncate_epoch_from_end_exact_interval_boundary():
    epoch = nel.EpochArray([(0, 2), (5, 7), (10, 13), (15, 18), (20, 25)])

    truncated = truncate_epoch(epoch, time=8, from_end=True)

    expected = np.array([[15, 18], [20, 25]])
    np.testing.assert_allclose(truncated.data, expected)


def test_truncate_epoch_from_end_contiguous_epoch():
    epoch = nel.EpochArray([(0, 4 * 60 * 60)])

    truncated = truncate_epoch(epoch, time=60 * 60, from_end=True)

    np.testing.assert_allclose(truncated.data, [[3 * 60 * 60, 4 * 60 * 60]])


def test_truncate_epoch_from_end_no_truncation_needed():
    epoch = nel.EpochArray([(0, 2), (5, 7)])

    truncated = truncate_epoch(epoch, time=5, from_end=True)

    np.testing.assert_array_equal(truncated.data, epoch.data)


def test_truncate_empty_epoch_from_end():
    epoch = nel.EpochArray(empty=True)

    truncated = truncate_epoch(epoch, time=1, from_end=True)

    assert truncated.isempty


@pytest.mark.parametrize("from_end", [False, True])
@pytest.mark.parametrize("time", [0, -1])
def test_truncate_epoch_nonpositive_time(time, from_end):
    epoch = nel.EpochArray([(0, 2), (5, 7)])

    truncated = truncate_epoch(epoch, time=time, from_end=from_end)

    assert truncated.isempty
