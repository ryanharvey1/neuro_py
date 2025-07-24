from neuro_py.process.intervals import truncate_epoch
import nelpy as nel
import numpy as np


def test_truncate_epoch():
    # Create a sample epoch with intervals of various lengths
    start_times = [0, 5, 10, 15, 20]
    end_times =   [2, 7, 13, 18, 25]
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
    end_times =   [2, 7, 10.5, 18, 25]
    epoch_data = [(start, end) for start, end in zip(start_times, end_times)]
    epoch = nel.EpochArray(epoch_data)

    truncated = truncate_epoch(epoch, time=4.25)
    assert truncated.duration == 4.25

    # Test case 5: Truncation with partial interval
    start_times = [0, 5, 10, 15, 20]
    end_times =   [2, 7, 10.1, 18, 25]
    epoch_data = [(start, end) for start, end in zip(start_times, end_times)]
    epoch = nel.EpochArray(epoch_data)

    truncated = truncate_epoch(epoch, time=4.25)
    assert truncated.duration == 4.25