__all__ = [
    "find_intersecting_intervals",
    "find_interval",
    "in_intervals",
    "in_intervals_interval",
    "overlap_intersect",
    "randomize_epochs",
]

from typing import List, Union
import numpy as np
import random
from numba import jit, njit
import numba
from nelpy import core


def randomize_epochs(epoch, randomize_each=True, start_stop=None):
    """Randomly shifts the epochs of a EpochArray object and wraps them around the original time boundaries.

    This method takes a EpochArray object as input, and can either randomly shift each epoch by a different amount
    (if `randomize_each` is True) or shift all the epochs by the same amount (if `randomize_each` is False).
    In either case, the method wraps the shifted epochs around the original time boundaries to make sure they remain
    within the original time range. It then returns the modified EpochArray object.

    Args:
        epoch (EpochArray): The EpochArray object whose epochs should be shifted and wrapped.
        randomize_each (bool, optional): If True, each epoch will be shifted by a different random amount.
            If False, all the epochs will be shifted by the same random amount. Defaults to True.
        start_stop (array, optional): If not None, time support will be taken from start_stop

    Returns:
        new_epochs: The modified EpochArray object with the shifted and wrapped epochs.
    """

    def wrap_intervals(intervals, start, stop):
        idx = np.any(intervals > stop, axis=1)
        intervals[idx] = intervals[idx] - stop + start

        idx = np.any(intervals < start, axis=1)
        intervals[idx] = intervals[idx] - start + stop
        return intervals

    new_epochs = epoch.copy()

    if start_stop is None:
        start = new_epochs.start
        stop = new_epochs.stop
    else:
        start, stop = start_stop

    ts_range = stop - start

    if randomize_each:
        # Randomly shift each epoch by a different amount
        random_order = random.sample(
            range(-int(ts_range), int(ts_range)), new_epochs.n_intervals
        )

        new_intervals = new_epochs.data + np.expand_dims(random_order, axis=1)
        new_epochs._data = wrap_intervals(new_intervals, start, stop)
    else:
        # Shift all the epochs by the same amount
        random_shift = random.randint(-int(ts_range), int(ts_range))
        new_epochs._data = wrap_intervals((new_epochs.data + random_shift), start, stop)

    if not new_epochs.isempty:
        if np.any(new_epochs.data[:, 1] - new_epochs.data[:, 0] < 0):
            raise ValueError("start must be less than or equal to stop")

    new_epochs._sort()

    return new_epochs


def overlap_intersect(epoch, interval, return_indices=True):
    """
    Returns the epochs with overlap with interval
    Input:
        epoch: nelpy.EpochArray
            The epochs to check
        interval: nelpy.IntervalArray
            The interval to check for overlap
        return_indices: bool
            If True, returns the indices of the epochs (interval) that overlap
    Output:
        epoch: nelpy.EpochArray
            The epochs with overlap with interval
    """
    new_intervals = []
    indices = []
    for epa in epoch:
        if any((interval.starts < epa.stop) & (interval.stops > epa.start)):
            new_intervals.append([epa.start, epa.stop])
            cand_ep_idx = np.where(
                (interval.starts < epa.stop) & (interval.stops > epa.start)
            )
            indices.append(cand_ep_idx[0][0])
    out = type(epoch)(new_intervals)
    out._domain = epoch.domain
    if return_indices:
        return out, indices
    return out


@jit(nopython=True)
def find_intersecting_intervals_(set1, set2):

    intersecting_intervals = []
    for i, (start1, end1) in enumerate(set1):
        # Check if any of the intervals in set2 intersect with the current interval in set1
        for start2, end2 in set2:
            if start2 <= end1 and end2 >= start1:
                # Calculate the amount of intersection between the two intervals
                intersection = min(end1, end2) - max(start1, start2)
                intersecting_intervals.append(intersection)
                break
        else:
            intersecting_intervals.append(0)  # No intersection found

    return intersecting_intervals


def find_intersecting_intervals(
    set1, set2, return_indices: bool = True
) -> Union[np.ndarray, List[bool]]:
    """
    Find the amount of time two sets of intervals are intersecting each other for each intersection.

    Parameters
    ----------
    set1 : nelpy EpochArray
    set2 : nelpy EpochArray
    return_indices : bool, optional (default=True)
        if True, return the indices of the intervals in set2 that intersect with each interval in set1.
        If False, return the amount of time each interval in set1 intersects with any interval in set2.

    Returns
    -------
    list
        A list of floats, where each float represents the amount of time the corresponding interval in set1 intersects with any interval in set2.

    Examples
    --------
    >>> set1 = nel.EpochArray([(1, 3), (5, 7), (9, 10)])
    >>> set2 = nel.EpochArray([(2, 4), (6, 8)])
    >>> find_intersecting_intervals(set1, set2)
    [True, True, False]
    >>> find_intersecting_intervals(set1, set2, return_indices=False)
    [1, 2, 0]
    """
    if not isinstance(set1, core.IntervalArray) & isinstance(set2, core.IntervalArray):
        raise ValueError("only EpochArrays are supported")

    intersection = np.array(find_intersecting_intervals_(set1.data, set2.data))
    if return_indices:
        return intersection > 0
    return intersection


def find_interval(logical):
    """
    Find consecutive intervals of True values in a list of boolean values.

    Parameters:
    logical (List[bool]): The list of boolean values.

    Returns:
    List[Tuple[int, int]]: A list of tuples representing the start and end indices of each consecutive interval of True values in the logical list.

    Example:
    find_interval([0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1]) -> [(2, 4), (6, 7), (10, 11)]
    find_interval([1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1]) -> [(0, 2), (4, 5), (9, 10)]
    """
    intervals = []
    start = None
    for i, value in enumerate(logical):
        if value and start is None:
            start = i
        elif not value and start is not None:
            intervals.append((start, i - 1))
            start = None
    if start is not None:
        intervals.append((start, len(logical) - 1))
    return intervals


@njit(parallel=True)
def in_intervals(timestamps: np.ndarray, intervals: np.ndarray) -> np.ndarray:
    """
    Find which timestamps fall within the given intervals.

    Parameters
    ----------
    timestamps : ndarray
        An array of timestamp values. Assumes sorted.
    intervals : ndarray
        An array of time intervals, represented as pairs of start and end times.

    Returns
    -------
    ndarray
        A logical index indicating which timestamps fall within the intervals.

    Examples
    --------
    >>> timestamps = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    >>> intervals = np.array([[2, 4], [5, 7]])
    >>> in_intervals(timestamps, intervals)
    array([False,  True,  True,  True,  True,  True,  True, False])
    """
    in_interval = np.zeros(timestamps.shape, dtype=np.bool_)
    for start, end in intervals:
        # Find the leftmost index of a timestamp that is >= start
        left = np.searchsorted(timestamps, start, side="left")
        if left == len(timestamps):
            # If start is greater than all timestamps, skip this interval
            continue
        # Find the rightmost index of a timestamp that is <= end
        right = np.searchsorted(timestamps, end, side="right")
        if right == left:
            # If there are no timestamps in the interval, skip it
            continue
        # Mark the timestamps in the interval
        in_interval[left:right] = True
    return in_interval


@jit(nopython=True, parallel=True)
def in_intervals_interval(timestamps: np.ndarray, intervals: np.ndarray) -> np.ndarray:
    """
    for each timestamps value, the index of the interval to which it belongs (nan = none)

    Parameters
    ----------
    timestamps : ndarray
        An array of timestamp values. assumes sorted
    intervals : ndarray
        An array of time intervals, represented as pairs of start and end times.

    Returns
    -------
    ndarray
        A ndarray indicating for each timestamps which interval it was within.

    Examples
    --------
    >>> timestamps = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    >>> intervals = np.array([[2, 4], [5, 7]])
    >>> in_intervals_interval(timestamps, intervals)
    array([nan,  0,  0,  0,  1,  1,  1, nan])
    """
    in_interval = np.full(timestamps.shape, np.nan)
    for i in numba.prange(intervals.shape[0]):
        start, end = intervals[i]
        mask = (timestamps >= start) & (timestamps <= end)
        in_interval[mask] = i

    return in_interval
