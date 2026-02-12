import random
from typing import List, Optional, Tuple, Union

import nelpy as nel
import numba
import numpy as np
from nelpy import core
from nelpy.core import EpochArray
from numba import jit


def randomize_epochs(
    epoch: EpochArray,
    randomize_each: bool = True,
    start_stop: Optional[np.ndarray] = None,
) -> EpochArray:
    """
    Randomly shifts the epochs of a EpochArray object and wraps them around the original time boundaries.

    This method takes a EpochArray object as input, and can either randomly shift each epoch by a different amount
    (if `randomize_each` is True) or shift all the epochs by the same amount (if `randomize_each` is False).
    In either case, the method wraps the shifted epochs around the original time boundaries to make sure they remain
    within the original time range. It then returns the modified EpochArray object.

    Parameters
    ----------
    epoch : EpochArray
        The EpochArray object whose epochs should be shifted and wrapped.
    randomize_each : bool, optional
        If True, each epoch will be shifted by a different random amount.
        If False, all the epochs will be shifted by the same random amount. Defaults to True.
    start_stop : array, optional
        If not None, time support will be taken from start_stop

    Returns
    -------
    new_epochs : EpochArray
        The modified EpochArray object with the shifted and wrapped epochs.
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
        random_order = random.choices(
            range(-int(ts_range), int(ts_range)), k=new_epochs.n_intervals
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


def split_epoch_by_width(
    intervals: List[Tuple[float, float]], bin_width: float = 0.001
) -> np.ndarray:
    """
    Generate combined intervals (start, stop) at a specified width within given intervals.

    Parameters
    ----------
    intervals : List[Tuple[float, float]]
        A list of (start, end) tuples representing intervals.
    bin_width : float
        The width of each bin in seconds. Default is 0.001 (1 ms).

    Returns
    -------
    np.ndarray
        A 2D array containing (start, stop) pairs for all bins across intervals.
    """
    bin_intervals = []
    for start, end in intervals:
        # Generate bin edges
        edges = np.arange(start, end, bin_width)
        edges = np.append(edges, end)  # Ensure the final end is included
        # Generate intervals (start, stop) for each bin
        intervals = np.stack((edges[:-1], edges[1:]), axis=1)
        bin_intervals.append(intervals)
    return np.vstack(bin_intervals)


def split_epoch_equal_parts(
    intervals: np.ndarray, n_parts: int, return_epoch_array: bool = True
) -> Union[np.ndarray, nel.EpochArray]:
    """
    Split multiple intervals into equal parts.

    Parameters
    ----------
    intervals : array-like, shape (n_intervals, 2)
        The intervals to split.
    n_parts : int
        The number of parts to split each interval into.
    return_epoch_array : bool, optional
        If True, returns the intervals as a nelpy.EpochArray object. Defaults to True.

    Returns
    -------
    split_intervals : array-like, shape (n_intervals * n_parts, 2) or nelpy.EpochArray
        The split intervals.
    """
    # Ensure intervals is a numpy array
    intervals = np.asarray(intervals)

    # Number of intervals
    n_intervals = intervals.shape[0]

    # Preallocate the output array
    split_intervals = np.zeros((n_intervals * n_parts, 2))

    for i, interval in enumerate(intervals):
        start, end = interval
        epoch_parts = np.linspace(start, end, n_parts + 1)
        epoch_parts = np.vstack((epoch_parts[:-1], epoch_parts[1:])).T
        split_intervals[i * n_parts : (i + 1) * n_parts] = epoch_parts

    if return_epoch_array:
        return nel.EpochArray(split_intervals)
    return split_intervals


def overlap_intersect(
    epoch: nel.EpochArray, interval: nel.IntervalArray, return_indices: bool = True
) -> Union[nel.EpochArray, Tuple[nel.EpochArray, np.ndarray]]:
    """
    Returns the epochs with overlap with the given interval.

    Parameters
    ----------
    epoch : nelpy.EpochArray
        The epochs to check.
    interval : nelpy.IntervalArray
        The interval to check for overlap.
    return_indices : bool, optional
        If True, returns the indices of the overlapping epochs. Default is True.

    Returns
    -------
    nelpy.EpochArray
        The epochs with overlap with the interval.
    Tuple[nelpy.EpochArray, np.ndarray], optional
        If `return_indices` is True, also returns the indices of the overlapping epochs.
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
def _find_intersecting_intervals(set1: np.ndarray, set2: np.ndarray) -> List[float]:
    """
    Find the amount of time two sets of intervals are intersecting each other for each interval in set1.

    Parameters
    ----------
    set1 : ndarray
        An array of intervals represented as pairs of start and end times.
    set2 : ndarray
        An array of intervals represented as pairs of start and end times.

    Returns
    -------
    list of float
        A list of floats, where each float represents the amount of time the
        corresponding interval in set1 intersects with any interval in set2.
    """
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
    set1: nel.EpochArray, set2: nel.EpochArray, return_indices: bool = True
) -> Union[np.ndarray, List[bool]]:
    """
    Find the amount of time two sets of intervals are intersecting each other for each intersection.

    Parameters
    ----------
    set1 : nelpy EpochArray
        The first set of intervals to check for intersections.
    set2 : nelpy EpochArray
        The second set of intervals to check for intersections.
    return_indices : bool, optional
        If True, return the indices of the intervals in set2 that intersect with each interval in set1.
        If False, return the amount of time each interval in set1 intersects with any interval in set2.

    Returns
    -------
    Union[np.ndarray, List[bool]]
        If return_indices is True, returns a boolean array indicating whether each interval in set1 intersects with any interval in set2.
        If return_indices is False, returns a NumPy array with the amount of time each interval in set1 intersects with any interval in set2.

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

    intersection = np.array(_find_intersecting_intervals(set1.data, set2.data))
    if return_indices:
        return intersection > 0
    return intersection


def find_intersection_intervals_strict(
    set1: nel.EpochArray, set2: nel.EpochArray
) -> nel.EpochArray:
    """
    Find the intervals in set1 that are completely contained within set2.

    Parameters
    ----------
    set1 : nelpy EpochArray
        The first set of intervals to check for intersections.
    set2 : nelpy EpochArray
        The second set of intervals to check for intersections.
    Returns
    -------
    nelpy EpochArray
        An EpochArray containing the intervals in set1 which are completely contained within set2.

    Examples
    --------
    >>> set1 = nel.EpochArray([(1, 3), (5, 7), (9, 10)])
    >>> set2 = nel.EpochArray([(0, 4), (6, 8)])
    >>> find_intersection_intervals_strict(set1, set2)
    EpochArray([[1, 3]])

    Notes
    -----
    Common use cases:
    - Finding theta cycles (set1) that are completely within running periods (set2)
    - Finding SWRs (set1) that are completely within NREM periods (set2)
    - Generally useful when you don't want partial overlaps created by intersection of set1 & set2
    """
    overlap = find_intersecting_intervals(set1, set2, return_indices=False)
    out = nel.EpochArray(set1.data[set1.lengths == overlap])
    out._domain = set1.domain
    return out


def find_interval(logical: List[bool]) -> List[Tuple[int, int]]:
    """
    Find consecutive intervals of True values in a list of boolean values.

    Parameters
    ----------
    logical : List[bool]
        The list of boolean values.

    Returns
    -------
    List[Tuple[int, int]]
        A list of tuples representing the start and end indices of each consecutive interval of True values in the logical list.

    Examples
    --------
    >>> find_interval([0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1])
    [(2, 4), (6, 7), (10, 11)]
    >>> find_interval([1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1])
    [(0, 2), (4, 5), (9, 10)]
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


# @njit(parallel=True)
def in_intervals(
    timestamps: np.ndarray,
    intervals: np.ndarray,
    return_interval: bool = False,
    shift: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]:
    """
    Find which timestamps fall within the given intervals.

    Parameters
    ----------
    timestamps : ndarray
        An array of timestamp values. Assumes sorted.
    intervals : ndarray
        An array of time intervals, represented as pairs of start and end times.
    return_interval : bool, optional (default=False)
        If True, return the index of the interval to which each timestamp belongs.
    shift : bool, optional (default=False)
        If True, return the shifted timestamps

    Returns
    -------
    in_interval : ndarray
        A logical index indicating which timestamps fall within the intervals.
    interval : ndarray, optional
        A ndarray indicating for each timestamps which interval it was within.
    shifted_timestamps : ndarray, optional
        The shifted timestamps

    Examples
    --------
    >>> timestamps = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    >>> intervals = np.array([[2, 4], [5, 7]])
    >>> in_intervals(timestamps, intervals)
    array([False,  True,  True,  True,  True,  True,  True, False])

    >>> in_intervals(timestamps, intervals, return_interval=True)
    (array([False,  True,  True,  True,  True,  True,  True, False]),
    array([nan,  0.,  0.,  0.,  1.,  1.,  1., nan]))

    >>> in_intervals(timestamps, intervals, shift=True)
    (array([False,  True,  True,  True,  True,  True,  True, False]),
    array([0, 1, 2, 2, 3, 4]))

    >>> in_intervals(timestamps, intervals, return_interval=True, shift=True)
    (array([False,  True,  True,  True,  True,  True,  True, False]),
    array([0, 0, 0, 1, 1, 1]),
    array([0, 1, 2, 2, 3, 4]))
    """
    in_interval = np.zeros(timestamps.shape, dtype=np.bool_)
    interval = np.full(timestamps.shape, np.nan)

    for i, (start, end) in enumerate(intervals):
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
        interval[left:right] = i

    if shift:
        # Restrict to the timestamps that fall within the intervals
        interval = interval[in_interval].astype(int)

        # Calculate shifts based on intervals
        shifts = np.insert(np.cumsum(intervals[1:, 0] - intervals[:-1, 1]), 0, 0)[
            interval
        ]

        # Apply shifts to timestamps
        shifted_timestamps = timestamps[in_interval] - shifts - intervals[0, 0]

    if return_interval and shift:
        return in_interval, interval, shifted_timestamps

    if return_interval:
        return in_interval, interval

    if shift:
        return in_interval, shifted_timestamps

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

    Note: produces same result as in_intervals with return_interval=True

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


def truncate_epoch(
    epoch: nel.EpochArray, time: Union[int, float] = 3600
) -> nel.EpochArray:
    """
    Truncates an EpochArray to achieve a specified cumulative time duration.

    This function takes an input EpochArray 'epoch' and a 'time' value representing
    the desired cumulative time duration in seconds. It returns a new EpochArray
    containing intervals that cumulatively match the specified time.

    Parameters
    ----------
    epoch : nel.EpochArray
        The input EpochArray containing intervals to be truncated.
    time : Union[int, float], optional
        The desired cumulative time in seconds (default is 3600).

    Returns
    -------
    nel.EpochArray
        A new EpochArray containing intervals that cumulatively match
        the specified time.

    Algorithm
    ---------
    1. Calculate the cumulative lengths of intervals in the 'epoch'.
    2. If the cumulative time of the 'epoch' is already less than or equal to 'time',
        return the original 'epoch'.
    3. Find the last interval that fits within the specified 'time' and create a new EpochArray
        'truncated_intervals' with intervals up to that point.
    4. To achieve the desired cumulative time, calculate the remaining time needed to reach 'time'.
    5. Add portions of the next interval to 'truncated_intervals' until the desired 'time' is reached
        or all intervals are used.

    Examples
    --------
    >>> epoch_data = [(0, 2), (3, 6), (8, 10)]
    >>> epoch = nel.EpochArray(epoch_data)
    >>> truncated_epoch = truncate_epoch(epoch, time=7)
    """

    if epoch.isempty:
        return epoch

    # calcuate cumulative lengths
    cumulative_lengths = epoch.lengths.cumsum()

    # No truncation needed
    if cumulative_lengths[-1] <= time:
        return epoch

    # Find the last interval that fits within the time and make new epoch
    idx = cumulative_lengths <= time
    truncated_intervals = nel.EpochArray(epoch.data[idx])

    # It's unlikely that the last interval will fit perfectly, so add the remainder from the next interval
    #   until the epoch is the desired length
    interval_i = 0
    while (time - truncated_intervals.duration) > 1e-10 or interval_i > len(epoch):
        # Add the last interval
        next_interval = int(np.where(cumulative_lengths >= time)[0][interval_i])

        remainder = (
            nel.EpochArray(
                [
                    epoch[next_interval].start,
                    epoch[next_interval].start + (time - truncated_intervals.duration),
                ]
            )
            & epoch[next_interval]
        )
        truncated_intervals = truncated_intervals | remainder
        interval_i += 1

    return truncated_intervals


def shift_epoch_array(
    epoch: nel.EpochArray, epoch_shift: nel.EpochArray
) -> nel.EpochArray:
    """
    Shift an EpochArray by another EpochArray.

    Shifting means that intervals in 'epoch' will be relative to
    intervals in 'epoch_shift' as if 'epoch_shift' intervals were without gaps.

    Parameters
    ----------
    epoch : nel.EpochArray
        The intervals to shift.
    epoch_shift : nel.EpochArray
        The intervals to shift by.

    Returns
    -------
    nel.EpochArray
        The shifted EpochArray.

    Notes
    -----
    This function restricts 'epoch' to those within 'epoch_shift' as
    epochs between 'epoch_shift' intervals would result in a duration of 0.

    Visual representation:
    inputs:
        epoch       =   [  ]   [  ] [  ]  []
        epoch_shift =   [    ] [    ]   [    ]
    becomes:
        epoch       =   [  ]  [  ]    []
        epoch_shift =   [    ][    ][    ]
    """
    # input validation
    if not isinstance(epoch, nel.EpochArray):
        raise TypeError("epoch must be a nelpy EpochArray")
    if not isinstance(epoch_shift, nel.EpochArray):
        raise TypeError("epoch_shift must be a nelpy EpochArray")

    # restrict epoch to epoch_shift and extract starts and stops
    epoch_starts, epoch_stops = epoch[epoch_shift].data.T

    # shift starts and stops by epoch_shift
    _, epoch_starts_shifted = in_intervals(epoch_starts, epoch_shift.data, shift=True)
    _, epoch_stops_shifted = in_intervals(epoch_stops, epoch_shift.data, shift=True)

    # shift time support as well, if one exists
    support_starts_shifted, support_stops_shifted = -np.inf, np.inf
    if epoch.domain.start != -np.inf:
        _, support_starts_shifted = in_intervals(
            epoch.domain.start, epoch_shift.data, shift=True
        )
    if epoch.domain.stop != np.inf:
        _, support_stops_shifted = in_intervals(
            epoch.domain.stop, epoch_shift.data, shift=True
        )

    session_domain = nel.EpochArray([support_starts_shifted, support_stops_shifted])

    # package shifted intervals into epoch array with shifted time support
    return nel.EpochArray(
        np.array([epoch_starts_shifted, epoch_stops_shifted]).T, domain=session_domain
    )


def get_overlapping_intervals(
    start: float, stop: float, interval_width: float, slideby: float
) -> np.ndarray:
    """
    Generate overlapping intervals within a specified time range.

    Parameters
    ----------
    start : float
        The start time of the time range.
    stop : float
        The stop time of the time range.
    interval_width : float
        The width of each interval in seconds.
    slideby : float
        The amount to slide the interval by in seconds.

    Returns
    -------
    np.ndarray
        A 2D array containing (start, stop) pairs for all overlapping intervals.

    Examples
    --------
    >>> get_overlapping_intervals(0, 10, 2, 1)
    array([[0, 2],
        [1, 3],
        [2, 4],
        [3, 5],
        [4, 6],
        [5, 7],
        [6, 8],
        [7, 9]])
    """
    starts = np.arange(start, stop - interval_width, slideby)
    stops = starts + interval_width
    return np.column_stack((starts, stops))
