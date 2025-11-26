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
    start_stop: Optional[
        Union[
            np.ndarray,
            nel.EpochArray,
            core.IntervalArray,
            List[List[float]],
            Tuple[float, float],
        ]
    ] = None,
    avoid_split: bool = False,
) -> EpochArray:
    """
    Randomly shift epochs and wrap them within a time support that can be a single interval or
    multiple disjoint intervals (multi-interval support).

    This takes an EpochArray and circularly shifts the intervals along the given support while
    preserving gaps. If the support has multiple intervals, shifted epochs that cross support
    boundaries will be split to respect the gaps.

    Parameters
    ----------
    epoch : EpochArray
        The EpochArray whose intervals will be shifted.
    randomize_each : bool, optional
        If True, each epoch will be shifted by an independent random amount.
        If False, all epochs will be shifted by the same random amount. Defaults to True.
    start_stop : array-like | EpochArray | IntervalArray, optional
        If provided, defines the time support used for circular wrapping. Accepted forms:
        - [start, stop] (single interval)
        - shape (n, 2) array/list of intervals
        - nelpy.EpochArray or nelpy.core.IntervalArray
        If not provided, the support is taken from epoch.domain when finite; otherwise
        [epoch.start, epoch.stop].
    avoid_split : bool, optional
        If True, constrain the random shift so that each interval remains within one
        support segment (i.e., does not split across gaps). When randomize_each=False,
        a common shift will be chosen that satisfies this constraint for all intervals
        if possible; otherwise a ValueError is raised. Defaults to False.

    Returns
    -------
    EpochArray
        A new EpochArray with shifted intervals, wrapped to the specified support.

    Notes
    -----
    - With multi-interval support, an original epoch may map to multiple output intervals
      if the shift causes it to cross support boundaries. In that case, the output can have
      more intervals than the input.

    Examples
    --------
    Basic single-interval wrapping:
    >>> epoch = nel.EpochArray([(2.0, 4.0), (6.0, 7.0)])
    >>> out = randomize_epochs(epoch, randomize_each=False, start_stop=[0.0, 10.0])
    >>> isinstance(out, nel.EpochArray)
    True

    Multi-interval support (disjoint segments):
    >>> support = np.array([[0.0, 5.0], [10.0, 15.0]])
    >>> epoch = nel.EpochArray([(13.5, 14.7)])
    >>> out = randomize_epochs(epoch, randomize_each=False, start_stop=support)
    # output may split if the shifted interval crosses a gap between segments
    >>> out.n_intervals >= 1
    True

    Use epoch.domain as support (when defined):
    >>> domain = nel.EpochArray([[0.0, 5.0], [10.0, 12.0]])
    >>> epoch = nel.EpochArray([(1.0, 2.0), (10.5, 11.0)], domain=domain)
    >>> out = randomize_epochs(epoch)
    >>> np.allclose(out.domain.data, domain.data)
    True

    Preserve interval count with avoid_split=True:
    - Per-interval shifts
    >>> support = np.array([[0.0, 5.0], [10.0, 15.0]])
    >>> epoch = nel.EpochArray([(1.0, 2.0), (12.0, 13.0)])
    >>> out = randomize_epochs(epoch, randomize_each=True, start_stop=support, avoid_split=True)
    >>> out.n_intervals == epoch.n_intervals
    True

    - Common shift (randomize_each=False) may raise if impossible for all intervals:
    >>> support = np.array([[0.0, 2.0], [5.0, 7.0]])
    >>> epoch = nel.EpochArray([(1.95, 2.0), (5.0, 6.8)])
    >>> try:
    ...     _ = randomize_epochs(epoch, randomize_each=False, start_stop=support, avoid_split=True)
    ...     ok = True
    ... except ValueError:
    ...     ok = False
    >>> ok in (True, False)
    True
    """

    # --- prepare support intervals (as np.ndarray of shape (k,2)) ---
    def _to_support_intervals(
        ss: Optional[
            Union[
                np.ndarray,
                nel.EpochArray,
                core.IntervalArray,
                List[List[float]],
                Tuple[float, float],
            ]
        ],
        src_epoch: EpochArray,
    ) -> Tuple[np.ndarray, nel.EpochArray]:
        if ss is None:
            # Prefer finite domain if available; otherwise use [start, stop] of the data
            if (
                hasattr(src_epoch, "domain")
                and (
                    np.isfinite(src_epoch.domain.start)
                    or np.isfinite(src_epoch.domain.stop)
                )
                and src_epoch.domain.n_intervals > 0
            ):
                support_ep = nel.EpochArray(src_epoch.domain.data.copy())
            else:
                support_ep = nel.EpochArray(
                    np.array([[src_epoch.start, src_epoch.stop]])
                )
        else:
            if isinstance(ss, (nel.EpochArray, core.IntervalArray)):
                support_ep = nel.EpochArray(np.asarray(ss.data).copy())
            else:
                arr = np.asarray(ss, dtype=float)
                if arr.ndim == 1 and arr.shape[0] == 2:
                    arr = arr.reshape(1, 2)
                if arr.ndim != 2 or arr.shape[1] != 2:
                    raise ValueError(
                        "start_stop must be of shape (2,) or (n,2) or an EpochArray/IntervalArray"
                    )
                support_ep = nel.EpochArray(arr.copy())

        # normalize/sort
        support_ep._sort()
        return support_ep.data.copy(), support_ep

    support_intervals, support_epoch = _to_support_intervals(start_stop, epoch)

    if support_intervals.size == 0:
        return epoch.copy()

    # total support duration
    seg_lengths = support_intervals[:, 1] - support_intervals[:, 0]
    if np.any(seg_lengths < 0):
        raise ValueError("Support intervals must have start <= stop")
    total_T = np.sum(seg_lengths)
    if total_T <= 0:
        return epoch.copy()

    # linearized coordinates for support segments
    lin_ends = np.cumsum(seg_lengths)
    lin_starts = lin_ends - seg_lengths

    # helper: absolute -> linear (assumes t is within exactly one support segment)
    def abs_to_lin(t: float) -> float:
        # find segment containing t
        # using boolean mask; segments are non-overlapping and sorted
        idx = np.searchsorted(support_intervals[:, 0], t, side="right") - 1
        if idx < 0 or t > support_intervals[idx, 1]:
            # t outside support; raise to catch unexpected inputs
            raise ValueError("Timestamp outside support intervals")
        return float(lin_starts[idx] + (t - support_intervals[idx, 0]))

    # helper: map a linear interval [a,b] (a<b in [0, total_T]) back to absolute intervals
    def lin_to_abs_interval(a: float, b: float) -> List[Tuple[float, float]]:
        out: List[Tuple[float, float]] = []
        # iterate support segments and intersect
        for j in range(support_intervals.shape[0]):
            ls, le = lin_starts[j], lin_ends[j]
            # overlap in linearized space
            s = max(a, ls)
            e = min(b, le)
            if e > s:
                # map back to absolute
                abs_s = support_intervals[j, 0] + (s - ls)
                abs_e = support_intervals[j, 0] + (e - ls)
                out.append((float(abs_s), float(abs_e)))
        return out

    # utility to find segment index for absolute timestamp
    def _abs_segment_index(t: float) -> int:
        idx = np.searchsorted(support_intervals[:, 0], t, side="right") - 1
        if idx < 0 or t > support_intervals[idx, 1]:
            raise ValueError("Timestamp outside support intervals")
        return int(idx)

    # choose shift(s)
    if avoid_split and support_intervals.shape[0] > 1:
        # Constrain shifts so each interval stays within its original segment.
        # For an interval with linear [a_lin, b_lin] in segment j with [ls, le],
        # we need shift such that (a_lin+shift) and (b_lin+shift) both lie in [ls, le) without wrap.
        # That implies shift in [ls - a_lin, le - b_lin).
        per_ranges = []  # list of (low, high) ranges in linear coords
        a_lins = []
        b_lins = []
        seg_idx = []
        for a_abs, b_abs in epoch.data:
            j = _abs_segment_index(float(a_abs))
            # ensure both endpoints are within same segment
            j_b = _abs_segment_index(float(b_abs))
            if j != j_b:
                # The original interval itself spans segments; to preserve count without splitting,
                # there is no valid shift that keeps it in a single segment.
                raise ValueError(
                    "avoid_split=True requires each input interval to lie within a single support segment."
                )
            a_lin = abs_to_lin(float(a_abs))
            b_lin = abs_to_lin(float(b_abs))
            low = lin_starts[j] - a_lin
            high = lin_ends[j] - b_lin
            per_ranges.append((low, high))
            a_lins.append(a_lin)
            b_lins.append(b_lin)
            seg_idx.append(j)

        if randomize_each:
            # sample independently within each valid range
            shifts = np.zeros(epoch.n_intervals, dtype=float)
            for i, (low, high) in enumerate(per_ranges):
                if not np.isfinite(low) or not np.isfinite(high) or high <= low:
                    raise ValueError(
                        "No valid shift range to avoid split for an interval."
                    )
                s = float(np.random.uniform(low, high))
                shifts[i] = s
        else:
            # find a common shift satisfying all ranges via intersection
            low_common = -np.inf
            high_common = np.inf
            for low, high in per_ranges:
                low_common = max(low_common, low)
                high_common = min(high_common, high)
            if (
                not np.isfinite(low_common)
                or not np.isfinite(high_common)
                or high_common <= low_common
            ):
                raise ValueError(
                    "avoid_split=True with randomize_each=False: no common shift keeps all intervals within a single segment."
                )
            s = float(np.random.uniform(low_common, high_common))
            shifts = np.full(epoch.n_intervals, s, dtype=float)
        # Note: these shifts are absolute linear offsets (not modulo total_T). They keep intervals inside segments.
    else:
        # unconstrained circular shift over total support
        if randomize_each:
            shifts = np.random.uniform(0.0, total_T, size=epoch.n_intervals)
        else:
            s = float(np.random.uniform(0.0, total_T))
            shifts = np.full(epoch.n_intervals, s, dtype=float)

    out_intervals: List[Tuple[float, float]] = []

    # process each epoch interval
    for i, (a_abs, b_abs) in enumerate(epoch.data):
        # clip/restrict expectation: intervals should be within support
        # If any endpoint falls outside support, raise with informative message
        try:
            a_lin = abs_to_lin(float(a_abs))
            b_lin = abs_to_lin(float(b_abs))
        except ValueError:
            raise ValueError(
                "All epoch intervals must lie within the specified support intervals."
            )

        # maintain original duration in linearized space
        # apply shift with wrap-around
        if avoid_split and support_intervals.shape[0] > 1:
            # constrained shifts are linear offsets meant to keep within segment; do not mod total_T
            da = a_lin + shifts[i]
            db = b_lin + shifts[i]
        else:
            # unconstrained circular shift
            da = (a_lin + shifts[i]) % total_T
            db = (b_lin + shifts[i]) % total_T

        if da <= db:
            lin_ranges = [(da, db)]
        else:
            # wrapped around end -> split into two linear ranges
            lin_ranges = [(da, total_T), (0.0, db)]

        # map linear ranges back to absolute space (respect multi-interval support)
        for lr_start, lr_stop in lin_ranges:
            out_intervals.extend(lin_to_abs_interval(lr_start, lr_stop))

    # build output EpochArray and set domain to support
    if len(out_intervals) == 0:
        return nel.EpochArray([])

    new_epochs = nel.EpochArray(np.asarray(out_intervals))
    # normalize and attach domain/support used for wrapping
    new_epochs._sort()
    new_epochs._domain = (
        support_epoch.domain
    )  # inherit same type; support_epoch has exact data
    new_epochs._domain._data = support_intervals.copy()

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
