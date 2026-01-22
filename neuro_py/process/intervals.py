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
    preserve: bool = False,
) -> EpochArray:
    """Randomly shift intervals within a single or multi-interval support.

    This function applies either circular shifts (default) across the *combined* linearized
    support or segment-wise constrained shifts (`avoid_split`, `preserve` modes) to the intervals
    of an input :class:`nelpy.core.EpochArray`. Multi-interval supports (disjoint segments) are
    handled transparently and intervals may split when wrapping across gaps (except when
    constrained).

    Modes
    -----
    1. Default (no flags):
       Each interval (or the entire set) is circularly shifted along the concatenated support.
       Crossing a gap causes an interval to split, increasing interval count.
    2. ``avoid_split=True``:
       Intervals must remain wholly inside their originating segment. Picks per-interval or
       common shift from feasible ranges; raises if impossible for a common shift.
    3. ``preserve=True``:
       Guarantees identical interval count and total duration (duration of union of intervals).
       Intervals spanning multiple support segments are left unchanged. Intervals within a single
       segment are redistributed *without overlap* by randomly allocating gap space (Dirichlet)
       between them. Ordering inside a segment may change if ``randomize_each=True``.

    Parameter semantics
    -------------------
    ``randomize_each`` controls independence of shifts:
    - True: Each interval gets its own shift (or its own gap positioning in ``preserve`` mode).
    - False: One common circular shift (default mode) OR one feasible per-segment shift
      (``preserve=False, avoid_split=True``) OR segment-wise common shift with fallback 0
      (``preserve=True``).

    Support resolution order:
    1. Explicit ``start_stop`` argument (single [[start, stop]] or (n,2) array / EpochArray)
    2. Finite ``epoch.domain`` if present
    3. Fallback interval ``[epoch.start, epoch.stop]``

    Guarantees
    ----------
    - ``preserve=True`` keeps: interval count, total duration (sum of lengths), and prevents overlap
      within each support segment (except for spanning intervals which remain unchanged).
    - ``avoid_split=True`` keeps interval count unless your original intervals can’t all share a common
      shift when ``randomize_each=False`` (then a ValueError is raised).
    - All output intervals lie inside the provided/inferred support.

    Performance Notes
    -----------------
    Complexity is O(n + k) where n = number of intervals, k = support segments.
    Splitting in default mode can increase the number of intervals (worst-case doubling if each wraps once).

    Parameters
    ----------
    epoch : EpochArray
        Intervals to randomize. Must already lie within the chosen support.
    randomize_each : bool, default True
        Independent per-interval shifts (or per-interval gap placement in ``preserve`` mode).
        Set False for a single common shift or per-segment collective shift.
    start_stop : (2,) or (n,2) array-like | EpochArray | IntervalArray, optional
        Support definition. If None, uses finite ``epoch.domain`` when available else ``[epoch.start, epoch.stop]``.
    avoid_split : bool, default False
        Constrain shifts to remain inside original segment. May raise for infeasible common shift.
    preserve : bool, default False
        Redistribute intervals segment-wise without overlap, preserving count & total duration.

    Returns
    -------
    EpochArray
        Randomized (and possibly split or redistributed) intervals.

    Raises
    ------
    ValueError
        - Support intervals malformed (shape/ordering)
        - Interval outside support
        - ``avoid_split=True`` and interval spans multiple segments
        - ``avoid_split=True`` with ``randomize_each=False`` and no common feasible shift

    Examples
    --------
    Basic single interval support:
    >>> import nelpy as nel
    >>> from neuro_py.process.intervals import randomize_epochs
    >>> epochs = nel.EpochArray([[2.0, 4.0], [6.0, 7.0]])
    >>> out = randomize_epochs(epochs, randomize_each=False)
    >>> out.n_intervals in (2, 3)  # may split if wrapping
    True

    Multi-interval support supplied explicitly:
    >>> support = np.array([[0.0, 5.0], [10.0, 15.0]])
    >>> out2 = randomize_epochs(epochs, start_stop=support, randomize_each=True)
    >>> out2.domain.n_intervals == 2
    True

    Using domain if present:
    >>> domain = nel.EpochArray([[0.0, 5.0], [10.0, 12.0]])
    >>> epochs2 = nel.EpochArray([[1.0, 2.0], [10.5, 11.0]], domain=domain)
    >>> out3 = randomize_epochs(epochs2, randomize_each=False)
    >>> np.allclose(out3.domain.data, domain.data)
    True

    Avoid splitting intervals across gaps:
    >>> epochs3 = nel.EpochArray([[1.0, 2.0], [10.5, 11.0]], domain=domain)
    >>> out4 = randomize_epochs(epochs3, avoid_split=True, randomize_each=True)
    >>> out4.n_intervals == epochs3.n_intervals
    True

    Preserve count and total duration (non-overlapping redistribution):
    >>> preserved = randomize_epochs(epochs3, preserve=True)
    >>> np.isclose(preserved.duration, epochs3.duration)
    True
    >>> preserved.n_intervals == epochs3.n_intervals
    True

    Common shift with avoid_split (may raise if infeasible):
    >>> try:
    ...     _ = randomize_epochs(epochs3, avoid_split=True, randomize_each=False)
    ...     ok = True
    ... except ValueError:
    ...     ok = False
    >>> ok in (True, False)
    True

    Notes
    -----
    - For reproducibility set a global NumPy seed before calling (preserve mode uses a fresh RNG).
    - Intervals spanning multiple segments are never moved in ``preserve`` mode.
    - Setting ``preserve=True`` with ``randomize_each=False`` keeps relative ordering inside each segment.
    """

    # --- support normalization ---
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
        """Resolve support definition.

        Preference order:
        1. Explicit start_stop argument
        2. Finite epoch.domain if available (multi or single interval)
        3. Fallback to [epoch.start, epoch.stop]
        """
        if ss is None:
            use_domain = (
                hasattr(src_epoch, "domain")
                and src_epoch.domain.n_intervals > 0
                and np.isfinite(src_epoch.domain.start)
                and np.isfinite(src_epoch.domain.stop)
            )
            if use_domain:
                support_ep = nel.EpochArray(src_epoch.domain.data.copy())
            else:
                support_ep = nel.EpochArray([[src_epoch.start, src_epoch.stop]])
        else:
            if isinstance(ss, (nel.EpochArray, core.IntervalArray)):
                support_ep = nel.EpochArray(np.asarray(ss.data).copy())
            else:
                arr = np.asarray(ss, dtype=float)
                if arr.ndim == 1 and arr.shape[0] == 2:
                    arr = arr.reshape(1, 2)
                if arr.ndim != 2 or arr.shape[1] != 2:
                    raise ValueError("start_stop must be shape (2,) or (n,2)")
                support_ep = nel.EpochArray(arr)
        support_ep._sort()
        return support_ep.data.copy(), support_ep

    support_intervals, support_epoch = _to_support_intervals(start_stop, epoch)
    if support_intervals.size == 0:
        return epoch.copy()

    seg_lengths = support_intervals[:, 1] - support_intervals[:, 0]
    if np.any(seg_lengths < 0):
        raise ValueError("Support intervals must have start <= stop")
    total_T = float(np.sum(seg_lengths))
    if total_T <= 0:
        return epoch.copy()

    lin_ends = np.cumsum(seg_lengths)
    lin_starts = lin_ends - seg_lengths

    def _segment_index(t: float) -> int:
        idx = np.searchsorted(support_intervals[:, 0], t, side="right") - 1
        if idx < 0 or t > support_intervals[idx, 1]:
            raise ValueError("Timestamp outside support intervals")
        return int(idx)

    def abs_to_lin(t: float) -> float:
        j = _segment_index(t)
        return float(lin_starts[j] + (t - support_intervals[j, 0]))

    def lin_to_abs(a: float, b: float) -> List[Tuple[float, float]]:
        out: List[Tuple[float, float]] = []
        for j in range(support_intervals.shape[0]):
            ls, le = lin_starts[j], lin_ends[j]
            s = max(a, ls)
            e = min(b, le)
            if e > s:
                abs_s = support_intervals[j, 0] + (s - ls)
                abs_e = support_intervals[j, 0] + (e - ls)
                out.append((float(abs_s), float(abs_e)))
        return out

    # --- preserve mode (non-overlapping, keep total union duration & count) ---
    if preserve:
        # Group intervals by segment
        segment_map: dict[int, List[Tuple[float, float]]] = {}
        spanning: List[
            Tuple[float, float]
        ] = []  # intervals crossing segments (kept unchanged)
        for a, b in epoch.data:
            j_a = _segment_index(float(a))
            j_b = _segment_index(float(b))
            if j_a != j_b:
                spanning.append((float(a), float(b)))
            else:
                segment_map.setdefault(j_a, []).append((float(a), float(b)))

        out: List[Tuple[float, float]] = []
        rng = np.random.default_rng()
        for j, intervals_j in segment_map.items():
            seg_start, seg_stop = support_intervals[j]
            seg_len = seg_stop - seg_start
            # lengths
            lengths = [b - a for (a, b) in intervals_j]
            total_len = float(np.sum(lengths))
            slack = seg_len - total_len
            if slack <= 0:
                # No room: keep ordering, anchored to segment start
                cur = seg_start
                for L in lengths:
                    out.append((float(cur), float(cur + L)))
                    cur += L
                continue
            # Generate gap allocations (g_0 .. g_n) summing to slack
            n = len(lengths)
            # Dirichlet for gaps ensures non-negative
            gaps = rng.dirichlet(alpha=np.ones(n + 1)) * slack
            # Optionally shuffle intervals if randomize_each True
            if randomize_each and n > 1:
                order = rng.permutation(n)
                lengths_shuffled = [lengths[i] for i in order]
            else:
                lengths_shuffled = lengths
            # Build new positions
            pos = seg_start + gaps[0]
            for idx, L in enumerate(lengths_shuffled):
                start_new = float(pos)
                stop_new = float(pos + L)
                out.append((start_new, stop_new))
                pos = stop_new + gaps[idx + 1]

        # Add spanning intervals unchanged
        out.extend(spanning)

        new_epochs = nel.EpochArray(np.asarray(out))
        new_epochs._sort()
        new_epochs._domain = support_epoch.domain
        new_epochs._domain._data = support_intervals.copy()
        return new_epochs

    # --- avoid_split mode ---
    if avoid_split:
        ranges: List[Tuple[float, float]] = []
        for a, b in epoch.data:
            j_a = _segment_index(float(a))
            j_b = _segment_index(float(b))
            if j_a != j_b:
                raise ValueError(
                    "avoid_split=True requires each interval lie within a single support segment"
                )
            seg_start, seg_stop = support_intervals[j_a]
            low = seg_start - a
            high = seg_stop - b
            if high <= low:
                raise ValueError(
                    "Interval has no valid shift range under avoid_split constraint"
                )
            ranges.append((low, high))
        if randomize_each:
            shifts = np.array([float(np.random.uniform(l, h)) for (l, h) in ranges])
        else:
            low_c = max(l for (l, _h) in ranges)
            high_c = min(h for (_l, h) in ranges)
            if high_c <= low_c:
                raise ValueError(
                    "No common shift satisfying all intervals under avoid_split"
                )
            s = float(np.random.uniform(low_c, high_c))
            shifts = np.full(epoch.n_intervals, s, dtype=float)
        out = [(a + s, b + s) for (a, b), s in zip(epoch.data, shifts)]
        new_epochs = nel.EpochArray(np.asarray(out))
        new_epochs._sort()
        new_epochs._domain = support_epoch.domain
        new_epochs._domain._data = support_intervals.copy()
        return new_epochs

    # --- unconstrained circular shift over combined support ---
    if randomize_each:
        shifts = np.random.uniform(0.0, total_T, size=epoch.n_intervals)
    else:
        s = float(np.random.uniform(0.0, total_T))
        shifts = np.full(epoch.n_intervals, s, dtype=float)

    out: List[Tuple[float, float]] = []
    for i, (a, b) in enumerate(epoch.data):
        a_lin = abs_to_lin(float(a))
        b_lin = abs_to_lin(float(b))
        da = (a_lin + shifts[i]) % total_T
        db = (b_lin + shifts[i]) % total_T
        lin_ranges = [(da, db)] if da <= db else [(da, total_T), (0.0, db)]
        for lr_a, lr_b in lin_ranges:
            out.extend(lin_to_abs(lr_a, lr_b))

    new_epochs = nel.EpochArray(np.asarray(out))
    new_epochs._sort()
    new_epochs._domain = support_epoch.domain
    new_epochs._domain._data = support_intervals.copy()
    return new_epochs


# NOTE: previous experimental function `randomize_epochs_new` has been merged into `randomize_epochs`.


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
