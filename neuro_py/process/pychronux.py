import warnings
from typing import Any, Iterator, List, NamedTuple, Optional, Sequence, Tuple, Union

import nelpy as nel
import numpy as np
import pandas as pd
from numba import njit
from numpy.typing import NDArray
from scipy.signal.windows import dpss


_TimeSupport = Sequence[float] | Sequence[Sequence[float]] | NDArray[np.number[Any]]


def getfgrid(
    Fs: int, nfft: int, fpass: List[float]
) -> Tuple[NDArray[Any], NDArray[Any]]:
    """
    Get frequency grid for evaluation.

    Parameters
    ----------
    Fs : int
        Sampling frequency.
    nfft : int
        Number of points for FFT.
    fpass : List[float]
        Frequency range to evaluate (as [fmin, fmax]).

    Returns
    -------
    f : np.ndarray
        Frequency vector within the specified range.
    findx : np.ndarray
        Boolean array indicating the indices of the frequency vector that fall within the specified range.

    Notes
    -----
    The frequency vector is computed based on the sampling frequency and the number of FFT points.
    Only frequencies within the range defined by `fpass` are returned.

    Examples
    --------
    >>> from neuro_py.process import pychronux as px
    >>> frequencies, mask = px.getfgrid(128, 128, [6, 10])
    >>> frequencies.tolist()
    [6.0, 7.0, 8.0, 9.0, 10.0]
    """
    if Fs <= 0 or nfft < 1:
        raise ValueError("Fs and nfft must be positive")
    f = np.arange(nfft) * (Fs / nfft)
    findx = (f >= fpass[0]) & (f <= fpass[-1])
    f = f[findx]
    return f, findx


def dpsschk(
    tapers: Union[NDArray[Any], Tuple[float, int]], N: int, Fs: float
) -> NDArray[Any]:
    """
    Check and generate DPSS tapers.

    Parameters
    ----------
    tapers : Union[np.ndarray, Tuple[float, int]]
        Time-bandwidth product NW and taper count K.
    N : int
        Number of observed samples.
    Fs : float
        Sampling frequency.

    Returns
    -------
    tapers : np.ndarray
        Scaled tapers with shape (N, K).

    Notes
    -----
    The function computes symmetric DPSS (Discrete Prolate Spheroidal
    Sequences) tapers, as in Chronux, and scales them by ``sqrt(Fs)``.

    Examples
    --------
    >>> from neuro_py.process import pychronux as px
    >>> tapers = px.dpsschk((3, 5), N=128, Fs=128)
    >>> tapers.shape  # samples by tapers
    (128, 5)
    """
    return dpss(N, NW=tapers[0], Kmax=tapers[1], sym=True).T * np.sqrt(Fs)


def get_tapers(
    N: int,
    bandwidth: float,
    *,
    fs: float = 1.0,
    min_lambda: float = 0.95,
    n_tapers: Optional[int] = None,
) -> Tuple[NDArray[Any], NDArray[Any]]:
    """
    Compute tapers and associated energy concentrations for the Thomson
    multitaper method.

    Parameters
    ----------
    N : int
        Length of taper.
    bandwidth : float
        Half-bandwidth of the taper, in Hz (NW = bandwidth * N / fs).
    fs : float, optional
        Sampling rate, in Hz. Default is 1 Hz.
    min_lambda : float, optional
        Minimum energy concentration that each taper must satisfy. Default is 0.95.
    n_tapers : Optional[int], optional
        Number of tapers to compute. Default is to use all tapers that satisfy 'min_lambda'.

    Returns
    -------
    tapers : np.ndarray
        Array of tapers with shape (n_tapers, N).
    lambdas : np.ndarray
        Energy concentrations for each taper with shape (n_tapers,).

    Raises
    ------
    ValueError
        If not enough tapers are available or if none of the tapers satisfy the
        minimum energy concentration criteria.

    Examples
    --------
    >>> from neuro_py.process import pychronux as px
    >>> tapers, concentration = px.get_tapers(128, bandwidth=3, fs=128, n_tapers=3)
    >>> tapers.shape  # tapers by samples
    (3, 128)
    >>> bool((concentration > 0.95).all())
    True
    """

    NW = bandwidth * N / fs
    K = int(np.ceil(2 * NW)) - 1
    if n_tapers is not None:
        K = min(K, n_tapers)
    if K < 1:
        raise ValueError(
            f"Not enough tapers, with 'NW' of {NW}. Increase the bandwidth or "
            "use more data points"
        )

    tapers, lambdas = dpss(N, NW=NW, Kmax=K, sym=True, norm=2, return_ratios=True)
    mask = lambdas > min_lambda
    if not np.sum(mask) > 0:
        raise ValueError(
            "None of the tapers satisfied the minimum energy concentration"
            f" criteria of {min_lambda}"
        )
    tapers = tapers[mask]
    lambdas = lambdas[mask]

    if n_tapers is not None:
        if n_tapers > tapers.shape[0]:
            raise ValueError(
                f"'n_tapers' of {n_tapers} is greater than the {tapers.shape[0]}"
                f" that satisfied the minimum energy concentration criteria of {min_lambda}"
            )
        tapers = tapers[:n_tapers]
        lambdas = lambdas[:n_tapers]

    return tapers, lambdas


def mtfftpt(
    data: NDArray[Any],
    tapers: NDArray[Any],
    nfft: int,
    t: NDArray[Any],
    f: NDArray[Any],
    findx: NDArray[Any],
) -> Tuple[NDArray[Any], float, int]:
    """
    Multitaper FFT for point process times.

    Parameters
    ----------
    data : np.ndarray
        1D array of spike times (in seconds).
    tapers : np.ndarray
        DPSS tapers with shape (n_samples, n_tapers). Use tapers scaled by
        ``sqrt(Fs)`` for Chronux-normalized spectra.
    nfft : int
        FFT length, at least the number of samples in ``t``.
    t : np.ndarray
        Uniform taper time grid in seconds.
    f : np.ndarray
        Frequency bins in Hz returned by ``getfgrid`` for this FFT length.
    findx : np.ndarray of bool
        Matching FFT-bin mask returned by ``getfgrid``.

    Returns
    -------
    J : np.ndarray
        FFT of the data.
    Msp : float
        Mean spikes per time-grid sample.
    Nsp : int
        Total number of spikes in data.

    Notes
    -----
    The function computes the multitaper FFT of spike times using
    the specified tapers and returns the FFT result, mean spikes,
    and total spike count.

    Examples
    --------
    >>> import numpy as np
    >>> from neuro_py.process import pychronux as px
    >>> fs = nfft = 128
    >>> t = np.arange(nfft) / fs
    >>> spikes = np.array([0.09, 0.24, 0.48, 0.71])  # seconds
    >>> tapers = px.dpsschk((2, 3), len(t), fs)
    >>> frequencies, mask = px.getfgrid(fs, nfft, [1, 20])
    >>> transform, mean_spikes, n_spikes = px.mtfftpt(
    ...     spikes, tapers, nfft, t, frequencies, mask
    ... )
    >>> transform.shape, n_spikes  # (frequency bins, tapers), spike count
    ((20, 3), 4)
    """
    if nfft < len(t):
        raise ValueError("nfft must be at least the number of taper samples")
    taper_fft = _point_taper_fft(tapers, nfft, findx)
    return _point_fft(data, tapers, t, -2j * np.pi * f, taper_fft)


def _point_taper_fft(
    tapers: NDArray[Any], nfft: int, findx: NDArray[Any]
) -> NDArray[np.complex128]:
    return np.fft.fft(tapers, nfft, axis=0)[findx].astype(np.complex128, copy=False)


def _point_fft(
    data: NDArray[Any],
    tapers: NDArray[Any],
    t: NDArray[Any],
    angular_phase: NDArray[Any],
    taper_fft: NDArray[np.complex128],
) -> tuple[NDArray[np.complex128], float, int]:
    """Evaluate the spike-time Fourier sum using a reusable taper FFT."""
    times = data[(data >= t[0]) & (data <= t[-1])]
    count = len(times)
    mean_count = count / len(t)
    spectrum = np.zeros_like(taper_fft)
    if not count or not angular_phase.size:
        return spectrum, mean_count, count

    # Each frequency/spike phase is shared across all tapers. Bound the large
    # temporary to 2**18 complex values (4 MiB), even for dense spike trains.
    spike_block = 4096
    for lo in range(0, count, spike_block):
        block = times[lo : lo + spike_block]
        weights = np.column_stack([np.interp(block, t, taper) for taper in tapers.T])
        elapsed = block - t[0]
        frequency_block = (1 << 18) // len(block)
        for i in range(0, len(angular_phase), frequency_block):
            rows = slice(i, i + frequency_block)
            phase = angular_phase[rows, None] * elapsed
            np.exp(phase, out=phase)
            spectrum[rows] += np.einsum("fs,sk->fk", phase, weights, optimize=False)
    spectrum -= taper_fft * mean_count
    return spectrum, mean_count, count


def _point_tapers(tapers: NDArray[Any], n_samples: int) -> NDArray[Any]:
    """Normalize supplied point-process tapers to (samples, tapers)."""
    tapers = np.asarray(tapers)
    if tapers.ndim != 2:
        raise ValueError("tapers must be a two-dimensional sample/taper matrix")
    if tapers.shape[0] == n_samples:
        return tapers
    if tapers.shape[1] == n_samples:
        return tapers.T
    raise ValueError("tapers must have one sample dimension matching tapers_ts")


def _spike_times(data: NDArray[Any]) -> NDArray[np.float64]:
    times = np.asarray(data, dtype=np.float64)
    if times.ndim != 1 or not np.all(np.isfinite(times)):
        raise ValueError(
            "Each spike train must be a one-dimensional array of finite times"
        )
    return times


def _point_time_grid(
    trains: Sequence[NDArray[np.float64]],
    Fs: int,
    time_support: list[float] | None,
    tapers_ts: NDArray[Any] | None,
) -> NDArray[np.float64]:
    if not np.isfinite(Fs) or Fs <= 0:
        raise ValueError("Fs must be positive and finite")
    dt = 1 / Fs
    if time_support is not None:
        if (
            len(time_support) != 2
            or not np.all(np.isfinite(time_support))
            or time_support[0] >= time_support[1]
        ):
            raise ValueError("time_support must contain finite times with start < stop")
    if tapers_ts is not None:
        t = np.asarray(tapers_ts, dtype=np.float64)
        if (
            t.ndim != 1
            or t.size < 2
            or not np.all(np.isfinite(t))
            or not np.allclose(np.diff(t), dt, rtol=1e-6, atol=0)
        ):
            raise ValueError("tapers_ts must be a finite time grid spaced by 1 / Fs")
        if time_support is not None and (
            t[0] > time_support[0] + dt * 1e-6 or t[-1] < time_support[1] - dt * 1e-6
        ):
            raise ValueError("tapers_ts must cover time_support")
        return t
    if time_support is not None:
        start, stop = time_support
    else:
        nonempty = [train for train in trains if train.size]
        if not nonempty:
            raise ValueError(
                "time_support or tapers_ts is required for empty spike trains"
            )
        start = min(train.min() for train in nonempty)
        stop = max(train.max() for train in nonempty)
    # Match Chronux's start-dt:dt:stop+dt, including the final sample when it
    # falls on the grid. Count in samples to avoid a floating-point stop test.
    intervals = (stop - start) * Fs + 2
    n_samples = int(np.floor(intervals + 4 * np.spacing(intervals))) + 1
    return start + (np.arange(n_samples) - 1) / Fs


def _fft_size(n_samples: int, nfft: int | None = None, pad: int = 0) -> int:
    if nfft is None:
        return max(n_samples, (1 << (n_samples - 1).bit_length()) + pad)
    if nfft < n_samples:
        raise ValueError("nfft must be at least the number of taper samples")
    return nfft


def _spectrum_grid(
    Fs: int, nfft: int, fpass: list[float]
) -> tuple[NDArray[Any], NDArray[Any]]:
    if len(fpass) != 2 or not np.all(np.isfinite(fpass)):
        raise ValueError("fpass must be a pair of finite frequencies in Hz")
    if fpass[0] > fpass[1]:
        raise ValueError(
            "Invalid frequency range: fpass[0] should be less than fpass[1]."
        )
    if not 0 <= fpass[0] <= fpass[1] <= Fs / 2:
        raise ValueError("fpass must lie between 0 and the Nyquist frequency (Fs / 2)")
    return getfgrid(Fs, nfft, fpass)


def _restrict_spike_times(
    times: NDArray[np.float64], support: list[float] | None
) -> NDArray[np.float64]:
    if support is None:
        return times
    return times[(times >= support[0]) & (times <= support[1])]


def _support_array(support: _TimeSupport | None) -> NDArray[np.float64] | None:
    if support is None:
        return None
    message = "time_support must contain finite [start, stop] pairs with start < stop"
    try:
        values = np.asarray(support)
        if np.iscomplexobj(values):
            raise ValueError(message)
        bounds = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if (
        (bounds.shape != (2,) and (bounds.ndim != 2 or bounds.shape[1] != 2))
        or not bounds.size
        or not np.isfinite(bounds).all()
        or np.any(bounds[..., 0] >= bounds[..., 1])
    ):
        raise ValueError(message)
    return bounds


def _single_span(support: NDArray[np.float64] | None) -> list[float] | None:
    return None if support is None else [float(support[0]), float(support[1])]


def _merged_support(support: NDArray[np.float64]) -> NDArray[np.float64]:
    """Sort valid intervals and form their union without bridging gaps."""
    bounds = support.reshape(-1, 2)
    bounds = bounds[np.argsort(bounds[:, 0], kind="stable")]
    ends = np.maximum.accumulate(bounds[:, 1])
    cuts = np.flatnonzero(bounds[1:, 0] > ends[:-1]) + 1
    return np.column_stack(
        (bounds[np.r_[0, cuts], 0], ends[np.r_[cuts - 1, len(ends) - 1]])
    )


def _complete_bins(bounds: NDArray[np.float64], Fs: int) -> NDArray[np.int64]:
    # Subtracting absolute times can leave a nominally complete bin a few
    # ulps short. Account for precision on the original time axis.
    span = (bounds[:, 1] - bounds[:, 0]) * Fs
    tolerance = 4 * np.spacing(np.max(np.abs(bounds), axis=1)) * Fs
    return np.floor(span + tolerance).astype(np.int64)


class _PointWindows(NamedTuple):
    starts: NDArray[np.float64]
    stops: NDArray[np.float64]
    centers: NDArray[np.float64]
    frequencies: NDArray[np.float64]
    t: NDArray[np.float64]
    tapers: NDArray[np.float64]
    phase: NDArray[np.complex128]
    taper_fft: NDArray[np.complex128]
    observed_duration: float


def _point_windows(
    support: NDArray[np.float64],
    Fs: int,
    fpass: list[float],
    movingwin: tuple[float, float],
    NW: float,
    n_tapers: int,
    tapers: NDArray[Any] | None = None,
    nfft: int | None = None,
) -> _PointWindows:
    length, stride = _window_samples(movingwin, Fs)
    bounds = _merged_support(support)
    counts = _complete_bins(bounds, Fs)
    offsets = [np.arange(0, n - length + 1, stride) for n in counts]
    origins = np.repeat(bounds[:, 0], [len(part) for part in offsets])
    limits = np.repeat(bounds[:, 1], [len(part) for part in offsets])
    samples = np.concatenate(offsets)
    starts = origins + samples / Fs
    stops = np.minimum(origins + (samples + length) / Fs, limits)
    centers = origins + (samples + length / 2) / Fs

    t = np.arange(-1, length + 2) / Fs
    nfft = _fft_size(len(t), nfft)
    f, mask = _spectrum_grid(Fs, nfft, fpass)
    if starts.size:
        bank = (
            dpss(len(t), NW, n_tapers).T
            if tapers is None
            else _point_tapers(tapers, len(t))
        )
        bank = bank * np.sqrt(Fs)
        taper_fft = _point_taper_fft(bank, nfft, mask)
    else:
        bank = np.empty((len(t), 0))
        taper_fft = np.empty((len(f), 0), dtype=np.complex128)
    return _PointWindows(
        starts,
        stops,
        centers,
        f,
        t,
        bank,
        -2j * np.pi * f,
        taper_fft,
        float(np.sum(bounds[:, 1] - bounds[:, 0])),
    )


def _point_window_slices(
    times: NDArray[np.float64], windows: _PointWindows
) -> tuple[NDArray[np.float64], NDArray[np.intp], NDArray[np.intp]]:
    if np.any(times[1:] < times[:-1]):
        times = np.sort(times)
    return (
        times,
        np.searchsorted(times, windows.starts),
        np.searchsorted(times, windows.stops),
    )


def _point_window_fft(
    times: NDArray[np.float64], start: float, windows: _PointWindows
) -> NDArray[np.complex128]:
    j, _, _ = _point_fft(
        times - start, windows.tapers, windows.t, windows.phase, windows.taper_fft
    )
    return j


def _require_windows(count: int) -> None:
    if not count:
        raise ValueError(
            "No complete analysis windows fit in time_support; use a shorter window_duration"
        )


def _warn_low_coverage(count: int, covered: float, observed: float) -> None:
    if covered < 0.5 * observed:
        warnings.warn(
            f"{count} complete analysis windows cover only {covered / observed:.1%} of "
            "time_support. Check bout duration and choose window_duration for the "
            "frequency resolution and observation time you need.",
            UserWarning,
            stacklevel=4,
        )


def _point_window_power(
    trains: Sequence[NDArray[np.float64]], windows: _PointWindows
) -> pd.DataFrame:
    values = np.zeros((len(windows.frequencies), len(trains)))
    for unit, times in enumerate(trains):
        times, left, right = _point_window_slices(times, windows)
        for i in np.flatnonzero(right > left):
            j = _point_window_fft(times[left[i] : right[i]], windows.starts[i], windows)
            values[:, unit] += np.mean(np.abs(j) ** 2, axis=1)
    return pd.DataFrame(values / len(windows.starts), index=windows.frequencies)


def _point_window_pair(
    first: NDArray[Any], second: NDArray[Any], windows: _PointWindows
) -> tuple[NDArray[np.complex128], NDArray[np.float64], NDArray[np.float64]]:
    first, left1, right1 = _point_window_slices(_spike_times(first), windows)
    second, left2, right2 = _point_window_slices(_spike_times(second), windows)
    cross = np.zeros(len(windows.frequencies), dtype=np.complex128)
    power1 = np.zeros(len(windows.frequencies))
    power2 = np.zeros_like(power1)
    for i in np.flatnonzero((right1 > left1) | (right2 > left2)):
        start = windows.starts[i]
        j1 = j2 = None
        if right1[i] > left1[i]:
            j1 = _point_window_fft(first[left1[i] : right1[i]], start, windows)
            power1 += np.mean(np.abs(j1) ** 2, axis=1)
        if right2[i] > left2[i]:
            j2 = _point_window_fft(second[left2[i] : right2[i]], start, windows)
            power2 += np.mean(np.abs(j2) ** 2, axis=1)
        if j1 is not None and j2 is not None:
            cross += np.mean(j1.conj() * j2, axis=1)
    count = len(windows.starts)
    return cross / count, power1 / count, power2 / count


def _point_segment_plan(
    support: NDArray[np.float64],
    Fs: int,
    fpass: list[float],
    window_duration: float,
    NW: float,
    n_tapers: int,
    tapers: NDArray[Any] | None,
    tapers_ts: NDArray[Any] | None,
    nfft: int | None,
) -> _PointWindows:
    if tapers_ts is not None:
        raise ValueError(
            "tapers_ts is not supported for interval arrays; each window uses a local grid"
        )
    windows = _point_windows(
        support,
        Fs,
        fpass,
        (window_duration, window_duration),
        NW,
        n_tapers,
        tapers,
        nfft,
    )
    count = len(windows.starts)
    _require_windows(count)
    covered = float(np.sum(windows.stops - windows.starts))
    _warn_low_coverage(count, covered, windows.observed_duration)
    return windows


def mtspectrumpt(
    data: NDArray[Any] | Sequence[NDArray[Any]] | nel.SpikeTrainArray,
    Fs: int,
    fpass: list[float],
    NW: Union[int, float] = 2.5,
    n_tapers: int = 4,
    time_support: _TimeSupport | None = None,
    tapers: Union[NDArray[Any], None] = None,
    tapers_ts: Union[NDArray[Any], None] = None,
    nfft: Optional[int] = None,
    *,
    window_duration: float = 4.0,
) -> pd.DataFrame:
    """
    Multitaper power spectrum estimation for point process data.

    Parameters
    ----------
    data : np.ndarray, sequence of np.ndarray, or nelpy.SpikeTrainArray
        Spike times in seconds, one array per unit. A spike train object also
        supplies its observation support if no explicit support is passed.
    Fs : int
        Taper interpolation grid rate in Hz. Spike times remain in seconds.
    fpass : list of float
        Frequency range to evaluate.
    NW : Union[int, float], optional
        Time-bandwidth product (default is 2.5).
    n_tapers : int, optional
        Number of tapers (default is 4).
    time_support : array-like, optional
        A single ``[start, stop]`` pair retains the full-span estimator. An
        ``(n_intervals, 2)`` array in seconds uses pooled, equal-length windows
        wholly inside the union of the half-open intervals, excluding gaps.
        With raw arrays and no support, use ``tapers_ts`` or the first and last
        spike across units. Native spike trains use their support instead.
    tapers : Union[np.ndarray, None], optional
        Precomputed, unscaled DPSS tapers in samples-by-tapers or
        tapers-by-samples order. Internally scaled by ``sqrt(Fs)``. Default is None.
    tapers_ts : Union[np.ndarray, None], optional
        Optional taper timestamps in seconds, uniformly spaced by ``1 / Fs``.
        Available for a single span only. Windowed tapers must match the
        local grid from ``-1 / Fs`` to window duration plus ``1 / Fs``.
    nfft : Optional[int], optional
        FFT length, at least the length of the taper grid. Defaults to the
        next power of two.
    window_duration : float, optional
        Window duration in seconds (default 4), rounded to a grid sample.
        Used only with interval arrays. Short bouts and leftover partial
        windows are omitted; complete windows without spikes are retained.
        Warns if complete windows cover less than half of the supplied support.

    Returns
    -------
    pd.DataFrame
        Power density indexed by frequency in Hz, with one column per unit.
        Positive-frequency bins retain the two-sided normalization.

    Examples
    --------
    >>> import numpy as np
    >>> from neuro_py.process import pychronux as px
    >>> spikes = np.arange(0.06, 4, 0.125)  # spike times in seconds
    >>> power = px.mtspectrumpt(
    ...     [spikes, spikes + 0.01], Fs=128, fpass=[1, 20], time_support=[0, 4]
    ... )
    >>> power.shape[1]  # one column per unit
    2
    """

    if isinstance(data, nel.SpikeTrainArray):
        if data.data is None or not len(data.data):
            return pd.DataFrame()
        if time_support is None:
            time_support = np.asarray(data.support.data, dtype=np.float64)
        data = data.data

    support = _support_array(time_support)
    if len(data) == 0:
        if (
            support is None
            or support.ndim == 1
            or not isinstance(data, np.ndarray)
            or data.ndim != 1
        ):
            return pd.DataFrame()
        trains = [_spike_times(data)]  # observed window(s) from one silent unit
    elif isinstance(data, np.ndarray) and data.ndim == 1 and np.ndim(data[0]) == 0:
        trains = [_spike_times(np.asarray(data, dtype=np.float64))]
    else:
        trains = [_spike_times(train) for train in data]
    if support is not None and support.ndim == 2:
        windows = _point_segment_plan(
            support, Fs, fpass, window_duration, NW, n_tapers, tapers, tapers_ts, nfft
        )
        return _point_window_power(trains, windows)
    span = _single_span(support)
    t = _point_time_grid(trains, Fs, span, tapers_ts)
    nfft = _fft_size(len(t), nfft)
    f, findx = _spectrum_grid(Fs, nfft, fpass)
    if tapers is None:
        tapers = dpss(len(t), NW, n_tapers).T
    # Apply the same Chronux scaling to generated and supplied DPSS tapers.
    tapers = _point_tapers(tapers, len(t)) * np.sqrt(Fs)
    taper_fft = _point_taper_fft(tapers, nfft, findx)
    angular_phase = -2j * np.pi * f

    spec = np.empty((len(f), len(trains)))
    for i, train in enumerate(trains):
        observed = _restrict_spike_times(train, span)
        J, _, _ = _point_fft(observed, tapers, t, angular_phase, taper_fft)
        spec[:, i] = np.mean(np.abs(J) ** 2, axis=1)

    return pd.DataFrame(spec, index=f)


@njit(cache=True)
def _within_bout_pairs(
    bouts: NDArray[np.intp], bins: NDArray[np.int64], max_offset: int
) -> NDArray[np.int64]:
    """Count distinct spike pairs on the local grid without forming pair arrays."""
    counts = np.zeros(max_offset + 1, dtype=np.int64)
    for i in range(len(bins)):
        for j in range(i + 1, len(bins)):
            if bouts[j] != bouts[i]:
                break
            offset = bins[j] - bins[i]
            if offset > max_offset:
                break
            counts[offset] += 1
    counts[0] *= 2  # two ordered pairs for distinct events in the same bin
    return counts


def _lag_pair_totals(
    lengths: NDArray[np.int64],
    lags: NDArray[np.int64],
    first: NDArray[np.intp],
    weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Sum weights[b] * max(lengths[b] - lag, 0) without a bout-by-lag matrix."""
    weighted_lengths = np.r_[np.cumsum((lengths * weights)[::-1])[::-1], 0]
    remaining_weights = np.r_[np.cumsum(weights[::-1])[::-1], 0]
    return weighted_lengths[first] - lags * remaining_weights[first]


def point_lag_spectrum(
    data: NDArray[Any] | Sequence[NDArray[Any]] | nel.SpikeTrainArray,
    Fs: int,
    fpass: list[float],
    *,
    max_lag: float = 1.0,
    time_support: _TimeSupport | None = None,
    nfft: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Estimate rhythmic spiking by pooling pairs within observed bouts.

    Parameters
    ----------
    data : np.ndarray, sequence of np.ndarray, or nelpy.SpikeTrainArray
        Spike times in seconds, one array per unit. A native spike train
        supplies its support if no explicit support is given.
    Fs : int
        Spike binning rate in Hz, not the original timestamp sampling rate.
        Use substantially more than twice the upper frequency to limit
        attenuation from binning.
    fpass : list of float
        Inclusive frequency range in Hz.
    max_lag : float, optional
        Largest lag in seconds, rounded to a bin (default 1 s). The full
        width at half maximum of the Hann lag window's spectral kernel is
        approximately ``1 / max_lag`` Hz. Both the number of available pairs
        and the rhythmic persistence limit what can be resolved in practice.
    time_support : array-like, optional
        Half-open observation interval(s) in seconds. Required for raw
        spike times. Overlapping or touching intervals are merged, and
        only complete time bins within each interval are used.
    nfft : int, optional
        FFT length, at least twice the largest lag in bins plus one. The
        default is the next power of two. Padding only refines the grid.

    Returns
    -------
    spectrum : pd.DataFrame
        Signed excess spectral density (spikes squared per second, two-sided
        normalization) indexed by frequency in Hz, with one column per unit.
        This is not a nonnegative power spectral density.
    exposure : pd.DataFrame
        Lag in seconds as the index. Columns report the number of bouts
        that can contribute and eligible bin pairs expressed in seconds.
        Check exposure at the lags that determine your resolution.

    Notes
    -----
    For each lag, pool distinct spike pairs and divide by the number of
    eligible same-bout bin pairs. Subtract the expected counts under
    independent uniform spike times conditional on each bout's spike count.
    For a bout with ``m`` bins and ``n`` spikes this expectation at bin offset
    ``k`` is ``n * (n - 1) * max(m - k, 0) / m**2``; this also applies
    at lag zero after excluding spike self-pairs. Transform the symmetric,
    Hann-tapered excess correlation. Bouts can have unrelated phases, but
    the estimator does not remove an event-locked firing rate envelope or
    test phase locking to an LFP. No pair ever crosses a gap. A bout shorter
    than a requested lag contributes only to shorter lags.

    Examples
    --------
    >>> import numpy as np
    >>> from neuro_py.process import pychronux as px
    >>> bouts = np.array([[0., 2.], [30., 32.]])
    >>> spikes = np.r_[np.arange(0.1, 2, 0.125), np.arange(30.1, 32, 0.125)]
    >>> spectrum, exposure = px.point_lag_spectrum(spikes, 128, [4, 12], time_support=bouts)
    >>> float(exposure.loc[1.0, "pair_exposure_s"])
    2.0
    """
    if not np.isfinite(Fs) or Fs <= 0 or not np.isfinite(max_lag) or max_lag <= 0:
        raise ValueError("Fs and max_lag must be positive and finite")
    max_offset = int(np.floor(max_lag * Fs + 0.5))
    if max_offset < 2:
        raise ValueError("max_lag must span at least two bins")
    if isinstance(data, nel.SpikeTrainArray):
        if time_support is None:
            time_support = data.support.data
        data = data.data if data.data is not None else []
    support = _support_array(time_support)
    if support is None:
        raise ValueError("time_support is required for raw spike times")
    bounds = _merged_support(support)
    lengths = _complete_bins(bounds, Fs)
    keep = lengths > 0
    bounds, lengths = bounds[keep], lengths[keep]
    if not lengths.size or np.max(lengths) <= max_offset:
        raise ValueError("No same-bout bin pairs at max_lag; use a shorter max_lag")

    nfft = _fft_size(2 * max_offset + 1, nfft)
    f, mask = _spectrum_grid(Fs, nfft, fpass)
    starts = bounds[:, 0]
    stops = np.minimum(bounds[:, 1], starts + lengths / Fs)
    precision = 4 * np.spacing(np.max(np.abs(bounds), axis=1)) * Fs
    lags = np.arange(max_offset + 1, dtype=np.int64)
    order = np.argsort(lengths)
    sorted_lengths = lengths[order]
    first = np.searchsorted(sorted_lengths, lags, side="right")
    eligible = _lag_pair_totals(sorted_lengths, lags, first, np.ones(len(lengths)))
    lag_taper = 0.5 * (1 + np.cos(np.pi * lags / max_offset))
    if (
        isinstance(data, np.ndarray)
        and data.ndim == 1
        and (not data.size or np.ndim(data[0]) == 0)
    ):
        trains = [_spike_times(data)]
    else:
        trains = [_spike_times(train) for train in data]

    spectra = np.empty((len(f), len(trains)))
    for unit, times in enumerate(trains):
        if np.any(times[1:] < times[:-1]):
            times = np.sort(times)
        bouts = np.searchsorted(starts, times, side="right") - 1
        in_bout = (bouts >= 0) & (times < stops[bouts])
        bouts, times = bouts[in_bout], times[in_bout]
        local = (times - starts[bouts]) * Fs
        bins = np.minimum(
            np.floor(local + precision[bouts]).astype(np.int64), lengths[bouts] - 1
        )
        pairs = _within_bout_pairs(bouts, bins, max_offset)
        spikes_per_bout = np.bincount(bouts, minlength=len(lengths))[order].astype(
            float
        )
        baseline_weights = (
            spikes_per_bout * (spikes_per_bout - 1) / sorted_lengths.astype(float) ** 2
        )
        expected = _lag_pair_totals(sorted_lengths, lags, first, baseline_weights)
        tapered = lag_taper * (pairs - expected) / eligible
        cosine_sum = np.fft.rfft(tapered, n=nfft).real
        spectra[:, unit] = Fs * (2 * cosine_sum[mask[: len(cosine_sum)]] - tapered[0])

    exposure = pd.DataFrame(
        {"pair_exposure_s": eligible / Fs, "n_bouts": len(lengths) - first},
        index=pd.Index(lags / Fs, name="lag_s"),
    )
    return pd.DataFrame(spectra, index=pd.Index(f, name="frequency_hz")), exposure


def mtfftc(
    data: NDArray[Any], tapers: NDArray[Any], nfft: int, Fs: int
) -> NDArray[Any]:
    """
    Multi-taper Fourier Transform - Continuous Data (Single Signal)

    Parameters
    ----------
    data : np.ndarray
        1D array of data (samples).
    tapers : np.ndarray
        Precomputed DPSS tapers with shape (samples, tapers).
    nfft : int
        Length of padded data for FFT.
    Fs : int
        Sampling frequency.

    Returns
    -------
    J : np.ndarray
        FFT in the form (nfft, K), where K is the number of tapers.

    Examples
    --------
    >>> import numpy as np
    >>> from neuro_py.process import pychronux as px
    >>> fs = 128
    >>> signal = np.sin(2 * np.pi * 8 * np.arange(fs) / fs)
    >>> tapers = px.dpsschk((2, 3), len(signal), fs)
    >>> transform = px.mtfftc(signal, tapers, nfft=128, Fs=fs)
    >>> transform.shape  # FFT bins by tapers
    (128, 3)
    """
    # Ensure data is 1D
    if data.ndim != 1:
        raise ValueError("Input data must be a 1D array.")

    NC = data.shape[0]  # Number of samples in data
    NK, K = tapers.shape  # Number of samples and tapers

    if NK != NC:
        raise ValueError("Length of tapers is incompatible with length of data.")

    # Project data onto tapers
    data_proj = data[:, np.newaxis] * tapers  # Shape: (samples, tapers)

    # Compute FFT for each taper
    J = np.fft.fft(data_proj, n=nfft, axis=0) / Fs  # Shape: (nfft, K)

    return J


def mtspectrumc(
    data: NDArray[Any],
    Fs: int,
    fpass: list[float],
    tapers: NDArray[Any] | tuple[float, int],
    *,
    time_support: _TimeSupport | None = None,
    window_duration: float = 4.0,
) -> pd.Series:
    """
    Compute the multitaper power spectrum for continuous data.

    Parameters
    ----------
    data : np.ndarray
        1D array of continuous data (e.g., LFP).
    Fs : int
        Sampling frequency in Hz.
    fpass : list
        Frequency range to evaluate as [min_freq, max_freq].
    tapers : np.ndarray or tuple of (float, int)
        DPSS time-bandwidth product NW and taper count K, as ``(NW, K)``.
    time_support : array-like, optional
        Half-open interval or array of intervals in seconds from the first
        sample. An interval array pools power over complete windows without
        crossing gaps. Default uses the whole signal.
    window_duration : float, optional
        Complete, non-overlapping window duration for interval arrays (default 4 s).

    Returns
    -------
    S : pd.Series
        Power density with frequency in Hz as the index. The density has
        units of input amplitude squared per Hz. Positive-frequency bins
        retain the two-sided normalization.

    Notes
    -----
    This function utilizes the multitaper method for spectral estimation
    and returns the power spectrum as a pandas Series.

    Examples
    --------
    >>> import numpy as np
    >>> from neuro_py.process import pychronux as px
    >>> fs = 128
    >>> t = np.arange(4 * fs) / fs
    >>> signal = np.sin(2 * np.pi * 8 * t)
    >>> power = px.mtspectrumc(signal, fs, [1, 20], tapers=(3, 5))
    >>> abs(float(power.idxmax()) - 8) < 1  # peak frequency in Hz
    True
    """
    support = _support_array(time_support)
    if support is not None and support.ndim == 2:
        f, values = _continuous_window_spectrum(
            data, Fs, fpass, tapers, support, window_duration
        )
        return pd.Series(values, index=f)
    if support is not None:
        ((start, stop),) = _continuous_support_samples(support, len(data), Fs)
        data = data[start:stop]
    N = len(data)
    if N == 0:
        raise ValueError("data must contain at least one sample")
    nfft = _fft_size(N)
    # get the frequency grid
    f, findx = _spectrum_grid(Fs, nfft, fpass)
    # get the fft of the tapers
    tapers = dpsschk(tapers, N, Fs)
    # get the fft of the data
    J = mtfftc(data, tapers, nfft, Fs)
    # restrict fft of tapers to required frequencies
    J = J[findx, :]
    # get the power spectrum
    S = np.real(np.mean(np.conj(J) * J, 1))
    # return the power spectrum
    return pd.Series(index=f, data=S)


def point_spectra(
    times: NDArray[Any],
    Fs: int = 1250,
    freq_range: List[float] = [1, 20],
    tapers0: List[int] = [3, 5],
    pad: int = 0,
    nfft: Optional[int] = None,
    *,
    time_support: _TimeSupport | None = None,
    window_duration: float = 4.0,
) -> Tuple[NDArray[Any], NDArray[Any]]:
    """
    Compute multitaper power spectrum for point processes.

    Parameters
    ----------
    times : np.ndarray
        Array of spike times (in seconds).
    Fs : int, optional
        Taper interpolation grid rate in Hz (default is 1250 Hz).
    freq_range : List[float], optional
        Frequency range to evaluate (default is [1, 20] Hz).
    tapers0 : List[int], optional
        Time-bandwidth product and number of tapers (default is [3, 5]).
        The time-bandwidth product is used to compute the tapers.
    pad : int, optional
        Extra FFT bins beyond the default power-of-two length (default is 0).
        Ignored when ``nfft`` is specified.
    nfft : Optional[int], optional
        FFT length, at least the length of the taper grid. Default is None.
    time_support : array-like, optional
        Observation interval or array of intervals in seconds, as in
        ``mtspectrumpt``. Defaults to the first and last spike.
    window_duration : float, optional
        Complete, non-overlapping window duration for interval arrays (default 4 s).

    Returns
    -------
    spectra : np.ndarray
        One-dimensional power density, using the same normalization as
        ``mtspectrumpt``.
    f : np.ndarray
        Frequency vector.

    Notes
    -----
    Alternative function to `mtspectrumpt` for computing the power spectrum

    Examples
    --------
    >>> import numpy as np
    >>> from neuro_py.process import pychronux as px
    >>> spikes = np.arange(0.06, 4, 0.125)  # seconds
    >>> power, frequencies = px.point_spectra(
    ...     spikes, Fs=128, freq_range=[1, 20], time_support=[0, 4]
    ... )
    >>> power.ndim, power.shape == frequencies.shape
    (1, True)
    """

    times = _spike_times(times)
    support = _support_array(time_support)
    t = None
    if support is not None and support.ndim == 2:
        length, _ = _window_samples((window_duration, window_duration), Fs)
        nfft = _fft_size(length + 3, nfft, pad)
    else:
        t = _point_time_grid([times], Fs, _single_span(support), None)
        nfft = _fft_size(len(t), nfft, pad)
    NW, n_tapers = tapers0
    spectrum = mtspectrumpt(
        times[None, :],
        Fs,
        freq_range,
        NW,
        n_tapers,
        time_support=time_support,
        tapers_ts=t,
        nfft=nfft,
        window_duration=window_duration,
    )
    return spectrum.iloc[:, 0].to_numpy(), spectrum.index.to_numpy()


def _point_pair(
    data1: NDArray[Any],
    data2: NDArray[Any],
    Fs: int,
    fpass: list[float],
    NW: float,
    n_tapers: int,
    time_support: list[float] | None,
    tapers: NDArray[Any] | None,
    tapers_ts: NDArray[Any] | None,
    nfft: int | None,
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Compute two tapered transforms with a common time and frequency grid."""
    first, second = _spike_times(data1), _spike_times(data2)
    t = _point_time_grid([first, second], Fs, time_support, tapers_ts)
    nfft = _fft_size(len(t), nfft)
    f, findx = _spectrum_grid(Fs, nfft, fpass)
    if tapers is None:
        tapers = dpss(len(t), NW, n_tapers).T
    tapers = _point_tapers(tapers, len(t)) * np.sqrt(Fs)
    first = _restrict_spike_times(first, time_support)
    second = _restrict_spike_times(second, time_support)
    taper_fft = _point_taper_fft(tapers, nfft, findx)
    angular_phase = -2j * np.pi * f
    j1, _, _ = _point_fft(first, tapers, t, angular_phase, taper_fft)
    j2, _, _ = _point_fft(second, tapers, t, angular_phase, taper_fft)
    return f, j1, j2


def mtcsdpt(
    data1: NDArray[Any],
    data2: NDArray[Any],
    Fs: int,
    fpass: list[float],
    NW: Union[int, float] = 2.5,
    n_tapers: int = 4,
    time_support: _TimeSupport | None = None,
    tapers: Union[NDArray[Any], None] = None,
    tapers_ts: Union[NDArray[Any], None] = None,
    nfft: Optional[int] = None,
    *,
    return_complex: bool = False,
    window_duration: float = 4.0,
) -> pd.DataFrame:
    """
    Multitaper cross-spectral density (CSD) for point processes.

    Parameters
    ----------
    data1 : np.ndarray
        Array of spike times for the first signal (in seconds).
    data2 : np.ndarray
        Array of spike times for the second signal (in seconds).
    Fs : int
        Taper interpolation grid rate in Hz.
    fpass : list
        Frequency range to evaluate as [min_freq, max_freq].
    NW : Union[int, float], optional
        Time-bandwidth product, by default 2.5.
    n_tapers : int, optional
        Number of tapers, by default 4.
    time_support : array-like, optional
        Observation interval or array of intervals in seconds, as in
        ``mtspectrumpt``. Interval arrays pool complex cross-spectra across
        complete windows. Defaults to ``tapers_ts`` or the joint spike span.
    tapers : Union[np.ndarray, None], optional
        Precomputed, unscaled DPSS tapers in samples-by-tapers or
        tapers-by-samples order. Internally scaled by ``sqrt(Fs)``. Default is None.
    tapers_ts : Union[np.ndarray, None], optional
        Taper timestamps in seconds, uniformly spaced by ``1 / Fs``.
        Available for a single span only.
    nfft : Optional[int], optional
        FFT length, at least the length of the taper grid. Default is None.

    return_complex : bool, optional
        Return the full complex cross-spectrum when True. Default False retains
        the historical real cospectrum for existing callers.
    window_duration : float, optional
        Complete, non-overlapping window duration for interval arrays (default 4 s).

    Returns
    -------
    pd.DataFrame
        Cross-spectral density (or its real component by default).

    Examples
    --------
    >>> import numpy as np
    >>> from neuro_py.process import pychronux as px
    >>> first = np.arange(0.06, 4, 0.125)  # seconds
    >>> second = first + 0.01  # second spike train lags by 10 ms
    >>> cross = px.mtcsdpt(
    ...     first, second, 128, [1, 20], time_support=[0, 4], return_complex=True
    ... )
    >>> np.iscomplexobj(cross["CSD"])
    True
    """
    support = _support_array(time_support)
    if support is not None and support.ndim == 2:
        windows = _point_segment_plan(
            support, Fs, fpass, window_duration, NW, n_tapers, tapers, tapers_ts, nfft
        )
        f = windows.frequencies
        csd, _, _ = _point_window_pair(data1, data2, windows)
    else:
        f, J1, J2 = _point_pair(
            data1,
            data2,
            Fs,
            fpass,
            NW,
            n_tapers,
            _single_span(support),
            tapers,
            tapers_ts,
            nfft,
        )
        csd = np.mean(np.conj(J1) * J2, axis=1)
    if not return_complex:
        csd = csd.real

    csd_df = pd.DataFrame({"CSD": csd}, index=f)
    return csd_df


def mtcoherencept(
    data1: NDArray[Any],
    data2: NDArray[Any],
    Fs: int,
    fpass: list[float],
    NW: Union[int, float] = 2.5,
    n_tapers: int = 4,
    time_support: _TimeSupport | None = None,
    tapers: Union[NDArray[Any], None] = None,
    tapers_ts: Union[NDArray[Any], None] = None,
    nfft: Optional[int] = None,
    *,
    window_duration: float = 4.0,
) -> pd.DataFrame:
    """
    Multitaper coherence for point processes.

    Parameters
    ----------
    data1 : np.ndarray
        Array of spike times for the first signal (in seconds).
    data2 : np.ndarray
        Array of spike times for the second signal (in seconds).
    Fs : int
        Taper interpolation grid rate in Hz.
    fpass : list
        Frequency range to evaluate as [min_freq, max_freq].
    NW : Union[int, float], optional
        Time-bandwidth product, by default 2.5.
    n_tapers : int, optional
        Number of tapers, by default 4.
    time_support : array-like, optional
        Observation interval or array of intervals in seconds, as in
        ``mtspectrumpt``. Defaults to ``tapers_ts`` or the joint spike span.
    tapers : Union[np.ndarray, None], optional
        Precomputed, unscaled DPSS tapers in samples-by-tapers or
        tapers-by-samples order. Default is None.
    tapers_ts : Union[np.ndarray, None], optional
        Taper timestamps in seconds, uniformly spaced by ``1 / Fs``.
        Available for a single span only.
    nfft : Optional[int], optional
        FFT length, at least the length of the taper grid. Default is None.
    window_duration : float, optional
        Complete, non-overlapping window duration for interval arrays (default 4 s).

    Returns
    -------
    pd.DataFrame
        Coherence between the two point processes.

    Notes
    -----
    Both auto-spectra and the complex cross-spectrum use a shared time grid,
    support, taper set and FFT size. Supplied tapers may be samples-by-tapers
    or its transpose. Zero-power frequencies have undefined (NaN) coherence.
    With an interval array, average the complex cross-spectrum and both
    auto-spectra across complete windows before computing coherence. A single
    taper in a single window gives coherence of one where both powers are nonzero.

    Examples
    --------
    >>> import numpy as np
    >>> from neuro_py.process import pychronux as px
    >>> first = np.arange(0.06, 4, 0.125)  # 8 Hz spike train, in seconds
    >>> second = first + 0.01
    >>> coherence = px.mtcoherencept(first, second, 128, [1, 20], time_support=[0, 4])
    >>> float(coherence.loc[8.0, "Coherence"]) > 0.9
    True
    """
    support = _support_array(time_support)
    if support is not None and support.ndim == 2:
        windows = _point_segment_plan(
            support, Fs, fpass, window_duration, NW, n_tapers, tapers, tapers_ts, nfft
        )
        f = windows.frequencies
        values = _coherence_from_spectra(*_point_window_pair(data1, data2, windows))
    else:
        f, J1, J2 = _point_pair(
            data1,
            data2,
            Fs,
            fpass,
            NW,
            n_tapers,
            _single_span(support),
            tapers,
            tapers_ts,
            nfft,
        )
        values = _coherence(J1, J2)
    return pd.DataFrame({"Coherence": values}, index=f)


def _cross_and_auto(
    J1: NDArray[Any], J2: NDArray[Any]
) -> tuple[NDArray[np.complex128], NDArray[np.float64], NDArray[np.float64]]:
    return (
        np.mean(J1.conj() * J2, axis=1),
        np.mean(np.abs(J1) ** 2, axis=1),
        np.mean(np.abs(J2) ** 2, axis=1),
    )


def _coherence_from_spectra(
    cross: NDArray[np.complex128],
    power1: NDArray[np.float64],
    power2: NDArray[np.float64],
) -> NDArray[np.float64]:
    with np.errstate(divide="ignore", invalid="ignore"):
        # Suppress roundoff above the Cauchy–Schwarz bound; zero power stays NaN.
        return np.clip(np.abs(cross) ** 2 / (power1 * power2), 0, 1)


def _coherence(J1: NDArray[Any], J2: NDArray[Any]) -> NDArray[np.float64]:
    return _coherence_from_spectra(*_cross_and_auto(J1, J2))


def _check_continuous_pair(data1: NDArray[Any], data2: NDArray[Any]) -> None:
    if data1.ndim != 1 or data1.shape != data2.shape or not data1.size:
        raise ValueError("data1 and data2 must be nonempty 1D arrays of equal length")


def mtcoherencec(
    data1: NDArray[Any],
    data2: NDArray[Any],
    Fs: int,
    fpass: list[float],
    tapers: NDArray[Any] | tuple[float, int],
    *,
    time_support: _TimeSupport | None = None,
    window_duration: float = 4.0,
) -> pd.Series:
    """
    Estimate magnitude-squared coherence between two continuous signals.

    Parameters
    ----------
    data1, data2 : np.ndarray
        Aligned, equally sized 1D signals.
    Fs : int
        Sampling frequency in Hz.
    fpass : list of float
        Inclusive frequency range in Hz, from zero to Nyquist.
    tapers : np.ndarray or tuple of (float, int)
        DPSS time-bandwidth product and number of tapers, ``(NW, K)``.
    time_support : array-like, optional
        Half-open interval or array of intervals in seconds from the first
        sample. Default uses both whole signals. An interval array pools the
        cross- and auto-spectra before computing coherence.
    window_duration : float, optional
        Complete, non-overlapping window duration for interval arrays (default 4 s).

    Returns
    -------
    pd.Series
        Magnitude-squared coherence indexed by frequency in Hz. Frequencies
        with zero power in either signal have NaN coherence.

    Notes
    -----
    This is the square of the magnitude returned by Chronux ``coherencyc``.
    With one taper, coherence is one wherever both spectra have power.

    Examples
    --------
    >>> import numpy as np
    >>> from neuro_py.process import pychronux as px
    >>> fs = 128
    >>> t = np.arange(4 * fs) / fs
    >>> x = np.sin(2 * np.pi * 8 * t)
    >>> y = np.cos(2 * np.pi * 8 * t) + 0.5 * np.sin(2 * np.pi * 15 * t)
    >>> coherence = px.mtcoherencec(x, y, fs, [1, 20], tapers=(3, 5))
    >>> float(coherence.loc[8.0]) > 0.9  # shared oscillation
    True
    """
    _check_continuous_pair(data1, data2)
    support = _support_array(time_support)
    if support is not None and support.ndim == 2:
        f, values = _continuous_window_spectrum(
            data1, Fs, fpass, tapers, support, window_duration, other=data2
        )
        return pd.Series(values, index=f, name="Coherence")
    if support is not None:
        ((start, stop),) = _continuous_support_samples(support, len(data1), Fs)
        data1, data2 = data1[start:stop], data2[start:stop]
        _check_continuous_pair(data1, data2)
    nfft = _fft_size(len(data1))
    f, findx = _spectrum_grid(Fs, nfft, fpass)
    bank = dpsschk(tapers, len(data1), Fs)
    j1 = mtfftc(data1, bank, nfft, Fs)[findx]
    j2 = mtfftc(data2, bank, nfft, Fs)[findx]
    return pd.Series(_coherence(j1, j2), index=f, name="Coherence")


def _window_samples(movingwin: tuple[float, float], Fs: int) -> tuple[int, int]:
    """Convert window duration and step seconds to sample counts."""
    if not np.isfinite(Fs) or Fs <= 0:
        raise ValueError("Fs must be positive and finite")
    duration, step = movingwin
    if not np.all(np.isfinite(movingwin)) or duration <= 0 or step <= 0:
        raise ValueError(
            "movingwin must contain positive finite duration and step values"
        )
    length = int(np.floor(duration * Fs + 0.5))
    stride = int(np.floor(step * Fs + 0.5))
    if length < 2 or stride < 1:
        raise ValueError(
            "movingwin must span at least two samples and step by at least one"
        )
    return length, stride


def _gram_frame(
    values: NDArray[np.float64], f: NDArray[Any], centers: NDArray[Any]
) -> pd.DataFrame:
    return pd.DataFrame(
        values,
        index=pd.Index(f, name="frequency_hz"),
        columns=pd.Index(centers, name="time_s"),
    )


def _continuous_support_samples(
    support: NDArray[np.float64], n_samples: int, Fs: int
) -> NDArray[np.intp]:
    if not np.isfinite(Fs) or Fs <= 0:
        raise ValueError("Fs must be positive and finite")
    bounds = _merged_support(support)
    if bounds[0, 0] < 0 or bounds[-1, 1] > n_samples / Fs:
        raise ValueError(
            "time_support for continuous data must lie within [0, len(data) / Fs]"
        )
    # Indices of the first sample at or after each boundary.
    return np.ceil(np.nextafter(bounds * Fs, -np.inf)).astype(np.intp)


class _ContinuousWindows(NamedTuple):
    starts: NDArray[np.intp]
    length: int
    nfft: int
    frequencies: NDArray[np.float64]
    mask: NDArray[np.bool_]
    observed_samples: int


def _continuous_windows(
    data: NDArray[Any],
    Fs: int,
    fpass: list[float],
    movingwin: tuple[float, float],
    support: NDArray[np.float64] | None,
) -> _ContinuousWindows:
    if data.ndim != 1:
        raise ValueError("data must be a 1D signal")
    length, stride = _window_samples(movingwin, Fs)
    nfft = _fft_size(length)
    f, mask = _spectrum_grid(Fs, nfft, fpass)
    if support is None:
        starts = np.arange(0, len(data) - length + 1, stride, dtype=np.intp)
        observed_samples = len(data)
    else:
        bounds = _continuous_support_samples(support, len(data), Fs)
        observed_samples = int(np.sum(bounds[:, 1] - bounds[:, 0]))
        starts = np.concatenate(
            [
                np.arange(start, stop - length + 1, stride, dtype=np.intp)
                for start, stop in bounds
            ]
        )
    return _ContinuousWindows(starts, length, nfft, f, mask, observed_samples)


def _continuous_window_ffts(
    data: NDArray[Any],
    other: NDArray[Any] | None,
    Fs: int,
    tapers: NDArray[Any] | tuple[float, int],
    windows: _ContinuousWindows,
) -> Iterator[tuple[NDArray[np.complex128], NDArray[np.complex128] | None]]:
    if not windows.starts.size:
        return
    bank = dpsschk(tapers, windows.length, Fs)
    for start in windows.starts:
        window = slice(start, start + windows.length)
        j1 = mtfftc(data[window], bank, windows.nfft, Fs)[windows.mask]
        j2 = (
            None
            if other is None
            else mtfftc(other[window], bank, windows.nfft, Fs)[windows.mask]
        )
        yield j1, j2


def _continuous_window_spectrum(
    data: NDArray[Any],
    Fs: int,
    fpass: list[float],
    tapers: NDArray[Any] | tuple[float, int],
    support: NDArray[np.float64],
    window_duration: float,
    other: NDArray[Any] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    windows = _continuous_windows(
        data, Fs, fpass, (window_duration, window_duration), support
    )
    _require_windows(len(windows.starts))
    _warn_low_coverage(
        len(windows.starts),
        len(windows.starts) * windows.length,
        windows.observed_samples,
    )
    f = windows.frequencies
    power1 = np.zeros(len(f))
    power2 = np.zeros_like(power1)
    cross = np.zeros(len(f), dtype=np.complex128)
    for j1, j2 in _continuous_window_ffts(data, other, Fs, tapers, windows):
        if j2 is None:
            power1 += np.mean(np.abs(j1) ** 2, axis=1)
        else:
            xy, xx, yy = _cross_and_auto(j1, j2)
            cross += xy
            power1 += xx
            power2 += yy
    if other is None:
        return f, power1 / len(windows.starts)
    return f, _coherence_from_spectra(cross, power1, power2)


def _continuous_gram(
    data: NDArray[Any],
    Fs: int,
    fpass: list[float],
    tapers: NDArray[Any] | tuple[float, int],
    movingwin: tuple[float, float],
    other: NDArray[Any] | None = None,
    support: NDArray[np.float64] | None = None,
) -> pd.DataFrame:
    windows = _continuous_windows(data, Fs, fpass, movingwin, support)
    centers = (windows.starts + windows.length / 2) / Fs
    values = np.empty((len(windows.frequencies), len(windows.starts)), dtype=float)
    for i, (j1, j2) in enumerate(
        _continuous_window_ffts(data, other, Fs, tapers, windows)
    ):
        values[:, i] = (
            np.mean(np.abs(j1) ** 2, axis=1) if j2 is None else _coherence(j1, j2)
        )
    return _gram_frame(values, windows.frequencies, centers)


def mtspecgramc(
    data: NDArray[Any],
    Fs: int,
    fpass: list[float],
    tapers: NDArray[Any] | tuple[float, int],
    *,
    movingwin: tuple[float, float],
    time_support: _TimeSupport | None = None,
) -> pd.DataFrame:
    """
    Estimate a continuous-signal multitaper spectrogram.

    Parameters
    ----------
    data : np.ndarray
        1D uniformly sampled signal.
    Fs : int
        Sampling frequency in Hz.
    fpass : list of float
        Inclusive frequency range in Hz, from zero to Nyquist.
    tapers : np.ndarray or tuple of (float, int)
        DPSS time-bandwidth product and number of tapers, ``(NW, K)``.
    movingwin : tuple of float
        Window duration and step in seconds, rounded to whole samples.
    time_support : array-like, optional
        Half-open interval or array of intervals in seconds from the first
        sample. Only complete windows inside the intervals are returned.

    Returns
    -------
    pd.DataFrame
        Power density as in ``mtspectrumc``. Rows are frequencies in Hz and
        columns are centers of complete windows in seconds, measured from
        the start of the signal.

    Examples
    --------
    >>> import numpy as np
    >>> from neuro_py.process import pychronux as px
    >>> fs = 128
    >>> t = np.arange(4 * fs) / fs
    >>> signal = np.sin(2 * np.pi * (6 * t + 0.5 * t**2))  # chirp from 6 to 10 Hz
    >>> gram = px.mtspecgramc(signal, fs, [1, 20], (2, 3), movingwin=(2, 1))
    >>> gram.columns.tolist()  # window centers in seconds
    [1.0, 2.0, 3.0]
    """
    return _continuous_gram(
        data, Fs, fpass, tapers, movingwin, support=_support_array(time_support)
    )


def mtcohgramc(
    data1: NDArray[Any],
    data2: NDArray[Any],
    Fs: int,
    fpass: list[float],
    tapers: NDArray[Any] | tuple[float, int],
    *,
    movingwin: tuple[float, float],
    time_support: _TimeSupport | None = None,
) -> pd.DataFrame:
    """
    Estimate a continuous-signal multitaper coherogram.

    Parameters
    ----------
    data1, data2 : np.ndarray
        Aligned, equally sized 1D signals.
    Fs : int
        Sampling frequency in Hz.
    fpass : list of float
        Inclusive frequency range in Hz, from zero to Nyquist.
    tapers : np.ndarray or tuple of (float, int)
        DPSS time-bandwidth product and number of tapers, ``(NW, K)``.
    movingwin : tuple of float
        Window duration and step in seconds, rounded to whole samples.
    time_support : array-like, optional
        Half-open interval or array of intervals in seconds from the first
        sample. Only complete windows inside the intervals are returned.

    Returns
    -------
    pd.DataFrame
        Magnitude-squared coherence. Rows are frequencies in Hz and columns
        are centers of complete windows in seconds from the start of the
        signals. Frequencies with zero power in either signal are NaN.

    Examples
    --------
    >>> import numpy as np
    >>> from neuro_py.process import pychronux as px
    >>> fs = 128
    >>> t = np.arange(4 * fs) / fs
    >>> x = np.sin(2 * np.pi * 8 * t)
    >>> y = np.cos(2 * np.pi * 8 * t) + 0.5 * np.sin(2 * np.pi * 15 * t)
    >>> gram = px.mtcohgramc(x, y, fs, [1, 20], (2, 3), movingwin=(2, 1))
    >>> bool((gram.loc[8.0] > 0.9).all())  # high coherence at 8 Hz in each window
    True
    """
    _check_continuous_pair(data1, data2)
    return _continuous_gram(
        data1,
        Fs,
        fpass,
        tapers,
        movingwin,
        other=data2,
        support=_support_array(time_support),
    )


def mtspecgrampt(
    data: NDArray[Any],
    Fs: int,
    fpass: list[float],
    *,
    movingwin: tuple[float, float],
    NW: float = 2.5,
    n_tapers: int = 4,
    time_support: _TimeSupport | None = None,
) -> pd.DataFrame:
    """
    Estimate a multitaper spectrogram of one spike train.

    Parameters
    ----------
    data : np.ndarray
        One 1D array of finite spike times in seconds.
    Fs : int
        Taper interpolation grid rate in Hz.
    fpass : list of float
        Inclusive frequency range in Hz, from zero to Nyquist.
    movingwin : tuple of float
        Window duration and step in seconds, rounded to whole grid samples.
    NW : float, optional
        DPSS time-bandwidth product. Default is 2.5.
    n_tapers : int, optional
        Number of DPSS tapers. Default is 4.
    time_support : array-like, optional
        Observation interval or array of intervals in seconds. Windows never
        cross gaps. Defaults to the first and last spike, and is required
        for an empty spike train.

    Returns
    -------
    pd.DataFrame
        Power density as in ``mtspectrumpt``. Rows are frequencies in Hz
        and columns are centers of complete windows on the spike time axis.
        Windows include their left endpoint and exclude their right endpoint.
        Windows with no spikes have zero power.

    Examples
    --------
    >>> import numpy as np
    >>> from neuro_py.process import pychronux as px
    >>> spikes = np.arange(0.06, 4, 0.125)  # seconds
    >>> gram = px.mtspecgrampt(
    ...     spikes, Fs=128, fpass=[1, 20], movingwin=(2, 1), time_support=[0, 4]
    ... )
    >>> gram.columns.tolist()  # window centers in seconds
    [1.0, 2.0, 3.0]
    """
    times = _spike_times(data)
    support = _support_array(time_support)
    if support is None:
        if not times.size:
            raise ValueError("time_support is required for an empty spike train")
        support = np.array([times.min(), times.max()])
    windows = _point_windows(support, Fs, fpass, movingwin, NW, n_tapers)
    values = np.zeros((len(windows.frequencies), len(windows.starts)), dtype=float)
    times, left, right = _point_window_slices(times, windows)
    for i in np.flatnonzero(right > left):
        j = _point_window_fft(times[left[i] : right[i]], windows.starts[i], windows)
        values[:, i] = np.mean(np.abs(j) ** 2, axis=1)
    return _gram_frame(values, windows.frequencies, windows.centers)
