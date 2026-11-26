from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.typing import NDArray
from scipy.signal.windows import dpss


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
) -> Tuple[NDArray[Any], float, float]:
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
    Nsp : float
        Total number of spikes in data.

    Notes
    -----
    The function computes the multitaper FFT of spike times using
    the specified tapers and returns the FFT result, mean spikes,
    and total spike count.
    """
    if nfft < len(t):
        raise ValueError("nfft must be at least the number of taper samples")
    K = tapers.shape[1]
    nfreq = len(f)

    # get the FFT of the tapers
    H = np.zeros((nfft, K), dtype=np.complex128)
    for i in np.arange(K):
        H[:, i] = np.fft.fft(tapers[:, i], nfft, axis=0)

    H = H[findx, :]
    w = 2 * np.pi * f
    dtmp = data
    indx = np.logical_and(dtmp >= np.min(t), dtmp <= np.max(t))
    if len(indx):
        dtmp = dtmp[indx]
    Nsp = len(dtmp)

    # get the mean spike rate
    Msp = Nsp / len(t)

    if Msp != 0:
        # Interpolate spike times for each taper
        data_proj = np.empty((len(dtmp), K))
        for i in range(K):
            data_proj[:, i] = np.interp(dtmp, t, tapers[:, i])

        def compute_J(k):
            J_k = np.zeros(nfreq, dtype=np.complex128)
            for i, freq in enumerate(w):
                phase = -1j * freq * (dtmp - t[0])
                J_k[i] = np.sum(np.exp(phase) * data_proj[:, k])
            return J_k

        J = np.array(Parallel(n_jobs=-1)(delayed(compute_J)(k) for k in range(K))).T

        J -= H * Msp
    else:
        # No spikes: return zeros
        J = np.zeros((nfreq, K), dtype=np.complex128)

    return J, Msp, Nsp


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
        raise ValueError("Each spike train must be a one-dimensional array of finite times")
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
            t[0] > time_support[0] + dt * 1e-6
            or t[-1] < time_support[1] - dt * 1e-6
        ):
            raise ValueError("tapers_ts must cover time_support")
        return t
    if time_support is not None:
        start, stop = time_support
    else:
        nonempty = [train for train in trains if train.size]
        if not nonempty:
            raise ValueError("time_support or tapers_ts is required for empty spike trains")
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
        raise ValueError("Invalid frequency range: fpass[0] should be less than fpass[1].")
    if not 0 <= fpass[0] <= fpass[1] <= Fs / 2:
        raise ValueError("fpass must lie between 0 and the Nyquist frequency (Fs / 2)")
    return getfgrid(Fs, nfft, fpass)


def _restrict_spike_times(
    times: NDArray[np.float64], support: list[float] | None
) -> NDArray[np.float64]:
    if support is None:
        return times
    return times[(times >= support[0]) & (times <= support[1])]


def mtspectrumpt(
    data: NDArray[Any] | Sequence[NDArray[Any]],
    Fs: int,
    fpass: list[float],
    NW: Union[int, float] = 2.5,
    n_tapers: int = 4,
    time_support: Union[list[float], None] = None,
    tapers: Union[NDArray[Any], None] = None,
    tapers_ts: Union[NDArray[Any], None] = None,
    nfft: Optional[int] = None,
) -> pd.DataFrame:
    """
    Multitaper power spectrum estimation for point process data.

    Parameters
    ----------
    data : np.ndarray or sequence of np.ndarray
        One 1D array of spike times in seconds, or one array per unit.
    Fs : int
        Taper interpolation grid rate in Hz. Spike times remain in seconds.
    fpass : list of float
        Frequency range to evaluate.
    NW : Union[int, float], optional
        Time-bandwidth product (default is 2.5).
    n_tapers : int, optional
        Number of tapers (default is 4).
    time_support : Union[list, None], optional
        Observation interval ``[start, stop]`` in seconds. Spike times outside
        it are excluded. When omitted, use ``tapers_ts`` or the first and last spike.
    tapers : Union[np.ndarray, None], optional
        Precomputed, unscaled DPSS tapers in samples-by-tapers or
        tapers-by-samples order. Internally scaled by ``sqrt(Fs)``. Default is None.
    tapers_ts : Union[np.ndarray, None], optional
        Optional taper timestamps in seconds, uniformly spaced by ``1 / Fs``.
        Pass the matching grid when reusing precomputed tapers.
    nfft : Optional[int], optional
        FFT length, at least the length of the taper grid. Defaults to the
        next power of two.

    Returns
    -------
    pd.DataFrame
        Power density indexed by frequency in Hz, with one column per unit.
        Positive-frequency bins retain the two-sided normalization.


    Examples
    -------
    >>> spec = pychronux.mtspectrumpt(
    >>>    st.data,
    >>>    100,
    >>>    [1, 20],
    >>>    NW=3,
    >>>    n_tapers=5,
    >>>    time_support=[st.support.start, st.support.stop],
    >>> )
    """

    # check data
    if len(data) == 0:
        return pd.DataFrame()

    if isinstance(data, np.ndarray) and data.ndim == 1 and np.ndim(data[0]) == 0:
        trains = [_spike_times(np.asarray(data, dtype=np.float64))]
    else:
        trains = [_spike_times(train) for train in data]
    t = _point_time_grid(trains, Fs, time_support, tapers_ts)
    nfft = _fft_size(len(t), nfft)
    f, findx = _spectrum_grid(Fs, nfft, fpass)
    if tapers is None:
        tapers = dpss(len(t), NW, n_tapers).T
    # Apply the same Chronux scaling to generated and supplied DPSS tapers.
    tapers = _point_tapers(tapers, len(t)) * np.sqrt(Fs)

    spec = np.empty((len(f), len(trains)))
    for i, train in enumerate(trains):
        observed = _restrict_spike_times(train, time_support)
        J, _, _ = mtfftpt(observed, tapers, nfft, t, f, findx)
        spec[:, i] = np.mean(np.abs(J) ** 2, axis=1)

    return pd.DataFrame(spec, index=f)


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
    """
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
    time_support: list[float] | None = None,
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
    time_support : list of float, optional
        Observation interval in seconds. Defaults to the first and last spike.

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
    """

    times = _spike_times(times)
    t = _point_time_grid([times], Fs, time_support, None)
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
    j1, _, _ = mtfftpt(first, tapers, nfft, t, f, findx)
    j2, _, _ = mtfftpt(second, tapers, nfft, t, f, findx)
    return f, j1, j2


def mtcsdpt(
    data1: NDArray[Any],
    data2: NDArray[Any],
    Fs: int,
    fpass: list[float],
    NW: Union[int, float] = 2.5,
    n_tapers: int = 4,
    time_support: Union[list[float], None] = None,
    tapers: Union[NDArray[Any], None] = None,
    tapers_ts: Union[NDArray[Any], None] = None,
    nfft: Optional[int] = None,
    *,
    return_complex: bool = False,
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
    time_support : Union[list, None], optional
        Observation interval in seconds. When omitted, use ``tapers_ts`` or
        the joint first and last spike times.
    tapers : Union[np.ndarray, None], optional
        Precomputed, unscaled DPSS tapers in samples-by-tapers or
        tapers-by-samples order. Internally scaled by ``sqrt(Fs)``. Default is None.
    tapers_ts : Union[np.ndarray, None], optional
        Optional taper timestamps in seconds, uniformly spaced by ``1 / Fs``.
    nfft : Optional[int], optional
        FFT length, at least the length of the taper grid. Default is None.

    return_complex : bool, optional
        Return the full complex cross-spectrum when True. Default False retains
        the historical real cospectrum for existing callers.

    Returns
    -------
    pd.DataFrame
        Cross-spectral density (or its real component by default).
    """
    f, J1, J2 = _point_pair(
        data1, data2, Fs, fpass, NW, n_tapers, time_support, tapers, tapers_ts, nfft
    )

    # Cross-spectral density: Sxy = mean(conjugate(J1) * J2)
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
    time_support: Union[list[float], None] = None,
    tapers: Union[NDArray[Any], None] = None,
    tapers_ts: Union[NDArray[Any], None] = None,
    nfft: Optional[int] = None,
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
    time_support : Union[list, None], optional
        Observation interval in seconds. When omitted, use ``tapers_ts`` or
        the joint first and last spike times.
    tapers : Union[np.ndarray, None], optional
        Precomputed, unscaled DPSS tapers in samples-by-tapers or
        tapers-by-samples order. Default is None.
    tapers_ts : Union[np.ndarray, None], optional
        Optional taper timestamps in seconds, uniformly spaced by ``1 / Fs``.
    nfft : Optional[int], optional
        FFT length, at least the length of the taper grid. Default is None.

    Returns
    -------
    pd.DataFrame
        Coherence between the two point processes.

    Notes
    -----
    Both auto-spectra and the complex cross-spectrum use a shared time grid,
    support, taper set and FFT size. Supplied tapers may be samples-by-tapers
    or its transpose. Zero-power frequencies have undefined (NaN) coherence.
    With one taper, coherence is identically one wherever both powers are
    nonzero. Use multiple tapers for a meaningful estimate.
    """
    f, J1, J2 = _point_pair(
        data1, data2, Fs, fpass, NW, n_tapers, time_support, tapers, tapers_ts, nfft
    )
    return pd.DataFrame({"Coherence": _coherence(J1, J2)}, index=f)


def _coherence(J1: NDArray[Any], J2: NDArray[Any]) -> NDArray[np.float64]:
    cross = np.mean(J1.conj() * J2, axis=1)
    power_product = np.mean(np.abs(J1) ** 2, axis=1) * np.mean(np.abs(J2) ** 2, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        # Suppress roundoff above the Cauchy–Schwarz bound; zero power stays NaN.
        return np.clip(np.abs(cross) ** 2 / power_product, 0, 1)


def _check_continuous_pair(data1: NDArray[Any], data2: NDArray[Any]) -> None:
    if data1.ndim != 1 or data1.shape != data2.shape or not data1.size:
        raise ValueError("data1 and data2 must be nonempty 1D arrays of equal length")


def mtcoherencec(
    data1: NDArray[Any],
    data2: NDArray[Any],
    Fs: int,
    fpass: list[float],
    tapers: NDArray[Any] | tuple[float, int],
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

    Returns
    -------
    pd.Series
        Magnitude-squared coherence indexed by frequency in Hz. Frequencies
        with zero power in either signal have NaN coherence.

    Notes
    -----
    This is the square of the magnitude returned by Chronux ``coherencyc``.
    With one taper, coherence is one wherever both spectra have power.
    """
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
        raise ValueError("movingwin must contain positive finite duration and step values")
    length = int(np.floor(duration * Fs + 0.5))
    stride = int(np.floor(step * Fs + 0.5))
    if length < 2 or stride < 1:
        raise ValueError("movingwin must span at least two samples and step by at least one")
    return length, stride


def _gram_frame(
    values: NDArray[np.float64], f: NDArray[Any], centers: NDArray[Any]
) -> pd.DataFrame:
    return pd.DataFrame(
        values,
        index=pd.Index(f, name="frequency_hz"),
        columns=pd.Index(centers, name="time_s"),
    )


def _continuous_gram(
    data: NDArray[Any],
    Fs: int,
    fpass: list[float],
    tapers: NDArray[Any] | tuple[float, int],
    movingwin: tuple[float, float],
    other: NDArray[Any] | None = None,
) -> pd.DataFrame:
    if data.ndim != 1:
        raise ValueError("data must be a 1D signal")
    length, stride = _window_samples(movingwin, Fs)
    nfft = _fft_size(length)
    f, findx = _spectrum_grid(Fs, nfft, fpass)
    starts = np.arange(0, len(data) - length + 1, stride)
    centers = (starts + length / 2) / Fs
    values = np.empty((len(f), len(starts)), dtype=float)
    if starts.size:
        bank = dpsschk(tapers, length, Fs)
        for i, start in enumerate(starts):
            window = slice(start, start + length)
            j1 = mtfftc(data[window], bank, nfft, Fs)[findx]
            if other is None:
                values[:, i] = np.mean(np.abs(j1) ** 2, axis=1)
            else:
                j2 = mtfftc(other[window], bank, nfft, Fs)[findx]
                values[:, i] = _coherence(j1, j2)
    return _gram_frame(values, f, centers)


def mtspecgramc(
    data: NDArray[Any],
    Fs: int,
    fpass: list[float],
    tapers: NDArray[Any] | tuple[float, int],
    *,
    movingwin: tuple[float, float],
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

    Returns
    -------
    pd.DataFrame
        Power density as in ``mtspectrumc``. Rows are frequencies in Hz and
        columns are centers of complete windows in seconds, measured from
        the start of the signal.
    """
    return _continuous_gram(data, Fs, fpass, tapers, movingwin)


def mtcohgramc(
    data1: NDArray[Any],
    data2: NDArray[Any],
    Fs: int,
    fpass: list[float],
    tapers: NDArray[Any] | tuple[float, int],
    *,
    movingwin: tuple[float, float],
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

    Returns
    -------
    pd.DataFrame
        Magnitude-squared coherence. Rows are frequencies in Hz and columns
        are centers of complete windows in seconds from the start of the
        signals. Frequencies with zero power in either signal are NaN.
    """
    _check_continuous_pair(data1, data2)
    return _continuous_gram(data1, Fs, fpass, tapers, movingwin, other=data2)


def mtspecgrampt(
    data: NDArray[Any],
    Fs: int,
    fpass: list[float],
    *,
    movingwin: tuple[float, float],
    NW: float = 2.5,
    n_tapers: int = 4,
    time_support: list[float] | None = None,
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
    time_support : list of float, optional
        Observation interval in seconds, defaulting to the first and last
        spike. Required for an empty spike train.

    Returns
    -------
    pd.DataFrame
        Power density as in ``mtspectrumpt``. Rows are frequencies in Hz
        and columns are centers of complete windows on the spike time axis.
        Windows include their left endpoint and exclude their right endpoint.
        Windows with no spikes have zero power.
    """
    times = np.sort(_spike_times(data))
    length, stride = _window_samples(movingwin, Fs)
    if time_support is None:
        if not times.size:
            raise ValueError("time_support is required for an empty spike train")
        start, stop = times[0], times[-1]
    else:
        if len(time_support) != 2 or not np.all(np.isfinite(time_support)):
            raise ValueError("time_support must contain two finite times")
        start, stop = time_support
        if start >= stop:
            raise ValueError("time_support must have start < stop")

    span = (stop - start) * Fs
    n_samples = int(np.floor(span + 4 * np.spacing(span)))
    offsets = np.arange(0, n_samples - length + 1, stride)
    starts = start + offsets / Fs
    stops = start + (offsets + length) / Fs
    centers = start + (offsets + length / 2) / Fs
    t = np.arange(-1, length + 2) / Fs
    nfft = _fft_size(len(t))
    f, findx = _spectrum_grid(Fs, nfft, fpass)
    values = np.zeros((len(f), len(starts)), dtype=float)
    if starts.size:
        bank = dpss(len(t), NW, n_tapers).T * np.sqrt(Fs)
        left = np.searchsorted(times, starts)
        right = np.searchsorted(times, stops)
        for i, (lo, hi) in enumerate(zip(left, right)):
            if hi > lo:
                j, _, _ = mtfftpt(times[lo:hi] - starts[i], bank, nfft, t, f, findx)
                values[:, i] = np.mean(np.abs(j) ** 2, axis=1)
    return _gram_frame(values, f, centers)
