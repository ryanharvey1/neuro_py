import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pywt
from scipy import signal
from scipy.linalg import toeplitz
from scipy.signal import butter, filtfilt, firwin


def filter_signal(
    sig: np.ndarray,
    fs: float,
    pass_type: str,
    f_range: Union[float, Tuple[float, float]],
    filter_type: str = "fir",
    n_cycles: int = 3,
    n_seconds: Optional[float] = None,
    butterworth_order: int = 4,
    remove_edges: bool = True,
) -> np.ndarray:
    """
    Filter a neural signal using an FIR or IIR filter.

    Parameters
    ----------
    sig : np.ndarray
        Time series to be filtered.
    fs : float
        Sampling rate, in Hz.
    pass_type : {'bandpass', 'bandstop', 'lowpass', 'highpass'}
        Type of filter to apply.
    f_range : float or tuple of float
        Frequency range for filtering. For 'lowpass' and 'highpass', a single float can be provided.
        For 'bandpass' and 'bandstop', a tuple specifying (f_low, f_high) is required.
    filter_type : {'fir', 'iir'}, optional
        Type of filter to apply: 'fir' for FIR or 'iir' for IIR (Butterworth). Default is 'fir'.
    n_cycles : int, optional
        Number of cycles to define the kernel length for FIR filters. Default is 3.
    n_seconds : float, optional
        Length of the FIR filter in seconds. Overrides `n_cycles` if specified. Ignored for IIR.
    butterworth_order : int, optional
        Order of the Butterworth filter. Only applies to IIR filters. Default is 4.
    remove_edges : bool, optional
        If True, replace samples within half the kernel length with NaN (FIR filters only). Default is True.

    Returns
    -------
    np.ndarray
        Filtered time series.

    Examples
    --------
    Apply a lowpass FIR filter to a signal:

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from your_module import filter_signal
    >>> fs = 1000  # Sampling rate (Hz)
    >>> t = np.linspace(0, 1, fs, endpoint=False)
    >>> sig = np.sin(2 * np.pi * 1 * t) + 0.5 * np.sin(2 * np.pi * 50 * t)  # Signal with 1Hz and 50Hz components
    >>> pass_type = 'lowpass'
    >>> f_range = 10  # Lowpass filter at 10 Hz
    >>> filt_sig = filter_signal(sig, fs, pass_type, f_range, filter_type='fir')
    >>> plt.plot(t, sig, label='Original Signal')
    >>> plt.plot(t, filt_sig, label='Filtered Signal')
    >>> plt.legend()
    >>> plt.show()
    """

    # Validate pass_type
    if pass_type not in ["bandpass", "bandstop", "lowpass", "highpass"]:
        raise ValueError(
            "`pass_type` must be one of: 'bandpass', 'bandstop', 'lowpass', 'highpass'."
        )

    # Ensure `f_range` is properly defined for the filter type
    if isinstance(f_range, (int, float)):
        if pass_type == "lowpass":
            f_range = (None, f_range)  # Convert single value to tuple for lowpass
        elif pass_type == "highpass":
            f_range = (f_range, None)  # Convert single value to tuple for highpass
        else:
            raise ValueError(
                "`f_range` must be a tuple for 'bandpass' or 'bandstop' filters."
            )

    # Validate bandpass/bandstop filters
    if pass_type in ["bandpass", "bandstop"]:
        if not isinstance(f_range, tuple) or f_range[0] is None or f_range[1] is None:
            raise ValueError(
                "Both frequencies must be specified for 'bandpass' and 'bandstop' filters."
            )

    # Nyquist frequency
    nyquist = fs / 2

    # FIR filter implementation
    if filter_type == "fir":
        # Compute filter kernel length
        if n_seconds is not None:
            kernel_len = int(n_seconds * fs)
        else:
            kernel_len = (
                int((n_cycles / f_range[0]) * fs)
                if f_range[0]
                else int((n_cycles / f_range[1]) * fs)
            )
        if kernel_len % 2 == 0:
            kernel_len += 1  # Ensure kernel length is odd

        # Define FIR filter coefficients
        if pass_type in ["bandpass", "bandstop"]:
            fir_coefs = firwin(
                kernel_len,
                [f_range[0] / nyquist, f_range[1] / nyquist],
                pass_zero=(pass_type == "bandstop"),
            )
        elif pass_type == "lowpass":
            fir_coefs = firwin(kernel_len, f_range[1] / nyquist, pass_zero=True)
        elif pass_type == "highpass":
            fir_coefs = firwin(kernel_len, f_range[0] / nyquist, pass_zero=False)

        # Apply the FIR filter
        sig_filt = np.convolve(sig, fir_coefs, mode="same")

    # IIR filter implementation
    elif filter_type == "iir":
        # Design a Butterworth filter
        if pass_type in ["bandpass", "bandstop"]:
            b, a = butter(
                butterworth_order,
                [f_range[0] / nyquist, f_range[1] / nyquist],
                btype=pass_type,
            )
        elif pass_type == "lowpass":
            b, a = butter(butterworth_order, f_range[1] / nyquist, btype="low")
        elif pass_type == "highpass":
            b, a = butter(butterworth_order, f_range[0] / nyquist, btype="high")

        # Apply the IIR filter
        sig_filt = filtfilt(b, a, sig)

    else:
        raise ValueError("`filter_type` must be 'fir' or 'iir'.")

    # Optionally remove edges
    if remove_edges and filter_type == "fir":
        edge_len = kernel_len // 2
        sig_filt[:edge_len] = np.nan
        sig_filt[-edge_len:] = np.nan

    return sig_filt


def yule_walker(x, order=1, method="adjusted", df=None, inv=False, demean=True):
    """
    Estimate AR(p) parameters from a sequence using the Yule-Walker equations.

    Adjusted or maximum-likelihood estimator (mle)

    Parameters
    ----------
    x : array_like
        A 1d array.
    order : int, optional
        The order of the autoregressive process.  Default is 1.
    method : str, optional
       Method can be 'adjusted' or 'mle' and this determines
       denominator in estimate of autocorrelation function (ACF) at
       lag k. If 'mle', the denominator is n=X.shape[0], if 'adjusted'
       the denominator is n-k.  The default is adjusted.
    df : int, optional
       Specifies the degrees of freedom. If `df` is supplied, then it
       is assumed the X has `df` degrees of freedom rather than `n`.
       Default is None.
    inv : bool
        If inv is True the inverse of R is also returned.  Default is
        False.
    demean : bool
        True, the mean is subtracted from `X` before estimation.

    Returns
    -------
    rho : ndarray
        AR(p) coefficients computed using the Yule-Walker method.
    sigma : float
        The estimate of the residual standard deviation.

    Notes
    -----
    From statsmodels

    See https://en.wikipedia.org/wiki/Autoregressive_moving_average_model for
    further details.
    """

    if method not in ("adjusted", "mle"):
        raise ValueError("ACF estimation method must be 'adjusted' or 'MLE'")
    x = np.array(x, dtype=np.float64)
    if demean:
        x -= x.mean()
    n = df or x.shape[0]

    # this handles df_resid ie., n - p
    adj_needed = method == "adjusted"

    if x.ndim > 1 and x.shape[1] != 1:
        raise ValueError("expecting a vector to estimate AR parameters")
    r = np.zeros(order + 1, np.float64)
    r[0] = (x**2).sum() / n
    for k in range(1, order + 1):
        r[k] = (x[0:-k] * x[k:]).sum() / (n - k * adj_needed)
    R = toeplitz(r[:-1])

    try:
        rho = np.linalg.solve(R, r[1:])
    except np.linalg.LinAlgError as err:
        if "Singular matrix" in str(err):
            warnings.warn("Matrix is singular. Using pinv.")
            rho = np.linalg.pinv(R) @ r[1:]
        else:
            raise

    sigmasq = r[0] - (r[1:] * rho).sum()
    if not np.isnan(sigmasq) and sigmasq > 0:
        sigma = np.sqrt(sigmasq)
    else:
        sigma = np.nan
    if inv:
        return rho, sigma, np.linalg.inv(R)
    else:
        return rho, sigma


def whiten_lfp(lfp: np.ndarray, order: int = 2) -> np.ndarray:
    """
    Perform temporal whitening of Local Field Potential (LFP) data using an Autoregressive (AR) model.

    This function applies temporal whitening to LFP data by fitting an AR model of the specified order
    and using the model to remove temporal correlations, resulting in a 'whitened' signal.

    Parameters
    ----------
    lfp : ndarray
        A 1D numpy array containing the LFP data.
    order : int, optional (default=2)
        The order of the AR model to be used for whitening the LFP data.

    Returns
    -------
    ndarray
        The temporally whitened LFP data as a 1D numpy array.
    """

    rho, _ = yule_walker(lfp, order=order)

    a = np.concatenate(([1.0], -rho))

    # Apply the whitening filter to the LFP data and return the result as a 1D array
    return signal.convolve(lfp, a, "same")


def event_triggered_wavelet(
    signal: np.ndarray,
    timestamps: np.ndarray,
    events: np.ndarray,
    max_lag: float = 1,
    freq_min: float = 4,
    freq_max: float = 100,
    freq_step: float = 4,
    return_pandas: bool = False,
    parallel: bool = True,
    whiten: bool = True,
    whiten_order: int = 2,
    **kwargs,
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    Tuple[pd.DataFrame, pd.Series],
]:
    """
    Compute the event-triggered wavelet transform of a signal.

    Parameters
    ----------
    signal : 1d array
        Time series.
    timestamps : 1d array
        Time points for each sample in the signal.
    events : 1d array
        Time points of events.
    max_lag : float
        Maximum lag to consider, in seconds.
    freq_min : float
        Minimum frequency to consider, in Hz.
    freq_max : float
        Maximum frequency to consider, in Hz.
    freq_step : float
        Step size for frequency range, in Hz.
    return_pandas : bool
        If True, return the output as pandas objects.
    parallel : bool
        If True, use parallel processing to compute the wavelet transform.
    whiten : bool
        If True, whiten the signal before computing the wavelet transform.
    whiten_order : int
        Order of the autoregressive model used for whitening.
    kwargs
        Additional keyword arguments to pass to `compute_wavelet_transform`.

    Returns
    -------
    mwt : 2d array
        Time frequency representation of the input signal.
    sigs : 1d array
        Average signal.
    times : 1d array
        Time points for each sample in the output.
    freqs : 1d array
        Frequencies used in the wavelet transform.

    Examples
    -------
    >>> from neuro_py.lfp.spectral import event_triggered_wavelet

    >>> basepath = r"Z:\\Data\\hpc_ctx_project\\HP04\\day_34_20240503"

    >>> # load lfp
    >>> nChannels, fs, _, _ = loading.loadXML(basepath)
    >>> # Load the LFP data
    >>> lfp, ts = loading.loadLFP(basepath, n_channels=nChannels,
    >>>                channel=23,
    >>>                frequency=fs)
    >>> # load events
    >>> opto = loading.load_events(basepath, epoch_name="optoStim")
    >>> opto = opto.merge(gap=.1)

    >>> # compute event triggered averate
    >>> mwt, sigs, times, freqs = event_triggered_wavelet(
    >>>    lfp,
    >>>    ts,
    >>>    opto.starts,
    >>> )

    >>> # plot
    >>> plt.figure(figsize=set_size("thesis", fraction=1, subplots=(1, 1)))

    >>> im = plt.imshow(
    >>>     abs(mwt),
    >>>     aspect="auto",
    >>>     extent=[times[0], times[-1], freqs[-1], freqs[0]],
    >>>     cmap="magma",
    >>>     vmax=600,
    >>>     vmin=50,
    >>> )
    >>> plt.axhline(23, color="orange", linestyle="--", label="23hz")

    >>> plt.yscale("log")
    >>> # move legend outside of plot
    >>> plt.legend(loc="upper right", bbox_to_anchor=(1.1, 1.1), frameon=False)

    >>> plt.gca().invert_yaxis()

    >>> plt.colorbar(location="top", label="Power (uV^2)")
    >>> # move colorbar more to the left
    >>> plt.gcf().axes[1].set_position([0.5, 0.8, 0.4, 0.6])


    >>> plt.gca().set_ylabel("Frequency (Hz)")

    >>> plt.gca().set_xlabel("Time from opto stim (s)")

    >>> plt.twinx()
    >>> plt.yscale("linear")
    >>> plt.axvline(0, color="k", linestyle="--")
    >>> plt.axvline(0.5, color="k", linestyle="--")
    >>> plt.plot(times, sigs, "w", linewidth=0.5)


    >>> # plt.gca().set_xlabel('Time (s)')
    >>> plt.gca().set_ylabel("Voltage (uV)")
    >>> plt.gca().set_title("PFC during 23Hz stim in behavior", y=1)
    """

    signal_ = signal.copy()
    if whiten:
        signal_ = whiten_lfp(signal, order=whiten_order)

    # set up frequency range
    freqs = np.arange(freq_min, freq_max, freq_step)
    # set up time range
    ds = timestamps[1] - timestamps[0]
    fs = 1 / ds
    # Create times array based on the sample rate (fs)
    times = np.arange(-max_lag, max_lag, 1 / fs)
    # Number of samples corresponding to the time window around each event
    n_samples = int(max_lag * 2 * fs)

    # Ensure the length of times matches n_samples
    if len(times) != n_samples:
        times = np.linspace(-max_lag, max_lag, n_samples)

    n_freqs = len(freqs)
    n_samples = len(times)

    # set up mwt and sigs to store results
    mwt = np.zeros((n_freqs, n_samples))
    sigs = np.zeros(n_samples)

    event_i = 0

    def process_event(start):
        nonlocal event_i
        nonlocal mwt
        nonlocal sigs

        if start + max_lag > timestamps.max() or start - max_lag < timestamps.min():
            return None, None

        idx = (timestamps >= start - max_lag) & (timestamps <= start + max_lag)

        mwt_partial = np.abs(
            compute_wavelet_transform(sig=signal_[idx], fs=fs, freqs=freqs, **kwargs)
        )

        return mwt_partial, signal[idx]

    if parallel:
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(process_event, events))

        for mwt_partial, sig_partial in results:
            if mwt_partial is not None:
                # samples might be missing if the event is too close to the edge
                if mwt_partial.shape[1] != n_samples:
                    continue
                mwt += mwt_partial
                sigs += sig_partial
                event_i += 1
    else:
        for start in events:
            mwt_partial, sig_partial = process_event(start)
            if mwt_partial is not None:
                mwt += mwt_partial
                sigs += sig_partial
                event_i += 1

    mwt /= event_i
    sigs /= event_i

    if return_pandas:
        mwt = pd.DataFrame(mwt.T, index=times, columns=freqs)
        sigs = pd.Series(sigs, index=times)
        return mwt, sigs
    else:
        return mwt, sigs, times, freqs


def compute_wavelet_transform(
    sig: np.ndarray,
    fs: float,
    freqs: Union[np.ndarray, List[float], Tuple[float, float, Optional[float]]],
    wavelet: str = "cmor",
    center_frequency: float = 0.5,
    bandwidth_frequency: float = 1.5,
    method="conv",
) -> np.ndarray:
    """
    Compute the time-frequency representation of a signal using Morlet wavelets via PyWavelets.

    Parameters
    ----------
    sig : np.ndarray
        Time series data (1D array).
    fs : float
        Sampling rate, in Hz.
    freqs : Union[np.ndarray, List[float], Tuple[float, float, Optional[float]]]
        Frequencies to analyze with Morlet wavelets.
        - If an array or list, specifies exact frequency values.
        - If a tuple, defines a frequency range as `(freq_start, freq_stop[, freq_step])`.
            The `freq_step` is optional and defaults to 1. Range is inclusive of `freq_stop`.
    wavelet : str, optional
        The name of the wavelet to use for the CWT. Default is 'cmor'.
        - wavelist = pywt.wavelist(kind='continuous') to get a list of available wavelets.
    center_frequency : float, optional
        The center frequency of the Morlet wavelet.
    bandwidth_frequency : float, optional
        The bandwidth of the Morlet wavelet.
    method : {'conv', 'fft'}, optional
        The method used to compute the CWT. Can be any of:
            - ``conv`` uses ``numpy.convolve``.
            - ``fft`` uses frequency domain convolution.
            - ``auto`` uses automatic selection based on an estimate of the
              computational complexity at each scale.
    Returns
    -------
    np.ndarray
        The time-frequency representation of the input signal. Shape is `(n_freqs, n_time_points)`.

    Notes
    -----
    This function uses `pywt.cwt` with Morlet wavelets to compute the time-frequency representation.
    """

    # Convert the frequency range to an array if it is given as a list or tuple
    if isinstance(freqs, (tuple, list)):
        freqs = (
            np.arange(*freqs)
            if len(freqs) == 3
            else np.linspace(freqs[0], freqs[1], 100)
        )

    # Time step
    dt = 1 / fs

    # Define the wavelet name
    wavelet_name = f"{wavelet}{center_frequency}-{bandwidth_frequency}"

    # Convert frequencies to scales
    scales = pywt.frequency2scale(wavelet_name, freqs / fs)

    # Perform the Continuous Wavelet Transform
    coefficients, _ = pywt.cwt(
        sig, scales, wavelet_name, sampling_period=dt, method=method
    )

    return coefficients
