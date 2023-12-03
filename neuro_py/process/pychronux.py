import numpy as np
from scipy.signal.windows import dpss as dpss_scipy
import pandas as pd
from spectrum import dpss
from typing import Union


def getfgrid(Fs: int, nfft: int, fpass: list):
    """
    get frequency grid for evaluation
    Inputs:
        Fs: sampling frequency
        nfft: number of points for fft
        fpass: frequency range to evaluate
    Outputs:
        f: frequency vector
        findx: frequency index
    """
    df = Fs / nfft
    f = np.arange(0, Fs + df, df)
    f = f[0:nfft]
    # findx = np.logical_and(f >= fpass[0], f <= fpass[-1])
    findx = (f >= fpass[0]) & (f <= fpass[-1])
    f = f[findx]
    return f, findx


def dpsschk(tapers, N, Fs):
    """
    check tapers
    Inputs:
        tapers: [NW, K] or [tapers, eigenvalues]
        N: number of points for fft
        Fs: sampling frequency
    Outputs:
        tapers: [tapers, eigenvalues]
    """
    tapers, eigs = dpss_scipy(
        N, NW=tapers[0], Kmax=tapers[1], sym=False, return_ratios=True
    )
    tapers = tapers * np.sqrt(Fs)
    tapers = tapers.T
    return tapers


def get_tapers(N, bandwidth, *, fs=1, min_lambda=0.95, n_tapers=None):
    """
    Compute tapers and associated energy concentrations for the Thomson
    multitaper method
    Parameters
    ----------
    N : int
        Length of taper
    bandwidth : float
        Bandwidth of taper, in Hz
    fs : float, optional
        Sampling rate, in Hz.
        Default is 1 Hz.
    min_lambda : float, optional
        Minimum energy concentration that each taper must satisfy.
        Default is 0.95.
    n_tapers : int, optional
        Number of tapers to compute
        Default is to use all tapers that satisfied 'min_lambda'.

    Returns
    -------
    tapers : np.ndarray, with shape (n_tapers, N)
    lambdas : np.ndarray, with shape (n_tapers, )
        Energy concentrations for each taper
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

    tapers, lambdas = dpss(N, NW, Kmax=K, norm=2, return_ratios=True)
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


# @njit(nopython=True)
def mtfftpt(
    data: np.ndarray,
    tapers: np.ndarray,
    nfft: int,
    t: np.ndarray,
    f: np.ndarray,
    findx: list,
):
    """
    mt fft for point process times
    Inputs:
        data: 1d array of spike times (in seconds)
        tapers: tapers from dpss
        nfft: number of points for fft
        t: time vector
        f: frequency vector
        findx: frequency index
    Outputs:
        J: fft of data
        Msp: number of spikes in data
        Nsp: number of spikes in data
    """
    K = tapers.shape[1]
    nfreq = len(f)
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
    Msp = Nsp / len(t)

    if Msp != 0:
        data_proj = np.empty((len(dtmp), K))
        for i in np.arange(K):
            data_proj[:, i] = np.interp(dtmp, t, tapers[:, i])
        exponential = np.exp(np.atleast_2d(-1j * w).T * (dtmp - t[0]))
        J = np.dot(exponential, data_proj) - H * Msp
    else:
        J = np.zeros((nfreq, K))

    return J, Msp, Nsp


def mtspectrumpt(
    data: np.ndarray,
    Fs: int,
    fpass: list,
    NW: Union[int, float] = 2.5,
    n_tapers: int = 4,
    time_support: Union[list, None] = None,
    tapers: Union[np.ndarray, None] = None,
    tapers_ts: Union[np.ndarray, None] = None,
) -> pd.DataFrame:
    """
    mtspectrumpt from chronux toolbox
    Inputs:
        data: array of spike times (in seconds)
        Fs: sampling frequency
        fpass: frequency range to evaluate
        NW: time-bandwidth product
        n_tapers: number of tapers
        tapers: [NW, K] or [tapers, eigenvalues]
        time_support: time range to evaluate
    Outputs:
        S: power spectrum

    example:
        spec = pychronux.mtspectrumpt(
            st.data,
            1250,
            [1, 20],
            NW=3,
            n_tapers=5,
            time_support=[st.support.start, st.support.stop],
        )
    """
    if time_support is not None:
        mintime, maxtime = time_support
    else:
        mintime = np.min(data)
        maxtime = np.max(data)
    dt = 1 / Fs

    if tapers is None:
        tapers_ts = np.arange(mintime - dt, maxtime + dt, dt)
        N = len(tapers_ts)
        tapers, eigens = dpss(N, NW, n_tapers)

    N = len(tapers_ts)
    # number of points in fft of prolates
    nfft = np.max([int(2 ** np.ceil(np.log2(N))), N])
    f, findx = getfgrid(Fs, nfft, fpass)


    spec = np.zeros((len(f), len(data)))
    for i, d in enumerate(data):
        J, Msp, Nsp = mtfftpt(d, tapers, nfft, tapers_ts, f, findx)
        spec[:, i] = np.real(np.mean(np.conj(J) * J, 1))

    spectrum_df = pd.DataFrame(index=f, columns=np.arange(len(data)), dtype=np.float64)
    spectrum_df[:] = spec
    return spectrum_df


def mtfftc(data, tapers, nfft, Fs):
    """
    mt fft for continuous data
    Inputs:
        data: 1d array
        tapers: [NW, K] or [tapers, eigenvalues]
        nfft: number of points for fft
        Fs: sampling frequency
    Outputs:
        J: fft of data
    """
    NC = len(data)
    NK, K = tapers.shape
    assert NK == NC, "length of tapers is incompatible with length of data"
    tmp = np.repeat(np.atleast_2d(data), K, 0).T
    tmp2 = tmp * tapers
    J = np.fft.fft(tmp2.T, nfft) / float(Fs)
    return J


def mtspectrumc(data, Fs, fpass, tapers):
    """
    mtspectrumc from chronux toolbox
    Inputs:
        data: 1d array
        Fs: sampling frequency
        fpass: frequency range to evaluate
        tapers: [NW, K] or [tapers, eigenvalues]
    Outputs:
        S: power spectrum
    """
    N = len(data)
    nfft = np.max(
        [int(2 ** np.ceil(np.log2(N))), N]
    )  # number of points in fft of prolates
    f, findx = getfgrid(Fs, nfft, fpass)
    tapers = dpsschk(tapers, N, Fs)
    J = mtfftc(data, tapers, nfft, Fs)
    J = J.T[findx, :]
    S = np.real(np.mean(np.conj(J) * J, 1))
    return pd.Series(index=f, data=S)


def point_spectra(times, Fs=1250, freq_range=[1, 20], tapers0=[3, 5], pad=0):
    """
    Compute point spectra for a set of spike times
    Inputs:
        times: array of spike times (in seconds)
        Fs: sampling frequency
        freq_range: frequency range to evaluate
        tapers0: [NW, K] or [tapers, eigenvalues]
        pad: number of points to pad
    Outputs:
        spectra: power spectrum
        f: frequencies

    By Ryan H, converted from PointSpectra.m by Ralitsa Todorova
    """

    timesRange = [min(times), max(times)]
    t = np.mean(timesRange)
    window = np.floor(np.diff(timesRange))
    nSamplesPerWindow = int(np.round(Fs * window))  # number of samples in window
    nfft = np.max(
        [(int(2 ** np.ceil(np.log2(nSamplesPerWindow))) + pad), nSamplesPerWindow]
    )
    fAll = np.linspace(0, Fs, int(nfft))
    ok = (fAll >= freq_range[0]) & (fAll <= freq_range[1])
    Nf = sum(ok)
    tapers, eigens = dpss(nSamplesPerWindow, tapers0[0], tapers0[1])
    tapers = tapers * np.sqrt(Fs)
    spectra = np.zeros(Nf)
    H = np.fft.fft(tapers.T, int(nfft), 1)  # fft of tapers
    # restrict fft of tapers to required frequencies
    f = fAll[ok]
    H = H[:, ok]
    w = 2 * np.pi * f  # angular frequencies at which ft is to be evaluated
    timegrid = np.linspace(timesRange[0], timesRange[1], nSamplesPerWindow)

    # make sure times are within the range of timegrid
    data = times[(times >= timegrid[0]) & (times <= timegrid[-1])]
    data_proj = [np.interp(data, timegrid, taper) for taper in tapers.T]
    data_proj = np.vstack(data_proj)
    exponential = np.exp(np.outer(-1j * w, (data - timegrid[0])))
    J = exponential @ data_proj.T - H.T * len(data) / len(timegrid)
    spectra = np.squeeze(np.mean(np.real(np.conj(J) * J), axis=1))
    return spectra, f
