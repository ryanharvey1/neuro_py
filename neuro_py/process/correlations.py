import itertools
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.stats import poisson

from neuro_py.process.peri_event import crossCorr, deconvolve_peth


def compute_AutoCorrs(
    spks: np.ndarray, binsize: float = 0.001, nbins: int = 100
) -> pd.DataFrame:
    """
    Compute autocorrelations for spike trains.

    Parameters
    ----------
    spks : np.ndarray
        Nested ndarrays where each array contains the spike times for one neuron.
    binsize : float, optional
        The size of each bin in seconds, by default 0.001 (1 ms).
    nbins : int, optional
        The number of bins for the autocorrelation, by default 100.

    Returns
    -------
    pd.DataFrame
        A DataFrame where each column represents the autocorrelation of the corresponding neuron.
        The index is the time lag, and the values are the autocorrelations.
    """
    # First let's prepare a pandas dataframe to receive the data
    times = np.arange(0, binsize * (nbins + 1), binsize) - (nbins * binsize) / 2
    autocorrs = pd.DataFrame(index=times, columns=np.arange(len(spks)))

    # Now we can iterate over the dictionnary of spikes
    for i, s in enumerate(spks):
        if len(s) == 0:
            continue
        # Calling the crossCorr function
        autocorrs[i] = crossCorr(s, s, binsize, nbins)

    # And don't forget to replace the 0 ms for 0
    autocorrs.loc[0] = 0.0
    return autocorrs


def pairwise_corr(
    X: np.ndarray, method: str = "pearson", pairs: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute pairwise correlations between all rows of a matrix.

    Parameters
    ----------
    X : np.ndarray
        2D numpy array of shape (n, p), where n is the number of rows (variables) and p is the number of columns (features).
    method : str, optional
        Correlation method to use ('pearson', 'spearman', or 'kendall'), by default "pearson".
    pairs : np.ndarray, optional
        Array of shape (m, 2) specifying the pairs of rows to compute correlations between.
        If None, computes correlations for all unique row pairs.

    Returns
    -------
    rho : np.ndarray
        Array of correlation coefficients.
    pval : np.ndarray
        Array of p-values for the correlation tests.
    pairs : np.ndarray
        Array of pairs (indices) for which correlations were computed.

    Raises
    ------
    ValueError
        If the method is not 'pearson', 'spearman', or 'kendall'.

    Example
    -------
    >>> X = np.random.rand(10, 5)
    >>> rho, pval, pairs = pairwise_corr(X, method="spearman")
    """
    if pairs is None:
        x = np.arange(0, X.shape[0])
        pairs = np.array(list(itertools.combinations(x, 2)))

    rho = []
    pval = []
    for i, s in enumerate(pairs):
        if method == "pearson":
            rho_, pval_ = stats.pearsonr(X[s[0], :], X[s[1], :])
        elif method == "spearman":
            rho_, pval_ = stats.spearmanr(X[s[0], :], X[s[1], :])
        elif method == "kendall":
            rho_, pval_ = stats.kendalltau(X[s[0], :], X[s[1], :])
        else:
            raise ValueError("method must be pearson, spearman or kendall")
        rho.append(rho_)
        pval.append(pval_)
    return rho, pval, pairs


def pairwise_cross_corr(
    spks: np.ndarray,
    binsize: float = 0.001,
    nbins: int = 100,
    return_index: bool = False,
    pairs: Optional[np.ndarray] = None,
    deconvolve: bool = False,
) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
    """
    Compute pairwise time-lagged cross-correlations between spike trains of different cells.

    Parameters
    ----------
    spks : np.ndarray
        Nested numpy arrays, where each array contains the spike times for a cell.
    binsize : float, optional
        The size of time bins in seconds. Default is 0.001 (1 ms).
    nbins : int, optional
        Number of bins to use for the correlation window. Default is 100.
    return_index : bool, optional
        Whether to return the index (pairs) of cells used for the correlation. Default is False.
    pairs : np.ndarray, optional
        Precomputed list of pairs of cells (indices) to compute the cross-correlation for.
        If None, all unique pairs will be computed. Default is None.
    deconvolve : bool, optional
        Whether to apply deconvolution when computing the cross-correlation. Default is False.

    Returns
    -------
    crosscorrs : pd.DataFrame
        A pandas DataFrame of shape (t, n_pairs), where t is the time axis and n_pairs are the pairs of cells.
    pairs : np.ndarray, optional
        The pairs of cells for which cross-correlations were computed. Returned only if `return_index` is True.

    Example
    -------
    >>> spks = np.array([np.random.rand(100), np.random.rand(100)])
    >>> crosscorrs, pairs = pairwise_cross_corr(spks, binsize=0.01, nbins=50, return_index=True)
    """
    # Get unique combo without repeats
    if pairs is None:
        x = np.arange(0, spks.shape[0])
        pairs = np.array(list(itertools.combinations(x, 2)))

    # prepare a pandas dataframe to receive the data
    times = np.linspace(-(nbins * binsize) / 2, (nbins * binsize) / 2, nbins + 1)

    def compute_crosscorr(pair):
        i, j = pair
        crosscorr = crossCorr(spks[i], spks[j], binsize, nbins)
        return crosscorr

    def compute_crosscorr_deconvolve(pair):
        i, j = pair
        crosscorr, _ = deconvolve_peth(spks[i], spks[j], binsize, nbins)
        return crosscorr

    if deconvolve:
        crosscorrs = [compute_crosscorr_deconvolve(pair) for pair in pairs]
    else:
        crosscorrs = [compute_crosscorr(pair) for pair in pairs]

    crosscorrs = pd.DataFrame(
        index=times,
        data=np.array(crosscorrs).T,
        columns=np.arange(len(pairs)),
    )

    if return_index:
        return crosscorrs, pairs
    else:
        return crosscorrs


def pairwise_spatial_corr(
    X: np.ndarray, return_index: bool = False, pairs: np.ndarray = None
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute pairwise spatial correlations between cells' spatial maps.

    Parameters
    ----------
    X : np.ndarray
        A 3D numpy array of shape (n_cells, n_space, n_space) representing the spatial maps of cells.
    return_index : bool, optional
        If True, returns the indices of the cell pairs used for the correlation.
    pairs : np.ndarray, optional
        Array of cell pairs for which to compute the correlation. If not provided, all unique pairs are used.

    Returns
    -------
    spatial_corr : np.ndarray
        Array containing the Pearson correlation coefficients for each pair of cells.
    pairs : np.ndarray, optional
        Array of cell pairs used for the correlation (if return_index is True).
    """
    # Get unique combo without repeats
    if pairs is None:
        x = np.arange(0, X.shape[0])
        pairs = np.array(list(itertools.combinations(x, 2)))

    spatial_corr = []
    # Now we can iterate over spikes
    for i, s in enumerate(pairs):
        # Calling the crossCorr function
        x1 = X[s[0], :, :].flatten()
        x2 = X[s[1], :, :].flatten()
        bad_idx = np.isnan(x1) | np.isnan(x2)
        spatial_corr.append(np.corrcoef(x1[~bad_idx], x2[~bad_idx])[0, 1])

    if return_index:
        return np.array(spatial_corr), pairs
    else:
        return np.array(spatial_corr)


def compute_cross_correlogram(
    X: np.ndarray, dt: float = 1.0, window: float = 0.5
) -> pd.DataFrame:
    """
    Compute pairwise cross-correlograms between signals in an array.

    Parameters
    ----------
    X : np.ndarray
        N-dimensional array of shape (n_signals, n_timepoints) representing the signals.
    dt : float, optional
        Time step between samples in seconds, default is 1.0.
    window : float, optional
        Window size in seconds for the cross-correlogram. The output will include values
        within +/- window from the center. If None, returns the full correlogram.

    Returns
    -------
    pd.DataFrame
        Pairwise cross-correlogram with time lags as the index and signal pairs as columns.
    """

    crosscorrs = {}
    pairs = list(itertools.combinations(np.arange(X.shape[0]), 2))
    for i, j in pairs:
        auc = signal.correlate(X[i], X[j])
        times = signal.correlation_lags(len(X[i]), len(X[j])) * dt
        # normalize by coeff
        normalizer = np.sqrt((X[i] ** 2).sum(axis=0) * (X[j] ** 2).sum(axis=0))
        auc /= normalizer

        crosscorrs[(i, j)] = pd.Series(index=times, data=auc, dtype="float32")
    crosscorrs = pd.DataFrame.from_dict(crosscorrs)

    if window is None:
        return crosscorrs
    else:
        return crosscorrs[(crosscorrs.index >= -window) & (crosscorrs.index <= window)]


def local_firfilt(x: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Apply a FIR filter to the input signal x using the provided filter coefficients W.

    Parameters
    ----------
    x : np.ndarray
        The input signal to be filtered.
    W : np.ndarray
        The FIR filter coefficients.

    Returns
    -------
    np.ndarray
        The filtered signal.
    """
    C = int(len(W))
    D = int(np.ceil(C / 2) - 1)
    xx = [np.flipud(x[:C]), x, np.flipud(x[-C:])]
    xx = list(itertools.chain(*xx))
    Y = signal.lfilter(W, 1, xx)
    Y = Y[C + D : len(Y) - C + D]
    return Y


def local_gausskernel(sigma: float, N: int) -> np.ndarray:
    """
    Generate a Gaussian kernel with the given standard deviation and size.

    Parameters
    ----------
    sigma : float
        The standard deviation of the Gaussian.
    N : int
        The size of the kernel (number of points). Must be an odd number.

    Returns
    -------
    np.ndarray
        A 1D Gaussian kernel.
    """
    x = np.arange(-(N - 1) / 2, ((N - 1) / 2) + 1)
    k = 1 / (2 * np.pi * sigma) * np.exp(-(x**2 / 2 / sigma**2))
    return k


def cch_conv(
    cch: np.ndarray, W: int = 30, HF: float = 0.6
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convolve the cross-correlogram with a Gaussian window and calculate p-values.

    Parameters
    ----------
    cch : np.ndarray
        The cross-correlogram data (1D array).
    W : int, optional
        The width of the Gaussian window (default is 30).
    HF : float, optional
        The height factor to modify the Gaussian peak (default is 0.6).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        - pvals : array of p-values.
        - pred : predicted values after convolution.
        - qvals : q-values (1 - pvals).
    """
    
    SDG = W / 2
    if round(SDG) == SDG:  # even W
        win = local_gausskernel(SDG, 6 * SDG + 1)
        cidx = int(SDG * 3 + 1)
    else:
        win = local_gausskernel(SDG, 6 * SDG + 2)
        cidx = int(SDG * 3 + 1.5)
    win[cidx - 1] = win[cidx - 1] * (1 - HF)
    win = win / sum(win)
    pred = local_firfilt(cch, win)
    pvals = 1 - poisson.cdf(cch - 1, pred) - poisson.pmf(cch, pred) * 0.5
    qvals = 1 - pvals
    return pvals, pred, qvals


def sig_mod(
    cch: np.ndarray,
    binsize: float = 0.005,
    sig_window: float = 0.2,
    alpha: float = 0.001,
    W: int = 30
) -> Tuple[bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Assess the significance of cross-correlogram values using Poisson statistics.

    Parameters
    ----------
    cch : np.ndarray
        The cross-correlogram data (1D array).
    binsize : float, optional
        The size of each bin in seconds (default is 0.005).
    sig_window : float, optional
        The window size to consider for significance (default is 0.2).
    alpha : float, optional
        The significance level (default is 0.001).
    W : int, optional
        The width of the Gaussian window for convolution (default is 30).

    Returns
    -------
    tuple[bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        - sig : boolean indicating whether the cross-correlogram is significant.
        - hiBound : upper bound for significance.
        - loBound : lower bound for significance.
        - pvals : array of p-values.
        - pred : predicted values after convolution.
    """
    # check and correct for negative values
    if np.any(cch < 0):
        cch = cch + np.abs(min(cch))

    pvals, pred, qvals = cch_conv(cch, W)

    nBonf = int(sig_window / binsize) * 2
    hiBound = poisson.ppf(1 - alpha / nBonf, pred)
    loBound = poisson.ppf(alpha / nBonf, pred)

    center_bins = np.arange(
        int(len(cch) / 2 - 0.1 / binsize), int(len(cch) / 2 + 0.1 / binsize)
    )
    # at least 2 bins more extreme than bound to be sig
    sig = (sum(cch[center_bins] > max(hiBound)) > 2) | (
        sum(cch[center_bins] < min(loBound)) > 2
    )
    return sig, hiBound, loBound, pvals, pred
