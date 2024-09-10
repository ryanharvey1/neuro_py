import itertools

import numpy as np
import pandas as pd
from lazy_loader import attach as _attach
from scipy import signal, stats
from scipy.stats import poisson

from neuro_py.process.peri_event import crossCorr, deconvolve_peth

__all__ = (
    "compute_AutoCorrs",
    "pairwise_corr",
    "pairwise_cross_corr",
    "pairwise_spatial_corr",
    "compute_cross_correlogram",

)
__getattr__, __dir__, __all__ = _attach(f"{__name__}", submodules=__all__)
del _attach


def compute_AutoCorrs(spks, binsize=0.001, nbins=100):
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


def pairwise_corr(X, method="pearson", pairs=None):
    """
    Compute pairwise correlations between all rows of matrix
    Input:
        X: numpy array of shape (n,p)
    Output:
        corr: numpy array rho
        pval: numpy array pval
        c: numpy array ref and target from which the correlation was computed
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


def pairwise_cross_corr(spks, binsize=0.001, nbins=100, return_index=False, pairs=None, deconvolve=False):
    """
    Compute pairwise time-lagged correlations between cells
    Input:
        spks: list of numpy arrays of shape (n,)
        binsize: float, size of bins in seconds
        nbins: int, number of bins
        return_index: bool, return the index of the cells used for the correlation
        pairs: list of pairs of cells to compute the correlation
    Output:
        crosscorrs: pandas dataframe of shape (t,n pairs)

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


def pairwise_spatial_corr(X, return_index=False, pairs=None):
    """
    Compute pairwise spatial correlations between cells
    Input:
        X: numpy array of shape (n_cells, n_space, n_space)
        return_index: bool, return the index of the cells used for the correlation
        pairs: list of pairs of cells to compute the correlation
    Output:
        spatial_corr: the pearson correlation between the cells in pairs
        pairs: list of pairs of cells used for the correlation

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


def compute_cross_correlogram(X, dt=1, window=0.5):
    """
    Cross-correlate two N-dimensional arrays (pairwise).
    Input:
        X: N-dimensional array of shape  (n_signals, n_timepoints)
        dt: time step in seconds, default 1 is nlags
        window: window size in seconds, output will be +- window
    Output:
        cross_correlogram: pandas dataframe with pairwise cross-correlogram
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


def local_firfilt(x, W):
    C = int(len(W))
    D = int(np.ceil(C / 2) - 1)
    xx = [np.flipud(x[:C]), x, np.flipud(x[-C:])]
    xx = list(itertools.chain(*xx))
    Y = signal.lfilter(W, 1, xx)
    Y = Y[C + D : len(Y) - C + D]
    return Y


def local_gausskernel(sigma, N):
    x = np.arange(-(N - 1) / 2, ((N - 1) / 2) + 1)
    k = 1 / (2 * np.pi * sigma) * np.exp(-(x**2 / 2 / sigma**2))
    return k


def cch_conv(cch, W=30, HF=0.6):
    # Stark and Abeles JNM 2009
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


def sig_mod(cch, binsize=0.005, sig_window=0.2, alpha=0.001, W=30):
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