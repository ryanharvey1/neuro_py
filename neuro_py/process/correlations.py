__all__ = [
    "compute_AutoCorrs",
    "pairwise_corr",
    "pairwise_cross_corr",
    "pairwise_spatial_corr",
    "compute_cross_correlogram",
]
import numpy as np
import itertools
import pandas as pd
from scipy import stats
from neuro_py.process.peri_event import crossCorr
from scipy import signal

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


def pairwise_cross_corr(spks, binsize=0.001, nbins=100, return_index=False, pairs=None):
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

    crosscorrs = pd.DataFrame(index=times, columns=np.arange(len(pairs)))

    # Now we can iterate over spikes
    for i, s in enumerate(pairs):
        # Calling the crossCorr function
        crosscorrs[i] = crossCorr(spks[s[0]], spks[s[1]], binsize, nbins)

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
