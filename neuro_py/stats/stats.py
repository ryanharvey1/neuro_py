__all__ = ["get_significant_events", "confidence_intervals","reindex_df"]
import numpy as np
import scipy.stats as stats
import pandas as pd

def get_significant_events(scores, shuffled_scores, q=95, tail="both"):
    """
    Return the significant events based on percentiles,
    the p-values and the standard deviation of the scores
    in terms of the shuffled scores.
    Parameters
    ----------
    scores : array of shape (n_events,)
        The array of scores for which to calculate significant events
    shuffled_scores : array of shape (n_shuffles, n_events)
        The array of scores obtained from randomized data
    q : float in range of [0,100]
        Percentile to compute, which must be between 0 and 100 inclusive.
    Returns
    -------
    sig_event_idx : array of shape (n_sig_events,)
        Indices (from 0 to n_events-1) of significant events.
    pvalues : array of shape (n_events,)
        The p-values
    stddev : array of shape (n_events,)
        The standard deviation of the scores in terms of the shuffled scores
    """
    # check shape and correct if needed
    if isinstance(scores, list) | isinstance(scores, np.ndarray):
        if shuffled_scores.shape[1] != len(scores):
            shuffled_scores = shuffled_scores.T

    n = shuffled_scores.shape[0]
    if tail == "both":
        r = np.sum(np.abs(shuffled_scores) >= np.abs(scores), axis=0)
    elif tail == "right":
        r = np.sum(shuffled_scores >= scores, axis=0)
    elif tail == "left":
        r = np.sum(shuffled_scores <= scores, axis=0)
    else:
        raise ValueError("tail must be 'left', 'right', or 'both'")
    pvalues = (r + 1) / (n + 1)

    # set nan scores to 1
    if isinstance(np.isnan(scores), np.ndarray):
        pvalues[np.isnan(scores)] = 1

    sig_event_idx = np.argwhere(
        scores > np.percentile(shuffled_scores, axis=0, q=q)
    ).squeeze()

    # calculate how many standard deviations away from shuffle
    stddev = (np.abs(scores) - np.nanmean(np.abs(shuffled_scores), axis=0)) / np.nanstd(
        np.abs(shuffled_scores), axis=0
    )

    return np.atleast_1d(sig_event_idx), np.atleast_1d(pvalues), np.atleast_1d(stddev)


def confidence_intervals(X: np.ndarray, conf: float = 0.95):
    """
    confidence_intervals: calculates upper and lower .95 confidence intervals on matrix

    Input:
        X - numpy ndarray, (n signals, n samples)
        conf - float, confidence level value (default: .95)
    """
    # remove nans
    X = X[np.sum(np.isnan(X), axis=1) == 0, :]
    # compute interval for each column
    interval = [
        stats.t.interval(conf, len(a) - 1, loc=np.mean(a), scale=stats.sem(a))
        for a in X.T
    ]
    interval = np.vstack(interval)
    lower = interval[:, 0]
    upper = interval[:, 1]
    return lower, upper


def reindex_df(df: pd.core.frame.DataFrame, weight_col: str) -> pd.core.frame.DataFrame:
    """
    reindex_df: expands dataframe by weights

    expand the dataframe to prepare for resampling result is 1 row per count per sample

    Helpful when making weighted proportion plots

    Input:
            df - pandas dataframe
            weight_col - column name that contains weights (should be int)
    Output:
            df - new pandas dataframe with resampling
    """

    df = df.reindex(df.index.repeat(df[weight_col])).copy()

    df.reset_index(drop=True, inplace=True)

    return df