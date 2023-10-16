__all__ = [
    "get_significant_events",
    "confidence_intervals",
    "reindex_df",
    "regress_out",
]
import numpy as np
import scipy.stats as stats
import pandas as pd
import warnings


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
    # compute interval for each column
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        interval = [
            stats.t.interval(
                conf,
                len(a) - 1,
                loc=np.nanmean(a),
                scale=stats.sem(a, nan_policy="omit"),
            )
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


def regress_out(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Regress b from a keeping a's original mean.


    a_prime = regress_out(a, b) performs regression of variable b from variable a
    while preserving the original mean of a. The function calculates the residual
    component of a that remains after removing the effect of b. The regression is
    performed using the ordinary least squares method.

    adapted from the seaborn function of the same name
        https://github.com/mwaskom/seaborn/blob/824c102525e6a29cde9bca1ce0096d50588fda6b/seaborn/regression.py#L337

    Parameters
    ----------
    a : array-like
        The variable to be regressed. Must be 1-dimensional.
    b : array-like
        The variable to regress on a. Must be 1-dimensional.

    Returns
    -------
    a_prime : array-like
        The residual of a after regressing out b. Has the same shape as a.
    """
    # remove nans and infs from a and b, and make a_result vector for output
    a_result = np.full_like(a, np.nan)

    valid_mask = np.isfinite(a) & np.isfinite(b)
    a = np.asarray(a)[valid_mask]
    b = np.asarray(b)[valid_mask]

    # remove mean from a and b
    a_mean = a.mean()
    a = a - a_mean
    b = b - b.mean()

    # calculate regression and subtract from a to get a_prime
    b = np.c_[b]
    a_prime = a - b @ np.linalg.pinv(b) @ a

    # add mean back to a_prime
    a_prime = np.asarray(a_prime + a_mean).reshape(a.shape)

    # put a_prime back into a_result vector to preserve nans
    a_result[valid_mask] = a_prime
    return a_result
