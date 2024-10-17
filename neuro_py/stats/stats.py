import warnings
from typing import Tuple, Union

import numpy as np
import pandas as pd
import scipy.stats as stats


def get_significant_events(
    scores: Union[list, np.ndarray],
    shuffled_scores: np.ndarray,
    q: float = 95,
    tail: str = "both",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return the significant events based on percentiles,
    the p-values, and the standard deviation of the scores
    in terms of the shuffled scores.

    Parameters
    ----------
    scores : Union[list, np.ndarray]
        The array of scores for which to calculate significant events.
    shuffled_scores : np.ndarray
        The array of scores obtained from randomized data (shape: (n_shuffles, n_events)).
    q : float, optional
        Percentile to compute, which must be between 0 and 100 inclusive (default is 95).
    tail : str, optional
        Tail for the test, which can be 'left', 'right', or 'both' (default is 'both').

    Returns
    -------
    sig_event_idx : np.ndarray
        Indices (from 0 to n_events-1) of significant events.
    pvalues : np.ndarray
        The p-values.
    stddev : np.ndarray
        The standard deviation of the scores in terms of the shuffled scores.
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


def confidence_intervals(
    X: np.ndarray, conf: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate upper and lower confidence intervals on a matrix.

    Parameters
    ----------
    X : np.ndarray
        A numpy ndarray of shape (n_signals, n_samples).
    conf : float, optional
        Confidence level value (default is 0.95).

    Returns
    -------
    lower : np.ndarray
        Lower bounds of the confidence intervals (shape: (n_signals,)).
    upper : np.ndarray
        Upper bounds of the confidence intervals (shape: (n_signals,)).
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
    # stack intervals into array
    interval = np.vstack(interval)
    # split into lower and upper
    lower = interval[:, 0]
    upper = interval[:, 1]

    return lower, upper


def reindex_df(df: pd.DataFrame, weight_col: str) -> pd.DataFrame:
    """
    Expand the dataframe by weights.

    This function expands the dataframe to prepare for resampling,
    resulting in 1 row per count per sample, which is helpful
    when making weighted proportion plots.

    Parameters
    ----------
    df : pd.DataFrame
        The pandas dataframe to be expanded.
    weight_col : str
        The column name that contains weights (should be int).

    Returns
    -------
    pd.DataFrame
        A new pandas dataframe with resampling based on the weights.
    """

    df = df.reindex(df.index.repeat(df[weight_col])).copy()

    df.reset_index(drop=True, inplace=True)

    return df


def regress_out(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Regress b from a while keeping a's original mean.

    This function performs regression of variable b from variable a while
    preserving the original mean of a. It calculates the residual component
    of a that remains after removing the effect of b using ordinary least squares.

    Parameters
    ----------
    a : np.ndarray
        The variable to be regressed. Must be 1-dimensional.
    b : np.ndarray
        The variable to regress on a. Must be 1-dimensional.

    Returns
    -------
    np.ndarray
        The residual of a after regressing out b. Has the same shape as a.

    Notes
    -----
    Adapted from the seaborn function of the same name:
    https://github.com/mwaskom/seaborn/blob/824c102525e6a29cde9bca1ce0096d50588fda6b/seaborn/regression.py#L337
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
