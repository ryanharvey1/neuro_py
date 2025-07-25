import warnings
from typing import Callable, Optional, Tuple, Union

import bottleneck as bn
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

    if tail == "both":
        threshold = np.percentile(np.abs(shuffled_scores), axis=0, q=q)
        sig_event_idx = np.where(np.abs(scores) > threshold)[0]
    elif tail == "right":
        threshold = np.percentile(shuffled_scores, axis=0, q=q)
        sig_event_idx = np.where(scores > threshold)[0]
    elif tail == "left":
        threshold = np.percentile(shuffled_scores, axis=0, q=100 - q)
        sig_event_idx = np.where(scores < threshold)[0]

    # calculate how many standard deviations away from shuffle
    if tail == "both":
        stddev = (
            np.abs(scores) - np.nanmean(np.abs(shuffled_scores), axis=0)
        ) / np.nanstd(np.abs(shuffled_scores), axis=0)
    elif tail == "right":
        stddev = (scores - np.nanmean(shuffled_scores, axis=0)) / np.nanstd(
            shuffled_scores, axis=0
        )
    elif tail == "left":
        stddev = (np.nanmean(shuffled_scores, axis=0) - scores) / np.nanstd(
            shuffled_scores, axis=0
        )

    return np.atleast_1d(sig_event_idx), np.atleast_1d(pvalues), np.atleast_1d(stddev)


def confidence_intervals(
    X: np.ndarray,
    conf: float = 0.95,
    estimator: Callable = np.nanmean,
    n_boot: int = 1000,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate upper and lower confidence intervals on a matrix using a specified estimator.

    Parameters
    ----------
    X : np.ndarray
        A numpy ndarray of shape (n_signals, n_samples).
    conf : float, optional
        Confidence level value (default is 0.95).
    estimator : Callable, optional
        Function to use for central tendency (default: np.nanmean). You may use numpy (np.nanmean, np.nanmedian, etc.) or Bottleneck (bn.nanmean, bn.nanmedian, etc.) for faster computation.
    n_boot : int, optional
        Number of bootstrap samples for CI if estimator is not mean/median (default: 1000).
    random_state : int, optional
        Random seed for bootstrapping.

    Returns
    -------
    lower : np.ndarray
        Lower bounds of the confidence intervals (shape: (n_signals,)).
    upper : np.ndarray
        Upper bounds of the confidence intervals (shape: (n_signals,)).
    """
    if estimator in (np.nanmean, bn.nanmean, np.nanmedian, bn.nanmedian):
        # compute interval for each column using t-interval
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            interval = [
                stats.t.interval(
                    conf,
                    len(a) - 1,
                    loc=estimator(a),
                    scale=stats.sem(a, nan_policy="omit"),
                )
                for a in X.T
            ]
        interval = np.vstack(interval)
        lower = interval[:, 0]
        upper = interval[:, 1]
    else:
        # Bootstrap CI for arbitrary estimator
        rng = np.random.default_rng(random_state)
        n_signals = X.shape[1]
        boot_stats = np.empty((n_boot, n_signals))
        for i in range(n_boot):
            sample_idx = rng.integers(0, X.shape[0], size=X.shape[0])
            boot_stats[i] = estimator(X[sample_idx, :], axis=0)
        lower = np.percentile(boot_stats, 100 * (1 - conf) / 2, axis=0)
        upper = np.percentile(boot_stats, 100 * (1 + conf) / 2, axis=0)
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
    valid_mask = np.isfinite(a) & np.isfinite(b)
    a_valid = np.asarray(a)[valid_mask]
    b_valid = np.asarray(b)[valid_mask]

    # remove mean from a and b
    a_mean = a_valid.mean() if a_valid.size > 0 else 0.0
    a_centered = a_valid - a_mean
    b_centered = b_valid - b_valid.mean() if b_valid.size > 0 else b_valid

    # calculate regression and subtract from a to get a_prime
    if b_centered.size > 0:
        b_mat = np.c_[b_centered]
        a_prime = a_centered - b_mat @ np.linalg.pinv(b_mat) @ a_centered
        a_prime = np.asarray(a_prime + a_mean).reshape(a_centered.shape)
    else:
        a_prime = a_centered

    # Build output: fill with np.nan where invalid, otherwise with result
    a_result = np.empty_like(a, dtype=float)
    a_result[:] = np.nan
    a_result[valid_mask] = a_prime
    return a_result
