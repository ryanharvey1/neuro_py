import warnings
from typing import Any, Callable, Optional, Tuple, Union

import bottleneck as bn
import numpy as np
import pandas as pd
import scipy.stats as stats
from numpy.typing import NDArray


def get_significant_events(
    scores: Union[list[float], NDArray[Any]],
    shuffled_scores: NDArray[Any],
    q: float = 95,
    tail: str = "both",
) -> Tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
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
        Nonfinite draws are excluded separately for each event from p-value
        denominators, percentile thresholds, and standardized scores.
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

    Notes
    -----
    Nonfinite observations and events with no finite shuffled scores are not
    significant: their p-value is 1 and standardized score is NaN. Percentile
    selection and the finite-sample p-value correction are otherwise unchanged.
    """
    scores_array = np.atleast_1d(np.asarray(scores, dtype=float))
    if scores_array.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    shuffled_scores = np.asarray(shuffled_scores, dtype=float)
    if shuffled_scores.ndim == 1 and scores_array.size == 1:
        shuffled_scores = shuffled_scores[:, None]
    if shuffled_scores.ndim != 2:
        raise ValueError("shuffled_scores must have shape (n_shuffles, n_events)")
    if shuffled_scores.shape[1] != scores_array.size:
        if shuffled_scores.shape[0] != scores_array.size:
            raise ValueError("shuffled_scores must have one column per observed score")
        shuffled_scores = shuffled_scores.T
    if tail not in ("both", "right", "left"):
        raise ValueError("tail must be 'left', 'right', or 'both'")
    if not np.isfinite(q) or not 0 <= q <= 100:
        raise ValueError("q must be between 0 and 100")

    # Each event has its own number of usable null draws. Nonfinite draws
    # cannot contribute to the denominator without contributing comparisons.
    valid = np.isfinite(shuffled_scores)
    n = valid.sum(axis=0)
    values = np.where(valid, shuffled_scores, np.nan)
    observed = scores_array
    if tail == "both":
        np.abs(values, out=values)
        observed = np.abs(scores_array)
    if tail == "left":
        r = np.sum(valid & (values <= observed), axis=0)
    else:
        r = np.sum(valid & (values >= observed), axis=0)

    testable = np.isfinite(scores_array) & (n > 0)
    pvalues = np.ones(scores_array.size, dtype=float)
    pvalues[testable] = (r[testable] + 1) / (n[testable] + 1)
    threshold = np.full(scores_array.size, np.nan)
    stddev = np.full(scores_array.size, np.nan)
    if np.any(testable):
        usable = values[:, testable]
        threshold[testable] = np.nanpercentile(
            usable, q=100 - q if tail == "left" else q, axis=0
        )
        null_mean = np.nanmean(usable, axis=0)
        null_std = np.nanstd(usable, axis=0)
        delta = (
            null_mean - observed[testable]
            if tail == "left"
            else observed[testable] - null_mean
        )
        # Constant nulls retain the existing inf/NaN standardized-score behavior.
        with np.errstate(divide="ignore", invalid="ignore"):
            stddev[testable] = delta / null_std

    significant = observed < threshold if tail == "left" else observed > threshold
    sig_event_idx = np.flatnonzero(testable & significant)
    return sig_event_idx, pvalues, stddev


def confidence_intervals(
    X: NDArray[Any],
    conf: float = 0.95,
    estimator: Callable[..., NDArray[Any] | float] = np.nanmean,
    n_boot: int = 1000,
    random_state: Optional[int] = None,
) -> Tuple[NDArray[Any], NDArray[Any]]:
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


def regress_out(a: NDArray[Any], b: NDArray[Any]) -> NDArray[Any]:
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
