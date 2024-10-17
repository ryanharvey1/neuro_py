import multiprocessing
from typing import List, Optional, Tuple, Union

import numpy as np
from joblib import Parallel, delayed
from nelpy.analysis import replay
from nelpy.decoding import decode1D as decode
from nelpy.core import BinnedSpikeTrainArray
from nelpy import TuningCurve1D


def WeightedCorr(
    weights: np.ndarray, x: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None
) -> float:
    """
    Calculate the weighted correlation between the X and Y dimensions of the matrix.

    Parameters
    ----------
    weights : np.ndarray
        A matrix of weights.
    x : Optional[np.ndarray], optional
        X-values for each column and row, by default None.
    y : Optional[np.ndarray], optional
        Y-values for each column and row, by default None.

    Returns
    -------
    float
        The weighted correlation coefficient.
    """
    weights[np.isnan(weights)] = 0.0

    if x is not None and x.size > 0:
        if np.ndim(x) == 1:
            x = np.tile(x, (weights.shape[0], 1))
    else:
        x, _ = np.meshgrid(
            np.arange(1, weights.shape[1] + 1), np.arange(1, weights.shape[0] + 1)
        )

    if y is not None and y.size > 0:
        if np.ndim(y) == 1:
            y = np.tile(y, (weights.shape[0], 1))
    else:
        _, y = np.meshgrid(
            np.arange(1, weights.shape[1] + 1), np.arange(1, weights.shape[0] + 1)
        )

    x = x.flatten()
    y = y.flatten()
    w = weights.flatten()

    mX = np.nansum(w * x) / np.nansum(w)
    mY = np.nansum(w * y) / np.nansum(w)

    covXY = np.nansum(w * (x - mX) * (y - mY)) / np.nansum(w)
    covXX = np.nansum(w * (x - mX) ** 2) / np.nansum(w)
    covYY = np.nansum(w * (y - mY) ** 2) / np.nansum(w)

    c = covXY / np.sqrt(covXX * covYY)

    return c


def WeightedCorrCirc(
    weights: np.ndarray,
    x: Optional[np.ndarray] = None,
    alpha: Optional[np.ndarray] = None,
) -> float:
    """
    Compute the correlation between x and y dimensions of a matrix with angular (circular) values.

    Parameters
    ----------
    weights : np.ndarray
        A 2D numpy array of weights.
    x : Optional[np.ndarray], optional
        A 2D numpy array of x-values, by default None.
    alpha : Optional[np.ndarray], optional
        A 2D numpy array of angular (circular) y-values, by default None.

    Returns
    -------
    float
        The correlation between x and y dimensions.
    """
    weights[np.isnan(weights)] = 0.0

    if x is not None and x.size > 0:
        if np.ndim(x) == 1:
            x = np.tile(x, (weights.shape[0], 1))
    else:
        x, _ = np.meshgrid(
            np.arange(1, weights.shape[1] + 1), np.arange(1, weights.shape[0] + 1)
        )
    if alpha is None:
        alpha = np.tile(
            np.linspace(0, 2 * np.pi, weights.shape[0], endpoint=False),
            (weights.shape[1], 1),
        ).T

    rxs = WeightedCorr(weights, x, np.sin(alpha))
    rxc = WeightedCorr(weights, x, np.cos(alpha))
    rcs = WeightedCorr(weights, np.sin(alpha), np.cos(alpha))

    # Compute angular-linear correlation
    rho = np.sqrt((rxc**2 + rxs**2 - 2 * rxc * rxs * rcs) / (1 - rcs**2))
    return rho


def weighted_correlation(
    posterior: np.ndarray,
    time: Optional[np.ndarray] = None,
    place_bin_centers: Optional[np.ndarray] = None,
) -> float:
    """
    Calculate the weighted correlation between time and place bin centers using a posterior probability matrix.

    Parameters
    ----------
    posterior : np.ndarray
        A 2D numpy array representing the posterior probability matrix.
    time : Optional[np.ndarray], optional
        A 1D numpy array representing the time bins, by default None.
    place_bin_centers : Optional[np.ndarray], optional
        A 1D numpy array representing the place bin centers, by default None.

    Returns
    -------
    float
        The weighted correlation coefficient.
    """

    def _m(x, w) -> float:
        """Weighted Mean"""
        return np.sum(x * w) / np.sum(w)

    def _cov(x, y, w) -> float:
        """Weighted Covariance"""
        return np.sum(w * (x - _m(x, w)) * (y - _m(y, w))) / np.sum(w)

    def _corr(x, y, w) -> float:
        """Weighted Correlation"""
        return _cov(x, y, w) / np.sqrt(_cov(x, x, w) * _cov(y, y, w))

    if time is None:
        time = np.arange(posterior.shape[1])
    if place_bin_centers is None:
        place_bin_centers = np.arange(posterior.shape[0])

    place_bin_centers = place_bin_centers.squeeze()
    posterior[np.isnan(posterior)] = 0.0

    return _corr(time[:, np.newaxis], place_bin_centers[np.newaxis, :], posterior.T)


def shuffle_and_score(
    posterior_array: np.ndarray,
    w: np.ndarray,
    normalize: bool,
    tc: float,
    ds: float,
    dp: float,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Shuffle the posterior array and compute scores and weighted correlations.

    Parameters
    ----------
    posterior_array : np.ndarray
        The posterior probability array.
    w : np.ndarray
        Weights array.
    normalize : bool
        Whether to normalize the scores.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, float, float]
        Scores and weighted correlations for time-swapped and column-cycled arrays.
    """

    posterior_ts = replay.time_swap_array(posterior_array)
    posterior_cs = replay.column_cycle_array(posterior_array)

    scores_time_swap = replay.trajectory_score_array(
        posterior=posterior_ts, w=w, normalize=normalize
    )
    scores_col_cycle = replay.trajectory_score_array(
        posterior=posterior_cs, w=w, normalize=normalize
    )

    weighted_corr_time_swap = weighted_correlation(posterior_ts)
    weighted_corr_col_cycle = weighted_correlation(posterior_cs)

    return (
        scores_time_swap,
        scores_col_cycle,
        weighted_corr_time_swap,
        weighted_corr_col_cycle,
    )


def _shuffle_and_score(
    posterior_array: np.ndarray,
    tuningcurve: np.ndarray,
    w: np.ndarray,
    normalize: bool,
    ds: float,
    dp: float,
    n_shuffles: int,
) -> Tuple[
    np.ndarray, float, List[np.ndarray], List[np.ndarray], List[float], List[float]
]:
    """
    Shuffle the posterior array and compute scores and weighted correlations.

    Parameters
    ----------
    posterior_array : np.ndarray
        The posterior probability array.
    tuningcurve : np.ndarray
        The tuning curve array.
    w : np.ndarray
        Weights array.
    normalize : bool
        Whether to normalize the scores.
    ds : float
        Delta space.
    dp : float
        Delta probability.
    n_shuffles : int
        Number of shuffles.

    Returns
    -------
    Tuple[np.ndarray, float, List[np.ndarray], List[np.ndarray], List[float], List[float]]
        Scores and weighted correlations for original, time-swapped, and column-cycled arrays.
    """
    weighted_corr = weighted_correlation(posterior_array)
    scores = replay.trajectory_score_array(
        posterior=posterior_array, w=w, normalize=normalize
    )

    (
        scores_time_swap,
        scores_col_cycle,
        weighted_corr_time_swap,
        weighted_corr_col_cycle,
    ) = zip(
        *[
            shuffle_and_score(posterior_array, w, normalize, tuningcurve, ds, dp)
            for _ in range(n_shuffles)
        ]
    )
    return (
        scores,
        weighted_corr,
        scores_time_swap,
        scores_col_cycle,
        weighted_corr_time_swap,
        weighted_corr_col_cycle,
    )


def trajectory_score_bst(
    bst: BinnedSpikeTrainArray,
    tuningcurve: TuningCurve1D,
    w: Optional[int] = None,
    n_shuffles: int = 1000,
    weights: Optional[np.ndarray] = None,
    normalize: bool = False,
    parallel: bool = True,
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
]:
    """
    Calculate trajectory scores and weighted correlations for Bayesian spike train decoding.

    Parameters
    ----------
    bst : BinnedSpikeTrainArray
        Binned spike train object.
    tuningcurve : TuningCurve1D
        Tuning curve object.
    w : Optional[int], optional
        Window size, by default None.
    n_shuffles : int, optional
        Number of shuffles, by default 1000.
    weights : Optional[np.ndarray], optional
        Weights array, by default None.
    normalize : bool, optional
        Whether to normalize the scores, by default False.
    parallel : bool, optional
        Whether to run in parallel, by default True.

    Returns
    -------
    Union[
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray],
    ]
        Scores and weighted correlations for original, time-swapped, and column-cycled arrays.
    """

    if w is None:
        w = 0
    if not float(w).is_integer:
        raise ValueError("w has to be an integer!")

    if float(n_shuffles).is_integer:
        n_shuffles = int(n_shuffles)
    else:
        raise ValueError("n_shuffles must be an integer!")

    posterior, bdries, _, _ = decode(bst=bst, ratemap=tuningcurve)

    # scores = np.zeros(bst.n_epochs)
    # weighted_corr = np.zeros(bst.n_epochs)

    # if n_shuffles > 0:
    #     scores_time_swap = np.zeros((n_shuffles, bst.n_epochs))
    #     scores_col_cycle = np.zeros((n_shuffles, bst.n_epochs))
    #     weighted_corr_time_swap = np.zeros((n_shuffles, bst.n_epochs))
    #     weighted_corr_col_cycle = np.zeros((n_shuffles, bst.n_epochs))

    if parallel:
        num_cores = multiprocessing.cpu_count()

    ds, dp = bst.ds, np.diff(tuningcurve.bins)[0]

    (
        scores,
        weighted_corr,
        scores_time_swap,
        scores_col_cycle,
        weighted_corr_time_swap,
        weighted_corr_col_cycle,
    ) = zip(
        *Parallel(n_jobs=num_cores)(
            delayed(_shuffle_and_score)(
                posterior[:, bdries[idx] : bdries[idx + 1]],
                tuningcurve,
                w,
                normalize,
                ds,
                dp,
                n_shuffles,
            )
            for idx in range(bst.n_epochs)
        )
    )
    # for idx in range(bst.n_epochs):
    #     posterior_array = posterior[:, bdries[idx] : bdries[idx + 1]]
    #     scores[idx] = replay.trajectory_score_array(
    #         posterior=posterior_array, w=w, normalize=normalize
    #     )
    #     weighted_corr[idx] = weighted_correlation(posterior_array)

    #     if parallel:
    #         (
    #             scores_time_swap[:, idx],
    #             scores_col_cycle[:, idx],
    #             weighted_corr_time_swap[:, idx],
    #             weighted_corr_col_cycle[:, idx],
    #         ) = zip(
    #             *Parallel(n_jobs=num_cores)(
    #                 delayed(shuffle_and_score)(
    #                     posterior_array, w, normalize, tuningcurve, ds, dp
    #                 )
    #                 for _ in range(n_shuffles)
    #             )
    #         )
    #     else:
    #         (
    #             scores_time_swap[:, idx],
    #             scores_col_cycle[:, idx],
    #             weighted_corr_time_swap[:, idx],
    #             weighted_corr_col_cycle[:, idx],
    #         ) = zip(
    #             *[
    #                 shuffle_and_score(
    #                     posterior_array, w, normalize, tuningcurve, ds, dp
    #                 )
    #                 for _ in range(n_shuffles)
    #             ]
    #         )

    if n_shuffles > 0:
        return (
            np.array(scores),
            np.array(weighted_corr),
            np.array(scores_time_swap).T,
            np.array(scores_col_cycle).T,
            np.array(weighted_corr_time_swap).T,
            np.array(weighted_corr_col_cycle).T,
        )
    return scores, weighted_corr
