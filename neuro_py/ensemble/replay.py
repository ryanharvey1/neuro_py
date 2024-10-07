import multiprocessing

import numpy as np
from joblib import Parallel, delayed
from nelpy.analysis import replay
from nelpy.decoding import decode1D as decode


def WeightedCorr(weights, x=None, y=None):
    """
    Provide a matrix of weights, and this function will check the correlation between the X and Y dimensions of the matrix.
    You can provide the X-values and the Y-values for each column and row.
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


def WeightedCorrCirc(weights, x=None, alpha=None):
    """Compute the correlation between x and y dimensions of a matrix with angular (circular) values

    Args:
    weights -- A 2D numpy array of weights.
    x -- A 2D numpy array of x-values.
    alpha -- A 2D numpy array of angular (circular) y-values. If None, generate it.

    Returns:
    rho -- The correlation between x and y dimensions.
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


def weighted_correlation(posterior, time=None, place_bin_centers=None):
    def _m(x, w):
        """Weighted Mean"""
        return np.sum(x * w) / np.sum(w)

    def _cov(x, y, w):
        """Weighted Covariance"""
        return np.sum(w * (x - _m(x, w)) * (y - _m(y, w))) / np.sum(w)

    def _corr(x, y, w):
        """Weighted Correlation"""
        return _cov(x, y, w) / np.sqrt(_cov(x, x, w) * _cov(y, y, w))

    if time is None:
        time = np.arange(posterior.shape[1])
    if place_bin_centers is None:
        place_bin_centers = np.arange(posterior.shape[0])

    place_bin_centers = place_bin_centers.squeeze()
    posterior[np.isnan(posterior)] = 0.0

    return _corr(time[:, np.newaxis], place_bin_centers[np.newaxis, :], posterior.T)


def shuffle_and_score(posterior_array, w, normalize, tc, ds, dp):

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


def _shuffle_and_score(posterior_array, tuningcurve, w, normalize, ds, dp, n_shuffles):
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
    bst,
    tuningcurve,
    w=None,
    n_shuffles=1000,
    weights=None,
    normalize=False,
    parallel=True,
):

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
