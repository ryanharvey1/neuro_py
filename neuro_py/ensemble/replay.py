import multiprocessing
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
from joblib import Parallel, delayed
from nelpy import TuningCurve1D
from nelpy.analysis import replay
from nelpy.core import BinnedSpikeTrainArray
from nelpy.decoding import decode1D as decode
from numba import jit, njit, prange

from neuro_py.ensemble.pairwise_bias_correlation import (
    cosine_similarity_matrices,
    skew_bias_matrix,
)
from neuro_py.process.peri_event import crossCorr


@njit(parallel=True, fastmath=False, cache=True)
def __weighted_corr_2d_jit(
    weights: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    time_coords: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray, float, float, float, float]:
    # Handle NaN weights
    weights = np.nan_to_num(weights, nan=0.0)

    # Use the same dtype as weights for internal arrays
    dtype = weights.dtype
    x_dim, y_dim, t_dim = weights.shape

    # Early exit if no valid weights
    total_weight = np.sum(weights)
    if total_weight == 0.0:
        nan_val = np.array(np.nan, dtype=dtype)
        return (
            np.nan,
            np.full(t_dim, nan_val[()], dtype=dtype),
            np.full(t_dim, nan_val[()], dtype=dtype),
            nan_val[()],
            nan_val[()],
            nan_val[()],
            nan_val[()],
        )

    # Compute weighted means more efficiently
    mean_x = 0.0
    mean_y = 0.0
    mean_t = 0.0

    for i in prange(x_dim):
        for j in range(y_dim):
            for k in range(t_dim):
                w = weights[i, j, k]
                mean_x += w * x_coords[i]
                mean_y += w * y_coords[j]
                mean_t += w * time_coords[k]

    mean_x /= total_weight
    mean_y /= total_weight
    mean_t /= total_weight

    # Compute covariances efficiently
    cov_xt = 0.0
    cov_yt = 0.0
    cov_tt = 0.0
    cov_xx = 0.0
    cov_yy = 0.0

    for i in prange(x_dim):
        for j in range(y_dim):
            for k in range(t_dim):
                w = weights[i, j, k]
                dx = x_coords[i] - mean_x
                dy = y_coords[j] - mean_y
                dt = time_coords[k] - mean_t

                cov_xt += w * dx * dt
                cov_yt += w * dy * dt
                cov_tt += w * dt * dt
                cov_xx += w * dx * dx
                cov_yy += w * dy * dy

    cov_xt /= total_weight
    cov_yt /= total_weight
    cov_tt /= total_weight
    cov_xx /= total_weight
    cov_yy /= total_weight

    # Compute denominators
    denom_x = np.sqrt(cov_xx * cov_tt)
    denom_y = np.sqrt(cov_yy * cov_tt)

    if denom_x == 0.0 or denom_y == 0.0 or cov_tt == 0.0:
        nan_val = np.array(np.nan, dtype=dtype)
        return (
            np.nan,
            np.full(t_dim, nan_val[()], dtype=dtype),
            np.full(t_dim, nan_val[()], dtype=dtype),
            nan_val[()],
            nan_val[()],
            nan_val[()],
            nan_val[()],
        )

    # Compute correlations and slopes
    corr_x = cov_xt / denom_x
    corr_y = cov_yt / denom_y
    slope_x = cov_xt / cov_tt
    slope_y = cov_yt / cov_tt

    # Compute trajectories vectorized
    x_traj = np.empty(t_dim, dtype=dtype)
    y_traj = np.empty(t_dim, dtype=dtype)

    for k in prange(t_dim):
        x_traj[k] = mean_x + slope_x * (time_coords[k] - mean_t)
        y_traj[k] = mean_y + slope_y * (time_coords[k] - mean_t)

    # Compute spatiotemporal correlation
    spatiotemporal_corr = np.sqrt((corr_x**2 + corr_y**2) / 2) * np.sign(
        corr_x + corr_y
    )

    return (
        spatiotemporal_corr,
        x_traj,
        y_traj,
        np.array(slope_x, dtype=dtype)[()],
        np.array(slope_y, dtype=dtype)[()],
        np.array(mean_x, dtype=dtype)[()],
        np.array(mean_y, dtype=dtype)[()],
    )


def weighted_corr_2d(
    weights: np.ndarray,
    x_coords: Optional[np.ndarray] = None,
    y_coords: Optional[np.ndarray] = None,
    time_coords: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray, np.ndarray, float, float, float, float]:
    """
    Calculate the weighted correlation between the X and Y dimensions of the matrix.

    Parameters
    ----------
    weights : np.ndarray
        A matrix of weights.
    x_coords : Optional[np.ndarray], optional
        X-values for each column and row, by default None.
    y_coords : Optional[np.ndarray], optional
        Y-values for each column and row, by default None.
    time_coords : Optional[np.ndarray], optional
        Time-values for each column and row, by default None.

    Returns
    -------
    Tuple[float, np.ndarray, np.ndarray, float, float, float, float]
        The weighted correlation coefficient, x trajectory, y trajectory,
        slope_x, slope_y, mean_x, mean_y.

    Examples
    --------
    >>> import numpy as np
    >>> weights = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    >>> x_coords = np.array([0, 1])
    >>> y_coords = np.array([0, 1])
    >>> time_coords = np.array([0, 1, 2])
    >>> weighted_corr_2d(weights, x_coords, y_coords, time_coords)

    """
    x_dim, y_dim, t_dim = weights.shape
    dtype = weights.dtype

    x_coords = (
        np.arange(x_dim, dtype=dtype)
        if x_coords is None
        else np.asarray(x_coords, dtype=dtype)
    )
    y_coords = (
        np.arange(y_dim, dtype=dtype)
        if y_coords is None
        else np.asarray(y_coords, dtype=dtype)
    )
    time_coords = (
        np.arange(t_dim, dtype=dtype)
        if time_coords is None
        else np.asarray(time_coords, dtype=dtype)
    )

    return __weighted_corr_2d_jit(weights, x_coords, y_coords, time_coords)


def _position_estimator_1d(
    posterior_prob: np.ndarray, bin_centers: np.ndarray, method: str, n_time_bins: int
):
    """Helper function for 1D position decoding."""
    if posterior_prob.shape[1] != len(bin_centers):
        raise ValueError(
            f"Posterior shape {posterior_prob.shape[1]} doesn't match "
            f"bin_centers length {len(bin_centers)}"
        )

    position = np.full(n_time_bins, np.nan)

    for t in range(n_time_bins):
        P = posterior_prob[t]
        if np.sum(P) > 0:
            if method == "com":
                # Normalize probabilities
                P_norm = P / np.sum(P)
                # Calculate center of mass
                position[t] = np.sum(bin_centers * P_norm)
            elif method == "max":
                # Find the index of the maximum probability
                max_idx = np.argmax(P)
                position[t] = bin_centers[max_idx]

    return position


def _position_estimator_2d(
    posterior_prob: np.ndarray,
    ybin_centers: np.ndarray,
    xbin_centers: np.ndarray,
    method: str,
    n_time_bins: int,
):
    """Helper function for 2D position decoding."""
    if posterior_prob.shape[1] != len(ybin_centers) or posterior_prob.shape[2] != len(
        xbin_centers
    ):
        raise ValueError(
            f"Posterior shape {posterior_prob.shape[1:]} doesn't match "
            f"bin_centers shapes ({len(ybin_centers)}, {len(xbin_centers)})"
        )

    # Create coordinate meshgrids
    # Using xy indexing so xx contains x-coords and yy contains y-coords
    xx, yy = np.meshgrid(xbin_centers, ybin_centers, indexing="xy")

    position = np.full((n_time_bins, 2), np.nan)  # [x, y] coordinates

    for t in range(n_time_bins):
        P = posterior_prob[t]
        if np.sum(P) > 0:
            if method == "com":
                # Normalize probabilities
                P_norm = P / np.sum(P)

                # Calculate center of mass
                position[t, 0] = np.sum(xx * P_norm)  # x-coordinate
                position[t, 1] = np.sum(yy * P_norm)  # y-coordinate

            elif method == "max":
                # Find the index of the maximum probability
                max_idx = np.unravel_index(np.argmax(P), P.shape)
                position[t, 0] = xx[max_idx]  # x-coordinate
                position[t, 1] = yy[max_idx]  # y-coordinate

    return position


def position_estimator(
    posterior_prob: np.ndarray, *bin_centers: np.ndarray, method: str = "com"
) -> np.ndarray:
    """
    Decode 1D or 2D position from posterior probability distributions.

    Parameters
    ----------
    posterior_prob : np.ndarray
        Posterior probability distributions over spatial bins for each time bin.
        For 1D: shape (n_time_bins, n_bins)
        For 2D: shape (n_time_bins, n_y_bins, n_x_bins)
        Each time slice should contain non-negative values.
    *bin_centers : np.ndarray
        Coordinate values for the center of each spatial bin.
        For 1D: single array of shape (n_bins,)
        For 2D: two arrays - y_bin_centers of shape (n_y_bins,) and
                x_bin_centers of shape (n_x_bins,)
    method : str, optional
        Decoding method to use. Options are:
        - "com" : Center of mass (weighted average) (default)
        - "max" : Maximum a posteriori (position of maximum probability)

    Returns
    -------
    position : np.ndarray
        Decoded positions for each time bin.
        For 1D: shape (n_time_bins,) containing position coordinates
        For 2D: shape (n_time_bins, 2) where position[:, 0] contains
                x-coordinates and position[:, 1] contains y-coordinates
        Time bins with zero probability sum are filled with NaN.

    Raises
    ------
    ValueError
        If method is not "com" or "max", or if dimensions don't match expectations.

    Notes
    -----
    For the center of mass method, probabilities are normalized before computing
    the weighted average. For time bins where all probabilities are zero,
    the decoded position is set to NaN.

    The function automatically detects whether to perform 1D or 2D decoding
    based on the shape of the posterior_prob array and number of bin_centers provided.

    Examples
    --------
    1D example:
    >>> posterior_1d = np.random.rand(10, 20)  # 10 time bins, 20 spatial bins
    >>> bin_centers = np.linspace(0, 19, 20)
    >>> positions = decode_position(posterior_1d, bin_centers)
    >>> positions.shape
    (10,)

    2D example:
    >>> posterior_2d = np.random.rand(10, 5, 4)  # 10 time bins, 5x4 spatial grid
    >>> y_centers = np.linspace(0, 4, 5)
    >>> x_centers = np.linspace(0, 3, 4)
    >>> positions = decode_position(posterior_2d, y_centers, x_centers)
    >>> positions.shape
    (10, 2)
    """
    if method not in ["com", "max"]:
        raise ValueError(f"Method '{method}' not recognized. Use 'com' or 'max'.")

    n_dims = len(posterior_prob.shape) - 1  # Subtract time dimension
    n_time_bins = posterior_prob.shape[0]

    if n_dims == 1:
        return _position_estimator_1d(
            posterior_prob, bin_centers[0], method, n_time_bins
        )
    elif n_dims == 2:
        if len(bin_centers) != 2:
            raise ValueError(
                "For 2D decoding, provide exactly 2 bin_centers arrays (y_centers, x_centers)"
            )
        return _position_estimator_2d(
            posterior_prob, bin_centers[0], bin_centers[1], method, n_time_bins
        )
    else:
        raise ValueError(f"Only 1D and 2D decoding supported, got {n_dims}D")


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
    return_full_output: bool = False,
) -> Union[float, Tuple[float, np.ndarray, float, float, float, float]]:
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
    return_full_output : bool, optional
        If True, return trajectory, slopes, and means in addition to correlation, by default False.

    Returns
    -------
    Union[float, Tuple[float, np.ndarray, float, float, float]]
        If return_full_output is False:
            The weighted correlation coefficient (float).
        If return_full_output is True:
            Tuple of (correlation, place_trajectory, slope_place, mean_time, mean_place)
            where:
            - correlation: weighted correlation coefficient
            - place_trajectory: place position at each time bin
            - slope_place: slope of place vs time
            - mean_time: mean time value
            - mean_place: mean place value
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

    time = np.asarray(time)
    place_bin_centers = place_bin_centers.squeeze()
    posterior = np.asarray(posterior, copy=True)
    posterior[np.isnan(posterior)] = 0.0

    correlation = _corr(
        time[:, np.newaxis], place_bin_centers[np.newaxis, :], posterior.T
    )

    if not return_full_output:
        return correlation

    # Compute full output (trajectory, slopes, means)
    weights = posterior.T.flatten()
    time_2d, place_2d = np.meshgrid(time, place_bin_centers, indexing="ij")
    time_flat = time_2d.flatten()
    place_flat = place_2d.flatten()

    # Compute weighted means
    total_weight = np.sum(weights)
    mean_time = np.sum(weights * time_flat) / total_weight
    mean_place = np.sum(weights * place_flat) / total_weight

    # Compute covariances
    cov_time_place = (
        np.sum(weights * (time_flat - mean_time) * (place_flat - mean_place))
        / total_weight
    )
    cov_time_time = np.sum(weights * (time_flat - mean_time) ** 2) / total_weight

    # Compute slope and trajectory
    slope_place = cov_time_place / cov_time_time if cov_time_time != 0 else 0.0
    place_trajectory = mean_place + slope_place * (time - mean_time)

    return (
        correlation,
        place_trajectory,
        slope_place,
        mean_time,
        mean_place,
    )


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

    num_cores = 1

    if parallel:
        # all but one core
        num_cores = multiprocessing.cpu_count() - 1

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


@jit(nopython=True)
def compute_bias_matrix_optimized_(spike_times, neuron_ids, total_neurons):
    """
    Optimized computation of the bias matrix B_k for a given sequence of spikes using vectorized operations.

    Parameters:
    - spike_times: list or array of spike times for the sequence.
    - neuron_ids: list or array of neuron identifiers corresponding to spike_times.
    - total_neurons: total number of neurons being considered.

    Returns:
    - bias_matrix: A matrix of size (total_neurons, total_neurons) representing the bias.
    """

    # Create an empty bias matrix
    bias_matrix = np.full((total_neurons, total_neurons), 0.5)

    # Create boolean masks for all neurons in advance
    masks = [neuron_ids == i for i in range(total_neurons)]

    # Iterate over each pair of neurons
    for i in range(total_neurons):
        spikes_i = spike_times[masks[i]]
        size_i = spikes_i.size

        if size_i == 0:
            continue  # Skip if neuron i has no spikes

        for j in range(total_neurons):
            if i == j:
                continue  # Skip self-correlation

            spikes_j = spike_times[masks[j]]
            size_j = spikes_j.size

            if size_j == 0:
                continue  # Skip if neuron j has no spikes

            crosscorr = crossCorr(spikes_i, spikes_j, 0.001, 100)

            # Count how many times neuron i spikes before neuron j
            bias_matrix[i, j] = np.divide(crosscorr[:50].sum(), crosscorr[51:].sum())

    return bias_matrix


class PairwiseBias(object):
    """
    Pairwise bias analysis for comparing task and post-task spike sequences.

    Parameters
    ----------
    num_shuffles : int, optional
        Number of shuffles to perform for significance testing. Default is 300.
    n_jobs : int, optional
        Number of parallel jobs to run for computing correlations. Default is 10.

    Attributes
    ----------
    total_neurons : int, or None
        Total number of neurons in the dataset.
    task_skew_bias : np.ndarray, or None
        Normalized skew-bias matrix for the task data.
    observed_correlation_ : np.ndarray, or None
        Observed cosine similarity between task and post-task bias matrices.
    shuffled_correlations_ : np.ndarray, or None
        Shuffled cosine similarities for significance testing.
    z_score_ : np.ndarray, or None
        Z-score of the observed correlation compared to the shuffled distribution.
    p_value_ : np.ndarray, or None
        p-value for significance test.

    Methods
    -------
    fit(task_spikes: Union[List[float], np.ndarray], task_neurons: Union[List[int], np.ndarray]) -> 'PairwiseBias'
        Fit the model using the task spike data.
    transform(post_spikes: Union[List[float], np.ndarray], post_neurons: Union[List[int], np.ndarray], post_intervals: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
        Transform the post-task data to compute z-scores and p-values.
    fit_transform(task_spikes: Union[List[float], np.ndarray], task_neurons: Union[List[int], np.ndarray], post_spikes: Union[List[float], np.ndarray], post_neurons: Union[List[int], np.ndarray], post_intervals: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
        Fit the model with task data and transform the post-task data.
    """

    def __init__(
        self, num_shuffles: int = 300, n_jobs: int = 10, fillneutral: float = np.nan
    ):
        self.num_shuffles = num_shuffles
        self.n_jobs = n_jobs
        self.fillneutral = fillneutral
        self.total_neurons = None
        self.task_skew_bias = None
        self.observed_correlation_ = None
        self.shuffled_correlations_ = None
        self.z_score_ = None
        self.p_value_ = None

    @staticmethod
    def bias_matrix(
        spike_times: np.ndarray,
        neuron_ids: np.ndarray,
        total_neurons: int,
        fillneutral: float = np.nan,
    ) -> np.ndarray:
        """
        Optimized computation of the bias matrix B_k for a given sequence of spikes using vectorized operations.

        Parameters
        ----------
        spike_times : np.ndarray
            Spike times for the sequence.
        neuron_ids : np.ndarray
            Neuron identifiers corresponding to spike_times.
        total_neurons : int
            Total number of neurons being considered.
        fillneutral : float, optional
            Value to fill the diagonal of the bias matrix and other empty
            combinations, by default np.nan.

        Returns
        -------
        np.ndarray
            A matrix of size (total_neurons, total_neurons) representing the bias.
        """
        return skew_bias_matrix(spike_times, neuron_ids, total_neurons, fillneutral)

    @staticmethod
    def cosine_similarity_matrices(matrix1: np.ndarray, matrix2: np.ndarray) -> float:
        """
        Computes the cosine similarity between two flattened bias matrices.

        Parameters
        ----------
        matrix1 : np.ndarray
            A normalized bias matrix.
        matrix2 : np.ndarray
            Another normalized bias matrix.

        Returns
        -------
        float
            The cosine similarity between the two matrices.
        """
        return cosine_similarity_matrices(matrix1, matrix2)

    def observed_and_shuffled_correlation(
        self,
        post_spikes: np.ndarray,
        post_neurons: np.ndarray,
        task_skew_bias: np.ndarray,
        post_intervals: np.ndarray,
        interval_i: int,
    ) -> Tuple[float, List[float]]:
        """
        Compute observed and shuffled correlation for a given post-task interval.

        Parameters
        ----------
        post_spikes : np.ndarray
            Spike times during post-task (e.g., sleep).
        post_neurons : np.ndarray
            Neuron identifiers for post-task spikes.
        task_normalized : np.ndarray
            Normalized task bias matrix.
        post_intervals : np.ndarray
            Intervals for post-task epochs.
        interval_i : int
            Index of the current post-task interval.

        Returns
        -------
        Tuple[float, List[float]]
            The observed correlation and a list of shuffled correlations.
        """
        post_neurons = np.asarray(post_neurons, dtype=int)

        start, end = post_intervals[interval_i]
        start_idx = np.searchsorted(post_spikes, start, side="left")
        end_idx = np.searchsorted(post_spikes, end, side="right")

        filtered_spikes = post_spikes[start_idx:end_idx]
        filtered_neurons = post_neurons[start_idx:end_idx]

        post_skew_bias = self.bias_matrix(
            filtered_spikes,
            filtered_neurons,
            self.total_neurons,
            fillneutral=self.fillneutral,
        )

        observed_correlation = self.cosine_similarity_matrices(
            task_skew_bias, post_skew_bias
        )

        shuffled_correlation = []
        for _ in range(self.num_shuffles):
            shuffled_neurons = np.random.permutation(filtered_neurons)
            shuffled_skew_bias = self.bias_matrix(
                filtered_spikes,
                shuffled_neurons,
                self.total_neurons,
                fillneutral=self.fillneutral,
            )
            shuffled_correlation.append(
                self.cosine_similarity_matrices(task_skew_bias, shuffled_skew_bias)
            )

        return observed_correlation, shuffled_correlation

    def fit(
        self,
        task_spikes: np.ndarray,
        task_neurons: np.ndarray,
        task_intervals: np.ndarray = None,
    ) -> "PairwiseBias":
        """
        Fit the model using the task spike data.

        Parameters
        ----------
        task_spikes : np.ndarray
            Spike times during the task.
        task_neurons : np.ndarray
            Neuron identifiers for task spikes.
        task_intervals : np.ndarray, optional
            Intervals for task epochs, by default None. If None, the entire task
            data is used. Otherwise, the average bias matrix is computed across
            all task intervals. Shape: (n_intervals, 2).

        Returns
        -------
        PairwiseBias
            Returns the instance itself.
        """
        # Convert task_neurons to numpy array of integers
        task_neurons = np.asarray(task_neurons, dtype=int)

        # Calculate the total number of neurons based on unique entries in task_neurons
        self.total_neurons = len(np.unique(task_neurons))

        if task_intervals is None:
            # Compute bias matrix for task data and normalize
            task_skew_bias = self.bias_matrix(
                task_spikes,
                task_neurons,
                self.total_neurons,
                fillneutral=self.fillneutral,
            )
            self.task_skew_bias = task_skew_bias
        else:
            # Compute bias matrices for each task interval
            task_skew_biases = []

            for interval in task_intervals:
                # find the indices of spikes within the interval
                start_idx = np.searchsorted(task_spikes, interval[0], side="left")
                end_idx = np.searchsorted(task_spikes, interval[1], side="right")

                # Extract spikes and neurons within the interval
                interval_spikes = task_spikes[start_idx:end_idx]
                interval_neurons = task_neurons[start_idx:end_idx]

                # Compute the bias matrix for the interval
                interval_skew_bias = self.bias_matrix(
                    interval_spikes,
                    interval_neurons,
                    self.total_neurons,
                    fillneutral=self.fillneutral,
                )
                task_skew_biases.append(interval_skew_bias)

            # Average the normalized bias matrices
            # I expect to see RuntimeWarnings in this block
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                self.task_skew_bias = np.nanmean(task_skew_biases, axis=0)
        return self

    def transform(
        self,
        post_spikes: np.ndarray,
        post_neurons: np.ndarray,
        post_intervals: np.ndarray,
        allow_reverse_replay: bool = False,
        parallel: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Transform the post-task data to compute z-scores and p-values.

        Parameters
        ----------
        post_spikes : np.ndarray
            Spike times during post-task (e.g., sleep).
        post_neurons : np.ndarray
            Neuron identifiers for post-task spikes.
        post_intervals : np.ndarray
            Intervals for post-task epochs. Shape: (n_intervals, 2).
        allow_reverse_replay : bool, optional
            Whether to allow reverse sequences, by default False.
        parallel : bool, optional
            Whether to run in parallel, by default True.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            z_score: The z-score of the observed correlation compared to the shuffled distribution.
            p_value: p-value for significance test.
            observed_correlation_: The observed correlation for each interval.
        """
        # Check if the number of jobs is less than the number of intervals
        if post_intervals.shape[0] < self.n_jobs:
            self.n_jobs = post_intervals.shape[0]

        if parallel:
            observed_correlation, shuffled_correlations = zip(
                *Parallel(n_jobs=self.n_jobs)(
                    delayed(self.observed_and_shuffled_correlation)(
                        post_spikes,
                        post_neurons,
                        self.task_skew_bias,
                        post_intervals,
                        interval_i,
                    )
                    for interval_i in range(post_intervals.shape[0])
                )
            )
        else:  # Run in serial for debugging
            observed_correlation, shuffled_correlations = zip(
                *[
                    self.observed_and_shuffled_correlation(
                        post_spikes,
                        post_neurons,
                        self.task_skew_bias,
                        post_intervals,
                        interval_i,
                    )
                    for interval_i in range(post_intervals.shape[0])
                ]
            )

        self.observed_correlation_ = np.array(
            observed_correlation
        )  # Shape: (n_intervals,)
        self.shuffled_correlations_ = np.array(
            shuffled_correlations
        )  # Shape: (n_intervals, n_shuffles)

        shuffled_mean = np.mean(self.shuffled_correlations_, axis=1)
        shuffled_std = np.std(self.shuffled_correlations_, axis=1)
        self.z_score_ = (self.observed_correlation_ - shuffled_mean) / shuffled_std

        observed_correlation = self.observed_correlation_
        shuffled_correlations = self.shuffled_correlations_
        if allow_reverse_replay:
            observed_correlation = np.abs(observed_correlation)
            shuffled_correlations = np.abs(shuffled_correlations)

        self.p_value_ = (
            np.sum(
                shuffled_correlations.T > observed_correlation,
                axis=0,
            )
            + 1
        ) / (self.num_shuffles + 1)

        return self.z_score_, self.p_value_, self.observed_correlation_

    def fit_transform(
        self,
        task_spikes: np.ndarray,
        task_neurons: np.ndarray,
        task_intervals: np.ndarray,
        post_spikes: np.ndarray,
        post_neurons: np.ndarray,
        post_intervals: np.ndarray,
        allow_reverse_replay: bool = False,
        parallel: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit the model with task data and transform the post-task data.

        Parameters
        ----------
        task_spikes : np.ndarray
            Spike times during the task.
        task_neurons : np.ndarray
            Neuron identifiers for task spikes.
        task_intervals : np.ndarray
            Intervals for task epochs. Shape: (n_intervals, 2).
        post_spikes : np.ndarray
            Spike times during post-task (e.g., sleep).
        post_neurons : np.ndarray
            Neuron identifiers for post-task spikes.
        post_intervals : np.ndarray
            Intervals for post-task epochs. Shape: (n_intervals, 2).
        allow_reverse_replay : bool, optional
            Whether to allow reverse sequences, by default False.
        parallel : bool, optional
            Whether to run in parallel, by default True.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            z_score: The z-score of the observed correlation compared to the shuffled distribution.
            p_value: p-value for significance test.
            observed_correlation_: The observed correlation for each interval.
        """
        self.fit(task_spikes, task_neurons, task_intervals)
        return self.transform(
            post_spikes, post_neurons, post_intervals, allow_reverse_replay, parallel
        )


def bottom_up_replay_detection(
    posterior: np.ndarray,
    time_centers: np.ndarray,
    bin_centers: np.ndarray,
    speed_times: np.ndarray,
    speed_values: np.ndarray,
    window_dt: Optional[float] = None,
    speed_thresh: float = 5.0,
    spread_thresh: float = 10.0,
    com_jump_thresh: float = 20.0,
    merge_spatial_gap: float = 20.0,
    merge_time_gap: float = 0.05,
    min_duration: float = 0.1,
    dispersion_thresh: float = 12.0,
    method: str = "com",
) -> Tuple[np.ndarray, dict]:
    """
    Bottom-up replay detector following Widloski & Foster (2022) "Replay detection and analysis".

    Parameters
    ----------
    posterior : np.ndarray
        Shape (n_time, n_space_bins) posterior probability for each time window.
    time_centers : np.ndarray
        Center time of each posterior time bin (length n_time) in seconds.
    bin_centers : np.ndarray
        Spatial bin centers (length n_space_bins) in same units as thresholds (cm).
    speed_times, speed_values : np.ndarray
        Time stamps and speed values (cm/s) for the animal; used to interpolate
        speed at time_centers.
    window_dt : Optional[float]
        Duration of each posterior time bin. If None, computed from time_centers diff.
    speed_thresh : float
        Speed threshold for filtering candidate replays (cm/s).
    spread_thresh : float
        Spread threshold for filtering candidate replays (cm).
    com_jump_thresh : float
        Center-of-mass jump threshold for filtering candidate replays (cm).
    merge_spatial_gap : float
        Spatial gap threshold for merging candidate replays (cm).
    merge_time_gap : float
        Temporal gap threshold for merging candidate replays (s).
    min_duration : float
        Minimum duration for keeping candidate replays (s).
    dispersion_thresh : float
        Dispersion threshold (mean absolute deviation of COM across sequence) for labeling replays (cm).

    Returns
    -------
    replays : np.ndarray
        Array of replay events that passed dispersion threshold; shape (k,2) start/end times.
    meta : dict
        Metadata dict with keys:
         - 'candidates': candidate subsequences before dispersion filter; list of dicts with keys:
            - 'start_time': start time of each candidate
            - 'end_time': end time of each candidate
            - 'start_idx': start index of each candidate
            - 'end_idx': end index of each candidate
            - 'duration': duration of each candidate (s)
            - 'D2': dispersion (RMS radial deviation) per candidate (cm)
            - 'com_trace': NaN-removed sequence of center-of-mass positions used for metrics
            - 'path_length': path length (sum of consecutive COM step distances) across valid bins (cm)
            - 'maxJump_NaN': maximum consecutive-step jump computed on the raw COM slice (may be NaN)
            - 'maxJump_NaNremoved': maximum consecutive-step jump computed on the NaN-removed COM trace (cm)
            - 'maxJump_NaNremoved_time': maximum temporal gap between valid (NaN-removed) samples (s)
            - 'posteriorSpreadMax': maximum per-bin posterior spread across the candidate (cm)
            - 'posteriorSpreadMean': mean per-bin posterior spread across the candidate (cm)
         - 'com': center-of-mass per kept bin (array of shape (n_time, ) or (n_time,2))
         - 'spread': posterior spread per kept bin (array of length n_time)
         - 'mask': boolean mask of kept bins (array of length n_time)

    Notes
    -----
    This implementation assumes 1D and 2D spatial decoding.
    """
    posterior = np.asarray(posterior)
    time_centers = np.asarray(time_centers)

    # Support posteriors where time is the last axis (space..., time)
    # Move time axis to front so internal code works with shape (n_time, ...)
    if posterior.ndim >= 2 and posterior.shape[0] != time_centers.shape[0]:
        if posterior.shape[-1] == time_centers.shape[0]:
            posterior = np.moveaxis(posterior, -1, 0)
        else:
            raise ValueError("posterior time dimension does not match time_centers")
    # bin_centers may be a 1D array for 1D posteriors or a tuple (y_centers, x_centers)
    # for 2D posteriors. Don't force-cast a tuple into an ndarray.
    if posterior.ndim == 3:
        if not (isinstance(bin_centers, (tuple, list)) and len(bin_centers) == 2):
            raise ValueError(
                "For 2D posterior, bin_centers must be (y_centers, x_centers)"
            )
        y_centers, x_centers = bin_centers
    else:
        bin_centers = np.asarray(bin_centers)

    if posterior.shape[0] != time_centers.shape[0]:
        raise ValueError(
            "posterior and time_centers must have matching first dimension"
        )

    n_time = posterior.shape[0]

    # bin duration
    if window_dt is None:
        if n_time > 1:
            window_dt = np.median(np.diff(time_centers))
        else:
            window_dt = 0.0

    # compute COMs and spread; support 1D and 2D posteriors
    if posterior.ndim == 2:
        # 1D posterior: shape (n_time, n_space)
        com = position_estimator(posterior, bin_centers, method=method)

        # compute posterior spread (weighted std)
        spread = np.full(n_time, np.nan)
        for ti in range(n_time):
            P = posterior[ti]
            s = np.sum(P)
            if s > 0:
                Pn = P / s
                mu = np.sum(bin_centers * Pn)
                spread[ti] = np.sqrt(np.sum(((bin_centers - mu) ** 2) * Pn))

        # compute COM jump sizes (between consecutive bins)
        com_jump = np.full(n_time, np.nan)
        com_diff = np.abs(np.diff(com, prepend=np.nan))
        com_jump[1:] = com_diff[1:]

    elif posterior.ndim == 3:
        # 2D posterior: shape (n_time, ny, nx)
        y_centers, x_centers = bin_centers
        xx, yy = np.meshgrid(x_centers, y_centers, indexing="xy")

        com = np.full((n_time, 2), np.nan)
        spread = np.full(n_time, np.nan)
        for ti in range(n_time):
            P = posterior[ti]
            s = np.sum(P)
            if s > 0:
                Pn = P / s
                mu_x = np.sum(xx * Pn)
                mu_y = np.sum(yy * Pn)
                com[ti, 0] = mu_x
                com[ti, 1] = mu_y
                # RMS distance from mean
                spread[ti] = np.sqrt(np.sum(((xx - mu_x) ** 2 + (yy - mu_y) ** 2) * Pn))

        # compute COM jump sizes (Euclidean distance between consecutive COMs)
        com_jump = np.full(n_time, np.nan)
        for ti in range(1, n_time):
            if not np.any(np.isnan(com[ti - 1])) and not np.any(np.isnan(com[ti])):
                com_jump[ti] = np.linalg.norm(com[ti] - com[ti - 1])
    else:
        raise ValueError("posterior must be 2D (time,x) or 3D (time,y,x)")

    # interpolate speed at time_centers
    speed = np.interp(time_centers, speed_times, speed_values)

    # treat NaN COM-jumps (e.g. first bin) as failing the com_jump criterion
    # to avoid letting NaNs silently pass (np.nan_to_num -> 0). Set NaNs to +inf
    # so they are excluded when compared to com_jump_thresh.
    com_jump = np.array(com_jump, copy=True)
    com_jump[np.isnan(com_jump)] = np.inf

    # mask time bins that satisfy all three criteria
    mask = (
        (speed < speed_thresh) & (spread < spread_thresh) & (com_jump < com_jump_thresh)
    )

    # find contiguous subsequences of True in mask
    subseqs = []
    if np.any(mask):
        edges = np.diff(mask.astype(int))
        starts = np.where(edges == 1)[0] + 1
        ends = np.where(edges == -1)[0] + 1
        if mask[0]:
            starts = np.concatenate(([0], starts))
        if mask[-1]:
            ends = np.concatenate((ends, [n_time]))

        for s, e in zip(starts, ends):
            subseqs.append(
                {
                    "start_idx": s,
                    "end_idx": e,
                    "start_time": time_centers[s],
                    "end_time": time_centers[e - 1],
                }
            )

    # merge neighboring subsequences based on spatial and temporal gaps
    merged = []
    for seq in subseqs:
        if not merged:
            merged.append(seq)
            continue
        prev = merged[-1]
        temporal_gap = seq["start_time"] - prev["end_time"]
        # compute spatial gap differently for 1D vs 2D COM
        if posterior.ndim == 2:
            spatial_gap = np.abs(com[seq["start_idx"]] - com[prev["end_idx"] - 1])
        else:
            # com entries are 2D vectors (mu_x, mu_y)
            a = com[seq["start_idx"]]
            b = com[prev["end_idx"] - 1]
            if np.any(np.isnan(a)) or np.any(np.isnan(b)):
                spatial_gap = np.inf
            else:
                spatial_gap = float(np.linalg.norm(a - b))

        if temporal_gap <= merge_time_gap and spatial_gap <= merge_spatial_gap:
            # merge
            prev["end_idx"] = seq["end_idx"]
            prev["end_time"] = seq["end_time"]
        else:
            merged.append(seq)

    # candidate sequences: duration > min_duration
    candidates = []
    for seq in merged:
        duration = (
            seq["end_time"] - seq["start_time"] + (window_dt if window_dt > 0 else 0.0)
        )
        if duration >= min_duration:
            # record COM trace for sequence and compute metrics on NaN-removed trace
            idxs = np.arange(seq["start_idx"], seq["end_idx"])
            com_trace = com[idxs]

            # remove NaN entries (bins with no posterior mass)
            if posterior.ndim == 2:
                # 1D case: com_trace is 1D array
                valid_mask = ~np.isnan(com_trace)
                com_trace_valid = com_trace[valid_mask]
            else:
                # 2D case: com_trace is (n_bins, 2)
                valid_mask = ~np.isnan(com_trace).any(axis=1)
                com_trace_valid = com_trace[valid_mask]

            # dispersion D2 = RMS radial deviation from centroid (match MATLAB)
            if com_trace_valid.size == 0:
                D2 = np.nan
                centroid = np.nan
            else:
                if posterior.ndim == 2:
                    centroid = np.nanmean(com_trace_valid)
                    D2 = np.sqrt(np.nanmean((com_trace_valid - centroid) ** 2))
                else:
                    centroid = np.nanmean(com_trace_valid, axis=0)
                    diffs = np.linalg.norm(com_trace_valid - centroid, axis=1)
                    D2 = np.sqrt(np.nanmean(diffs**2))

            # compute path length (sum of Euclidean distances between consecutive valid COM points)
            if com_trace_valid.size == 0:
                path_length = 0.0
            else:
                if com_trace_valid.ndim == 1:
                    steps = np.abs(np.diff(com_trace_valid))
                else:
                    steps = np.linalg.norm(np.diff(com_trace_valid, axis=0), axis=1)
                path_length = float(np.nansum(steps))

            # compute maxJump on raw (may contain NaNs) and on NaN-removed trace
            # For raw trace: compute diffs and allow NaNs to propagate (max may be NaN)
            try:
                if com_trace.size == 0:
                    maxJump_NaN = np.nan
                else:
                    if com_trace.ndim == 1:
                        raw_steps = np.abs(np.diff(com_trace))
                    else:
                        raw_steps = np.linalg.norm(np.diff(com_trace, axis=0), axis=1)
                    maxJump_NaN = (
                        float(np.nanmax(raw_steps)) if raw_steps.size > 0 else np.nan
                    )
            except Exception:
                maxJump_NaN = np.nan

            # For NaN-removed trace
            if com_trace_valid.size == 0:
                maxJump_NaNremoved = np.nan
                maxJump_NaNremoved_time = np.nan
            else:
                if com_trace_valid.ndim == 1:
                    valid_steps = np.abs(np.diff(com_trace_valid))
                else:
                    valid_steps = np.linalg.norm(
                        np.diff(com_trace_valid, axis=0), axis=1
                    )
                maxJump_NaNremoved = (
                    float(np.nanmax(valid_steps)) if valid_steps.size > 0 else np.nan
                )

                # compute times of valid samples to get max time gap
                times_seq = time_centers[idxs]
                times_valid = times_seq[valid_mask]
                if times_valid.size > 1:
                    maxJump_NaNremoved_time = float(np.max(np.diff(times_valid)))
                else:
                    maxJump_NaNremoved_time = np.nan

            # posteriorSpreadMax and posteriorSpreadMean for the sequence (NaN-removed)
            seq_spreads = spread[idxs]
            seq_spreads_valid = seq_spreads[~np.isnan(seq_spreads)]
            if seq_spreads_valid.size > 0:
                posteriorSpreadMax = float(np.max(seq_spreads_valid))
                posteriorSpreadMean = float(np.mean(seq_spreads_valid))
            else:
                posteriorSpreadMax = np.nan
                posteriorSpreadMean = np.nan

            candidates.append(
                {
                    "start_time": seq["start_time"],
                    "end_time": seq["end_time"],
                    "start_idx": seq["start_idx"],
                    "end_idx": seq["end_idx"],
                    "duration": duration,
                    "D2": D2,
                    "com_trace": com_trace_valid,
                    "path_length": path_length,
                    "maxJump_NaN": maxJump_NaN,
                    "maxJump_NaNremoved": maxJump_NaNremoved,
                    "maxJump_NaNremoved_time": maxJump_NaNremoved_time,
                    "posteriorSpreadMax": posteriorSpreadMax,
                    "posteriorSpreadMean": posteriorSpreadMean,
                }
            )

    # select replays by dispersion threshold
    replays = []
    for c in candidates:
        if c["D2"] > dispersion_thresh:
            replays.append([c["start_time"], c["end_time"]])

    replays = np.array(replays)

    meta = {
        "candidates": candidates,
        "com": com,
        "spread": spread,
        "mask": mask,
        "window_dt": window_dt,
    }

    return replays, meta
