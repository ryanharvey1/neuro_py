import multiprocessing
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
from joblib import Parallel, delayed
from nelpy import TuningCurve1D
from nelpy.analysis import replay
from nelpy.core import BinnedSpikeTrainArray
from nelpy.decoding import decode1D as decode
from numba import jit, njit

from neuro_py.ensemble.pairwise_bias_correlation import (
    cosine_similarity_matrices,
    skew_bias_matrix,
)
from neuro_py.process.peri_event import crossCorr


@njit
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

    n_points = x_dim * y_dim * t_dim

    # Preallocate flattened arrays
    w_flat = np.empty(n_points, dtype=dtype)
    x_flat = np.empty(n_points, dtype=dtype)
    y_flat = np.empty(n_points, dtype=dtype)
    t_flat = np.empty(n_points, dtype=dtype)

    idx = 0
    for i in range(x_dim):
        for j in range(y_dim):
            for k in range(t_dim):
                w = weights[i, j, k]
                w_flat[idx] = w
                x_flat[idx] = x_coords[i]
                y_flat[idx] = y_coords[j]
                t_flat[idx] = time_coords[k]
                idx += 1

    total_weight = np.sum(w_flat)
    if total_weight == 0.0:
        nan_val = np.array(np.nan, dtype=dtype)  # Create NaN in correct dtype
        return (
            np.nan,  # First value remains float64 for consistency
            np.full(t_dim, nan_val[()], dtype=dtype),  # Use same dtype as weights
            np.full(t_dim, nan_val[()], dtype=dtype),
            nan_val[()],
            nan_val[()],
            nan_val[()],
            nan_val[()],
        )

    mean_x = np.sum(w_flat * x_flat) / total_weight
    mean_y = np.sum(w_flat * y_flat) / total_weight
    mean_t = np.sum(w_flat * t_flat) / total_weight

    cov_xt = np.sum(w_flat * (x_flat - mean_x) * (t_flat - mean_t)) / total_weight
    cov_yt = np.sum(w_flat * (y_flat - mean_y) * (t_flat - mean_t)) / total_weight
    cov_tt = np.sum(w_flat * (t_flat - mean_t) ** 2) / total_weight
    cov_xx = np.sum(w_flat * (x_flat - mean_x) ** 2) / total_weight
    cov_yy = np.sum(w_flat * (y_flat - mean_y) ** 2) / total_weight

    denom_x = np.sqrt(cov_xx * cov_tt)
    denom_y = np.sqrt(cov_yy * cov_tt)

    if denom_x == 0.0 or denom_y == 0.0 or cov_tt == 0.0:
        nan_val = np.array(np.nan, dtype=dtype)  # Create NaN in correct dtype
        return (
            np.nan,  # First value remains float64 for consistency
            np.full(t_dim, nan_val[()], dtype=dtype),  # Use same dtype as weights
            np.full(t_dim, nan_val[()], dtype=dtype),
            nan_val[()],
            nan_val[()],
            nan_val[()],
            nan_val[()],
        )

    corr_x = cov_xt / denom_x
    corr_y = cov_yt / denom_y

    slope_x = cov_xt / cov_tt
    slope_y = cov_yt / cov_tt

    x_traj = mean_x + slope_x * (time_coords - mean_t)
    y_traj = mean_y + slope_y * (time_coords - mean_t)

    spatiotemporal_corr = np.sqrt((corr_x**2 + corr_y**2) / 2) * np.sign(
        corr_x + corr_y
    )

    return (
        spatiotemporal_corr,  # float64
        x_traj.astype(dtype),  # Ensure consistent dtype
        y_traj.astype(dtype),
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
