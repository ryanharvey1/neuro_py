import multiprocessing
from typing import List, Optional, Tuple, Union

import numpy as np
import scipy.stats as stats

from joblib import Parallel, delayed
from nelpy.analysis import replay
from nelpy.decoding import decode1D as decode
from nelpy.core import BinnedSpikeTrainArray
from nelpy import TuningCurve1D
from numba import jit

from neuro_py.ensemble.pairwise_bias_correlation import (
    bias_matrix_fast,
    cosine_similarity_matrices,
    normalize_bias_matrix,
)
from neuro_py.process.peri_event import crossCorr


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


# class PairwiseBiasAnalysis:
#     def __init__(self, total_neurons, num_shuffles=100, n_jobs=-1):
#         self.total_neurons = total_neurons
#         self.num_shuffles = num_shuffles
#         self.n_jobs = n_jobs

#     @staticmethod
#     def compute_bias_matrix_optimized(spike_times, neuron_ids, total_neurons):
#         """
#         Optimized computation of the bias matrix B_k for a given sequence of spikes using vectorized operations.

#         Parameters:
#         - spike_times: list or array of spike times for the sequence.
#         - neuron_ids: list or array of neuron identifiers corresponding to spike_times.
#         - total_neurons: total number of neurons being considered.

#         Returns:
#         - bias_matrix: A matrix of size (total_neurons, total_neurons) representing the bias.
#         """
#         bias_matrix = np.zeros((total_neurons, total_neurons))

#         # Create boolean masks for all neurons in advance
#         masks = [neuron_ids == i for i in range(total_neurons)]

#         for i in range(total_neurons):
#             spikes_i = spike_times[masks[i]]
#             size_i = spikes_i.size
#             for j in range(total_neurons):
#                 if i != j:
#                     spikes_j = spike_times[masks[j]]
#                     size_j = spikes_j.size

#                     if size_i > 0 and size_j > 0:
#                         # Count how many times neuron i spikes before neuron j using broadcasting
#                         count_ij = np.sum(spikes_i[:, np.newaxis] < spikes_j)
#                         bias_matrix[i, j] = count_ij / (size_i * size_j)
#                     else:
#                         bias_matrix[i, j] = 0.5  # Neutral bias if no spikes

#         return bias_matrix

#     @staticmethod
#     def normalize_bias_matrix(bias_matrix):
#         """
#         Normalize the bias matrix values to fall between -1 and 1.

#         Parameters:
#         - bias_matrix: A bias matrix of shape (n, n).

#         Returns:
#         - normalized_matrix: Normalized matrix with values between -1 and 1.
#         """
#         return 2 * bias_matrix - 1

#     @staticmethod
#     def compute_cosine_similarity(matrix1, matrix2):
#         """
#         Computes the cosine similarity between two flattened bias matrices.

#         Parameters:
#         - matrix1: A normalized bias matrix.
#         - matrix2: Another normalized bias matrix.

#         Returns:
#         - cosine_similarity: The cosine similarity between the two matrices.
#         """
#         vec1 = matrix1.flatten()
#         vec2 = matrix2.flatten()
#         return 1 - cosine(vec1, vec2)

#     def observed_and_shuffled_correlation(
#         self, post_spikes, post_neurons, task_normalized, post_intervals, interval_i
#     ):
#         idx = (post_spikes > post_intervals[interval_i][0]) & (
#             post_spikes < post_intervals[interval_i][1]
#         )

#         post_bias_matrix = self.compute_bias_matrix_optimized(
#             post_spikes[idx], post_neurons[idx], self.total_neurons
#         )
#         post_normalized = self.normalize_bias_matrix(post_bias_matrix)

#         # Compute cosine similarity between task and post-task bias matrices
#         observed_correlation = self.compute_cosine_similarity(
#             task_normalized, post_normalized
#         )

#         # Shuffle post-task spikes and compute bias matrix
#         shuffled_correlation = [
#             self.compute_cosine_similarity(
#                 task_normalized,
#                 self.normalize_bias_matrix(
#                     self.compute_bias_matrix_optimized(
#                         post_spikes[idx],
#                         np.random.permutation(post_neurons[idx]),
#                         self.total_neurons,
#                     )
#                 ),
#             )
#             for _ in range(self.num_shuffles)
#         ]

#         return observed_correlation, shuffled_correlation

#     def shuffled_significance(
#         self, task_spikes, task_neurons, post_spikes, post_neurons, post_intervals
#     ):
#         """
#         Computes the significance of the task-post correlation by comparing against shuffled distributions.

#         Parameters:
#         - task_spikes: list or array of spike times during the task.
#         - task_neurons: list or array of neuron identifiers for task spikes.
#         - post_spikes: list or array of spike times during post-task (e.g., sleep).
#         - post_neurons: list or array of neuron identifiers for post-task spikes.
#         - post_intervals: list or array of intervals for post-task epochs.

#         Returns:
#         - z_score: The z-score of the observed correlation compared to the shuffled distribution.
#         - p_value: p-value for significance test.
#         """
#         # Compute bias matrices for task epochs
#         task_bias_matrix = self.compute_bias_matrix_optimized(
#             task_spikes, task_neurons, self.total_neurons
#         )
#         # Normalize the bias matrices
#         task_normalized = self.normalize_bias_matrix(task_bias_matrix)

#         # Get shuffled and observed correlations using parallel processing
#         observed_correlation, shuffled_correlations = zip(
#             *Parallel(n_jobs=self.n_jobs)(
#                 delayed(self.observed_and_shuffled_correlation)(
#                     post_spikes,
#                     post_neurons,
#                     task_normalized,
#                     post_intervals,
#                     interval_i,
#                 )
#                 for interval_i in range(post_intervals.shape[0])
#             )
#         )
#         observed_correlation, shuffled_correlations = (
#             np.array(observed_correlation),
#             np.array(shuffled_correlations),
#         )

#         # Compute z-score
#         shuffled_mean = np.mean(shuffled_correlations, axis=1)
#         shuffled_std = np.std(shuffled_correlations, axis=1)
#         z_score = (observed_correlation - shuffled_mean) / shuffled_std

#         # Significance test between the observed correlation and the shuffled distribution
#         p_value = (np.sum(shuffled_correlations.T > observed_correlation, axis=0) + 1) / (
#             self.num_shuffles + 1
#         )

#         return z_score, p_value

# @jit(nopython=True)
# def compute_bias_matrix_optimized_(spike_times, neuron_ids, total_neurons):
#     """
#     Optimized computation of the bias matrix B_k for a given sequence of spikes using vectorized operations.

#     Parameters:
#     - spike_times: list or array of spike times for the sequence.
#     - neuron_ids: list or array of neuron identifiers corresponding to spike_times.
#     - total_neurons: total number of neurons being considered.

#     Returns:
#     - bias_matrix: A matrix of size (total_neurons, total_neurons) representing the bias.
#     """

#     # Create an empty bias matrix
#     bias_matrix = np.full((total_neurons, total_neurons), 0.5)

#     # Create boolean masks for all neurons in advance
#     masks = [neuron_ids == i for i in range(total_neurons)]

#     # Iterate over each pair of neurons
#     for i in range(total_neurons):
#         spikes_i = spike_times[masks[i]]
#         size_i = spikes_i.size

#         if size_i == 0:
#             continue  # Skip if neuron i has no spikes

#         for j in range(total_neurons):
#             if i == j:
#                 continue  # Skip self-correlation

#             spikes_j = spike_times[masks[j]]
#             size_j = spikes_j.size

#             if size_j == 0:
#                 continue  # Skip if neuron j has no spikes

#             # Count how many times neuron i spikes before neuron j
#             count_ij = np.sum(spikes_i[:, np.newaxis] < spikes_j)
#             bias_matrix[i, j] = count_ij / (size_i * size_j)

#     return bias_matrix
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
    task_normalized : np.ndarray, or None
        Normalized bias matrix for the task data.
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
    def __init__(self, num_shuffles: int = 300, n_jobs: int = 10):
        self.num_shuffles = num_shuffles
        self.n_jobs = n_jobs
        self.total_neurons = None
        self.task_normalized = None
        self.observed_correlation_ = None
        self.shuffled_correlations_ = None
        self.z_score_ = None
        self.p_value_ = None


    @staticmethod
    def bias_matrix(
        spike_times: np.ndarray, 
        neuron_ids: np.ndarray, 
        total_neurons: int
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

        Returns
        -------
        np.ndarray
            A matrix of size (total_neurons, total_neurons) representing the bias.
        """
        return bias_matrix_fast(spike_times, neuron_ids, total_neurons)


    @staticmethod
    def normalize_bias_matrix(bias_matrix: np.ndarray) -> np.ndarray:
        """
        Normalize the bias matrix values to fall between -1 and 1.

        Parameters
        ----------
        bias_matrix : np.ndarray
            A bias matrix of shape (n, n).

        Returns
        -------
        np.ndarray
            Normalized matrix with values between -1 and 1.
        """
        return normalize_bias_matrix(bias_matrix)

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
        task_normalized: np.ndarray,
        post_intervals: np.ndarray,
        interval_i: int
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

        idx = (post_spikes > post_intervals[interval_i][0]) & (
            post_spikes < post_intervals[interval_i][1]
        )

        post_bias_matrix = self.bias_matrix(
            post_spikes[idx], post_neurons[idx], self.total_neurons
        )
        post_normalized = self.normalize_bias_matrix(post_bias_matrix)

        observed_correlation = self.cosine_similarity_matrices(
            task_normalized, post_normalized
        )

        shuffled_correlation = [
            self.cosine_similarity_matrices(
                task_normalized,
                self.normalize_bias_matrix(
                    self.bias_matrix(
                        post_spikes[idx],
                        np.random.permutation(post_neurons[idx]),
                        self.total_neurons,
                    )
                ),
            )
            for _ in range(self.num_shuffles)
        ]

        return observed_correlation, shuffled_correlation

    def fit(
        self,
        task_spikes: np.ndarray,
        task_neurons: np.ndarray,
        task_intervals: np.ndarray = None
    ) -> 'PairwiseBias':
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
            task_bias_matrix = self.bias_matrix(
                task_spikes, task_neurons, self.total_neurons
            )
            self.task_normalized = self.normalize_bias_matrix(task_bias_matrix)
        else:
            n_intervals = task_intervals.shape[0]
            # Compute bias matrices for each task interval
            task_bias_matrices = [
                self.bias_matrix(
                    task_spikes[
                        (task_spikes > task_intervals[i][0])
                        & (task_spikes <= task_intervals[i][1])
                    ],
                    task_neurons[
                        (task_spikes > task_intervals[i][0])
                        & (task_spikes <= task_intervals[i][1])
                    ],
                    self.total_neurons,
                )
                for i in range(n_intervals)
            ]
            # Normalize each bias matrix
            task_normalized_matrices = [
                self.normalize_bias_matrix(task_bias_matrices[i])
                for i in range(n_intervals)
            ]
            # Average the normalized bias matrices
            self.task_normalized = np.mean(task_normalized_matrices, axis=0)
        return self

    def transform(
        self,
        post_spikes: np.ndarray,
        post_neurons: np.ndarray,
        post_intervals: np.ndarray,
        allow_reverse_replay: bool = False
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

        observed_correlation, shuffled_correlations = zip(
            *Parallel(n_jobs=self.n_jobs)(
                delayed(self.observed_and_shuffled_correlation)(
                    post_spikes,
                    post_neurons,
                    self.task_normalized,
                    post_intervals,
                    interval_i,
                )
                for interval_i in range(post_intervals.shape[0])
            )
        )
        # non parallel version
        # observed_correlation, shuffled_correlations = zip(
        #     *[
        #         self.observed_and_shuffled_correlation(
        #             post_spikes,
        #             post_neurons,
        #             self.task_normalized,
        #             post_intervals,
        #             interval_i,
        #         )
        #         for interval_i in range(post_intervals.shape[0])
        #     ]
        # )

        self.observed_correlation_ = np.array(observed_correlation)  # Shape: (n_intervals,)
        self.shuffled_correlations_ = np.array(shuffled_correlations)  # Shape: (n_intervals, n_shuffles)

        shuffled_mean = np.mean(self.shuffled_correlations_, axis=1)
        shuffled_std = np.std(self.shuffled_correlations_, axis=1)
        self.z_score_ = (
            self.observed_correlation_ - shuffled_mean
        ) / shuffled_std

        observed_correlation = self.observed_correlation_
        shuffled_correlations = self.shuffled_correlations_
        if allow_reverse_replay:
            observed_correlation = np.abs(observed_correlation)
            shuffled_correlations = np.abs(shuffled_correlations)

        self.p_value_ = (
            np.sum(
                shuffled_correlations.T
                > observed_correlation,
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
        post_intervals: np.ndarray
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

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            z_score: The z-score of the observed correlation compared to the shuffled distribution.
            p_value: p-value for significance test.
            observed_correlation_: The observed correlation for each interval.
        """
        self.fit(task_spikes, task_neurons, task_intervals)
        return self.transform(post_spikes, post_neurons, post_intervals)
