from joblib import Parallel, delayed
from typing import List, Tuple

import nelpy as nel
import numpy as np
import sklearn
import sklearn.metrics

from numba import njit
from neuro_py.io import loading
from neuro_py.process import intervals
from neuro_py.session.locate_epochs import (
    compress_repeated_epochs,
)
from neuro_py.spikes import spike_tools


def bias_matrix(
    spike_times: np.ndarray,
    neuron_ids: np.ndarray,
    total_neurons: int,
    fillneutral: float = 0.5
) -> np.ndarray:
    r"""
    Compute the bias matrix for a given sequence of spikes.

    Parameters
    ----------
    spike_times : numpy.ndarray
        Spike times for the sequence, assumed to be sorted.
    neuron_ids : numpy.ndarray
        Neuron identifiers corresponding to `spike_times`.
        Values should be integers between 0 and `total_neurons - 1`.
    total_neurons : int
        Total number of neurons being considered.
    fillneutral : float, optional
        Value to fill for neutral bias, by default 0.5

    Returns
    -------
    numpy.ndarray
        A bias matrix of size `(total_neurons, total_neurons)` where
        each entry represents the bias between neuron pairs.

    Notes
    -----
    The bias \( B_{ij} \) for neurons \( i \) and \( j \) is computed as:
    \[
    B_{ij} = \frac{nspikes_{ij}}{nspikes_i \cdot nspikes_j}
    \]
    where \( nspikes_{ij} \) is the count of spikes from neuron \( i \) occurring 
    before spikes from neuron \( j \). If there are no spikes for either neuron,
    the bias is set to 0.5 (neutral bias).
    """
    bias = np.empty((total_neurons, total_neurons))
    np.fill_diagonal(bias, fillneutral)

    # Create boolean masks for all neurons in advance
    masks = [neuron_ids == i for i in range(total_neurons)]

    for i in range(total_neurons):
        spikes_i = spike_times[masks[i]]  # timestamps for neuron i
        nspikes_i = spikes_i.size
        for j in range(total_neurons):
            if i != j:
                spikes_j = spike_times[masks[j]]
                nspikes_j = spikes_j.size

                if nspikes_i > 0 and nspikes_j > 0:
                    # Count how many times neuron i spikes before each neuron j spike
                    nspikes_ij = np.searchsorted(
                        spikes_i, spikes_j, side='right').sum()
                    bias[i, j] = nspikes_ij / (nspikes_i * nspikes_j)
                else:
                    bias[i, j] = fillneutral  # Neutral bias if no spikes

    return bias


def bias_matrix_fast(
    spike_times: np.ndarray,
    neuron_ids: np.ndarray,
    total_neurons: int,
    fillneutral: float = 0.5,
    return_counts: bool = False
) -> np.ndarray:
    r"""
    Compute the bias matrix for a given sequence of spikes.

    Parameters
    ----------
    spike_times : numpy.ndarray
        Spike times for the sequence, assumed to be sorted.
    neuron_ids : numpy.ndarray
        Neuron identifiers corresponding to `spike_times`.
        Values should be integers between 0 and `total_neurons - 1`.
    total_neurons : int
        Total number of neurons being considered.
    fillneutral : float, optional
        Value to fill for neutral bias, by default 0.5

    Returns
    -------
    bias : numpy.ndarray
        A bias matrix of size `(total_neurons, total_neurons)` where
        each entry represents the bias between neuron pairs.
    ibeforej : numpy.ndarray, optional
        A matrix of size `(total_neurons, total_neurons)` where
        each entry represents the count of spikes from neuron \( i \) occurring
        before spikes from neuron \( j \).
    prod_nspikes_ij : numpy.ndarray, optional
        A matrix of size `(total_neurons, total_neurons)` where
        each entry represents the product of spikes from neurons \( i \) and
        \( j \).

    Notes
    -----
    Refer to the `bias_matrix` function for better code interpretability.

    The bias \( B_{ij} \) for neurons \( i \) and \( j \) is computed as:
    \[
    B_{ij} = \frac{nspikes_{ij}}{nspikes_i \cdot nspikes_j}
    \]
    where \( nspikes_{ij} \) is the count of spikes from neuron \( i \) occurring 
    before spikes from neuron \( j \). If there are no spikes for either neuron,
    the bias is set to 0.5 (neutral bias).

    References
    ----------
    .. [1] Roth, Z. (2016). Analysis of neuronal sequences using pairwise
        biases. arXiv, 65-67 (2016). https://arxiv.org/abs/1603.02916
    """
    ibeforej = np.zeros((total_neurons, total_neurons))  # rows: i, cols: j
    prod_nspikes_ij = np.zeros((total_neurons, total_neurons))
    bias = np.empty((total_neurons, total_neurons))

    # Create boolean masks for all neurons in advance
    masks = [neuron_ids == i for i in range(total_neurons)]

    for i in range(total_neurons):
        spikes_i = spike_times[masks[i]]  # timestamps for neuron i
        nspikes_i = spikes_i.size
        for j in range(i + 1, total_neurons):
            spikes_j = spike_times[masks[j]]
            nspikes_j = spikes_j.size

            if nspikes_i > 0 and nspikes_j > 0:
                # Count how many times neuron i spikes before each neuron j spike
                nspikes_ij = np.searchsorted(
                    spikes_i, spikes_j, side='right').sum()
                ibeforej[i, j] = nspikes_ij
                prod_nspikes_ij[i, j] = nspikes_i * nspikes_j

    jbeforei = prod_nspikes_ij - ibeforej
    prod_nspikes_ij = prod_nspikes_ij + prod_nspikes_ij.T  # symmetrize
    ibeforej = ibeforej + jbeforei.T
    np.divide(ibeforej, prod_nspikes_ij, out=bias, where=prod_nspikes_ij != 0)
    # set remaining values to fillneutral
    bias[prod_nspikes_ij == 0] = fillneutral

    if return_counts:
        return bias, ibeforej, prod_nspikes_ij

    return bias


@njit
def bias_matrix_njit(
    spike_times: np.ndarray,
    neuron_ids: np.ndarray,
    total_neurons: int,
    fillneutral: float = 0.5
) -> np.ndarray:
    r"""
    Compute the bias matrix for a given sequence of spikes.

    Parameters
    ----------
    spike_times : numpy.ndarray
        Spike times for the sequence, assumed to be sorted.
    neuron_ids : numpy.ndarray
        Neuron identifiers corresponding to `spike_times`.
        Values should be integers between 0 and `total_neurons - 1`.
    total_neurons : int
        Total number of neurons being considered.
    fillneutral : float, optional
        Value to fill for neutral bias, by default 0.5

    Returns
    -------
    numpy.ndarray
        A bias matrix of size `(total_neurons, total_neurons)` where
        each entry represents the bias between neuron pairs.

    Notes
    -----
    Refer to the `bias_matrix` function for better code interpretability.

    The bias \( B_{ij} \) for neurons \( i \) and \( j \) is computed as:
    \[
    B_{ij} = \frac{nspikes_{ij}}{nspikes_i \cdot nspikes_j}
    \]
    where \( nspikes_{ij} \) is the count of spikes from neuron \( i \) occurring
    before spikes from neuron \( j \). If there are no spikes for either neuron,
    the bias is set to 0.5 (neutral bias).

    References
    ----------
    .. [1] Roth, Z. (2016). Analysis of neuronal sequences using pairwise
        biases. arXiv, 65-67 (2016). https://arxiv.org/abs/1603.02916
    """
    ibeforej = np.zeros((total_neurons, total_neurons))  # rows: i, cols: j
    prod_nspikes_ij = np.zeros((total_neurons, total_neurons))
    bias = np.empty((total_neurons, total_neurons))

    # Create boolean masks for all neurons in advance
    masks = [neuron_ids == i for i in range(total_neurons)]

    for i in range(total_neurons):
        spikes_i = spike_times[masks[i]]
        nspikes_i = spikes_i.size
        for j in range(i + 1, total_neurons):
            spikes_j = spike_times[masks[j]]
            nspikes_j = spikes_j.size

            if nspikes_i > 0 and nspikes_j > 0:
                nspikes_ij = np.searchsorted(
                    spikes_i, spikes_j, side='right').sum()
                ibeforej[i, j] = nspikes_ij
                prod_nspikes_ij[i, j] = nspikes_i * nspikes_j

    jbeforei = prod_nspikes_ij - ibeforej
    prod_nspikes_ij = prod_nspikes_ij + prod_nspikes_ij.T
    ibeforej = ibeforej + jbeforei.T
    bias = ibeforej / prod_nspikes_ij
    bias = np.where(prod_nspikes_ij == 0, fillneutral, bias)

    return bias


def normalize_bias_matrix(bias: np.ndarray) -> np.ndarray:
    """
    Normalize the bias matrix values to fall between -1 and 1.

    Parameters
    ----------
    bias : numpy.ndarray
        A bias matrix of shape (n_neurons, n_neurons).

    Returns
    -------
    numpy.ndarray
        A normalized matrix with values between -1 and 1.
    """
    return 2 * bias - 1


def cosine_similarity_matrices(
    matrix1: np.ndarray,
    matrix2: np.ndarray
) -> float:
    """
    Compute the cosine similarity between two flattened matrices

    Parameters
    ----------
    matrix1 : numpy.ndarray
        A normalized bias matrix
    matrix2 : numpy.ndarray
        Another normalized bias matrix

    Returns
    -------
    float
        The cosine similarity between the two matrices.
    """
    # Flatten matrices
    x = matrix1.flatten().reshape(1, -1)
    y = matrix2.flatten().reshape(1, -1)

    if np.all(np.isnan(x)) or np.all(np.isnan(y)):
        return np.nan

    # handle nan values
    x = np.nan_to_num(x)
    y = np.nan_to_num(y)

    cossim = sklearn.metrics.pairwise.cosine_similarity(x, y)

    # Compute cosine similarity
    return cossim.item()


def observed_and_shuffled_correlation(
    post_spikes: np.ndarray,
    post_neurons: np.ndarray,
    total_neurons: int,
    task_normalized: np.ndarray,
    post_intervals: np.ndarray,
    interval_i: int,
    num_shuffles: int = 100,
) -> Tuple[float, List[float]]:
    """
    Calculate observed and shuffled correlations between task and post-task neural activity.

    This function computes the correlation between normalized task bias matrix and
    post-task bias matrix, as well as correlations with shuffled post-task data.

    Parameters
    ----------
    post_spikes : np.ndarray
        Array of post-task spike times.
    post_neurons : np.ndarray
        Array of neuron IDs corresponding to post_spikes.
    total_neurons : int
        Total number of neurons in the dataset.
    task_normalized : np.ndarray
        Normalized bias matrix from task period.
    post_intervals : np.ndarray
        Array of post-task intervals, shape (n_intervals, 2).
    interval_i : int
        Index of the current interval to analyze.
    num_shuffles : int, optional
        Number of times to shuffle post-task data for null distribution, by default 100.

    Returns
    -------
    Tuple[float, List[float]]
        A tuple containing:
        - observed_correlation: float
            Cosine similarity between task and post-task bias matrices.
        - shuffled_correlation: List[float]
            List of cosine similarities between task and shuffled post-task bias matrices.
    """
    # for i_interval in range(post_intervals.shape[0]):
    idx = (post_spikes > post_intervals[interval_i][0]) & (
        post_spikes < post_intervals[interval_i][1]
    )

    post_bias_matrix = bias_matrix(
        post_spikes[idx], post_neurons[idx], total_neurons
    )
    post_normalized = normalize_bias_matrix(post_bias_matrix)

    # Compute cosine similarity between task and post-task bias matrices
    observed_correlation = cosine_similarity_matrices(task_normalized, post_normalized)

    # Shuffle post-task spikes and compute bias matrix
    shuffled_correlation = [
        cosine_similarity_matrices(
            task_normalized,
            normalize_bias_matrix(
                bias_matrix(
                    post_spikes[idx],
                    np.random.permutation(post_neurons[idx]),
                    total_neurons,
                )
            ),
        )
        for _ in range(num_shuffles)
    ]

    return observed_correlation, shuffled_correlation


def shuffled_significance(
    task_spikes: np.ndarray,
    task_neurons: np.ndarray,
    post_spikes: np.ndarray,
    post_neurons: np.ndarray,
    total_neurons: int,
    post_intervals: np.ndarray = np.array([[-np.inf, np.inf]]),
    num_shuffles: int = 100,
    n_jobs: int = -1,
):
    """
    Computes the significance of the task-post correlation by comparing against shuffled distributions.

    Parameters
    ----------
    task_spikes : np.ndarray
        Spike timestamps during the task. Shape is (n_spikes_task,)
    task_neurons : np.ndarray
        Neuron identifiers corresponding to each of `task_spikes`. Shape is
        (n_spikes_task,)
    post_spikes : np.ndarray
        Spike timestamps during post-task (e.g., sleep). Shape is
        (n_spikes_post,)
    post_neurons : np.ndarray
        Neuron identifiers corresponding to `post_spikes`. Shape is
        (n_spikes_post,)
    total_neurons : int
        Total number of neurons being considered
    post_intervals : np.ndarray, optional
        Intervals for post-task epochs, with shape (n_intervals, 2).
        Each row defines the start and end of an interval. May correspond to
        specific sleep states. Default is `np.array([[-np.inf, np.inf]])`,
        representing the entire range of post-task epochs
    num_shuffles : int, optional
        Number of shuffles to compute the significance. Default is 100
    n_jobs : int, optional
        Number of parallel jobs to use for shuffling. Default is -1 (use all
        available cores).

    Returns
    -------
    z_score : np.ndarray
        Z-scores of the observed correlations compared to the shuffled distributions. 
        Shape is (n_intervals,).
    p_value : np.ndarray
        P-values indicating the significance of the observed correlation. 
        Shape is (n_intervals,).

    Notes
    -----
    The function uses parallel processing to compute observed and shuffled 
    correlations for each post-task interval. The z-score is calculated as:

        z_score = (observed_correlation - mean(shuffled_correlations)) / std(shuffled_correlations)

    The p-value is computed as the proportion of shuffled correlations greater than 
    the observed correlation, with a small constant added for numerical stability.

    Examples
    --------
    >>> task_spikes = np.array([1.2, 3.4, 5.6])
    >>> task_neurons = np.array([0, 1, 0])
    >>> post_spikes = np.array([2.3, 4.5, 6.7])
    >>> post_neurons = np.array([1, 0, 1])
    >>> total_neurons = 2
    >>> post_intervals = np.array([[0, 10]])
    >>> z_score, p_value = shuffled_significance(task_spikes, task_neurons, post_spikes, post_neurons, total_neurons, post_intervals)
    >>> z_score
    array([1.23])
    >>> p_value
    array([0.04])
    """
    # set random seed for reproducibility
    np.random.seed(0)

    # Compute bias matrices for task epochs
    task_bias_matrix = bias_matrix(
        task_spikes, task_neurons, total_neurons
    )
    # Normalize the bias matrices
    task_normalized = normalize_bias_matrix(task_bias_matrix)

    # Get shuffled and observed correlations using parallel processing
    observed_correlation, shuffled_correlations = zip(
        *Parallel(n_jobs=n_jobs)(
            delayed(observed_and_shuffled_correlation)(
                post_spikes,
                post_neurons,
                total_neurons,
                task_normalized,
                post_intervals,
                interval_i,
                num_shuffles,
            )
            for interval_i in range(post_intervals.shape[0])
        )
    )
    observed_correlation, shuffled_correlations = np.array(
        observed_correlation
    ), np.array(shuffled_correlations)
    # Compute z-score
    shuffled_mean = np.mean(shuffled_correlations, axis=1)
    shuffled_std = np.std(shuffled_correlations, axis=1)
    z_score = (observed_correlation - shuffled_mean) / shuffled_std

    # significance test between the observed correlation and the shuffled distribution
    p_value = (np.sum(shuffled_correlations.T > observed_correlation, axis=0) + 1) / (
        num_shuffles + 1
    )

    return z_score, p_value


if __name__ == "__main__":
    basepath = r"/run/user/1000/gvfs/smb-share:server=ayadatab.local,share=ayadatab1/data/HMC/HMC1/day8"
    epoch_df = loading.load_epoch(basepath)
    # get session bounds to provide support
    session_bounds = nel.EpochArray(
        [epoch_df.startTime.iloc[0], epoch_df.stopTime.iloc[-1]]
    )
    # compress repeated sleep sessions
    epoch_df = compress_repeated_epochs(epoch_df)
    beh_epochs = nel.EpochArray(epoch_df[["startTime", "stopTime"]].values)

    st, cell_metrics = loading.load_spikes(
        basepath, putativeCellType="Pyr", brainRegion="CA1"
    )
    swr = loading.load_ripples_events(basepath, return_epoch_array=True)

    theta = nel.EpochArray(loading.load_SleepState_states(basepath)["THETA"])

    spike_spindices = spike_tools.get_spindices(st.data)

    task_idx = intervals.in_intervals(
        spike_spindices.spike_times, (beh_epochs[1] & theta).data
    )

    post_idx = intervals.in_intervals(spike_spindices.spike_times, beh_epochs[2].data)

    z_score, p_value = shuffled_significance(
        spike_spindices[task_idx]["spike_times"].values,
        spike_spindices[task_idx]["spike_id"].values,
        spike_spindices[post_idx]["spike_times"].values,
        spike_spindices[post_idx]["spike_id"].values,
        st.n_active,
        post_intervals=(beh_epochs[2] & swr)[:10].data,
    )
    print(z_score, p_value)
