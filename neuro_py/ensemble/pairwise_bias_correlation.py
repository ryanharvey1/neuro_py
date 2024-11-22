import nelpy as nel
import numpy as np
from joblib import Parallel, delayed
from scipy.spatial.distance import cosine

from neuro_py.io import loading
from neuro_py.process import intervals
from neuro_py.session.locate_epochs import (
    compress_repeated_epochs,
)
from neuro_py.spikes import spike_tools


def compute_bias_matrix_optimized(spike_times, neuron_ids, total_neurons):
    """
    Optimized computation of the bias matrix B_k for a given sequence of spikes using vectorized operations.

    Parameters:
    - spike_times: list or array of spike times for the sequence.
    - neuron_ids: list or array of neuron identifiers corresponding to spike_times.
    - total_neurons: total number of neurons being considered.

    Returns:
    - bias_matrix: A matrix of size (total_neurons, total_neurons) representing the bias.
    """
    bias_matrix = np.zeros((total_neurons, total_neurons))

    # for i in range(total_neurons):
    #     for j in range(total_neurons):
    #         if i != j:
    #             # Boolean masks for spike times of neurons i and j
    #             mask_i = neuron_ids == i
    #             mask_j = neuron_ids == j
    #             spikes_i = spike_times[mask_i]
    #             spikes_j = spike_times[mask_j]

    #             if spikes_i.size > 0 and spikes_j.size > 0:
    #                 # Count how many times neuron i spikes before neuron j using broadcasting
    #                 count_ij = np.sum(spikes_i[:, np.newaxis] < spikes_j)
    #                 bias_matrix[i, j] = count_ij / (spikes_i.size * spikes_j.size)
    #             else:
    #                 bias_matrix[i, j] = 0.5  # Neutral bias if no spikes

    # return bias_matrix


    # Create boolean masks for all neurons in advance
    masks = [neuron_ids == i for i in range(total_neurons)]

    for i in range(total_neurons):
        spikes_i = spike_times[masks[i]]
        size_i = spikes_i.size
        for j in range(total_neurons):
            if i != j:
                spikes_j = spike_times[masks[j]]
                size_j = spikes_j.size

                if size_i > 0 and size_j > 0:
                    # Count how many times neuron i spikes before neuron j using broadcasting
                    count_ij = np.sum(spikes_i[:, np.newaxis] < spikes_j)
                    bias_matrix[i, j] = count_ij / (size_i * size_j)
                else:
                    bias_matrix[i, j] = 0.5  # Neutral bias if no spikes

    return bias_matrix



def normalize_bias_matrix(bias_matrix):
    """
    Normalize the bias matrix values to fall between -1 and 1.

    Parameters:
    - bias_matrix: A bias matrix of shape (n, n).

    Returns:
    - normalized_matrix: Normalized matrix with values between -1 and 1.
    """
    return 2 * bias_matrix - 1


def compute_cosine_similarity(matrix1, matrix2):
    """
    Computes the cosine similarity between two flattened bias matrices.

    Parameters:
    - matrix1: A normalized bias matrix.
    - matrix2: Another normalized bias matrix.

    Returns:
    - cosine_similarity: The cosine similarity between the two matrices.
    """
    # Flatten matrices
    vec1 = matrix1.flatten()
    vec2 = matrix2.flatten()

    # Compute cosine similarity
    return 1 - cosine(vec1, vec2)


def observed_and_shuffled_correlation(
    post_spikes,
    post_neurons,
    total_neurons,
    task_normalized,
    post_intervals,
    interval_i,
    num_shuffles=100,
):

    # for i_interval in range(post_intervals.shape[0]):
    idx = (post_spikes > post_intervals[interval_i][0]) & (
        post_spikes < post_intervals[interval_i][1]
    )

    post_bias_matrix = compute_bias_matrix_optimized(
        post_spikes[idx], post_neurons[idx], total_neurons
    )
    post_normalized = normalize_bias_matrix(post_bias_matrix)

    # Compute cosine similarity between task and post-task bias matrices
    observed_correlation = compute_cosine_similarity(task_normalized, post_normalized)

    # Shuffle post-task spikes and compute bias matrix
    shuffled_correlation = [
        compute_cosine_similarity(
            task_normalized,
            normalize_bias_matrix(
                compute_bias_matrix_optimized(
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

    Parameters:
    - task_spikes: list or array of spike times during the task.
    - task_neurons: list or array of neuron identifiers for task spikes.
    - post_spikes: list or array of spike times during post-task (e.g., sleep).
    - post_neurons: list or array of neuron identifiers for post-task spikes.
    - total_neurons: total number of neurons being considered.
    - post_intervals: list or array of intervals for post-task epochs.
    - num_shuffles: Number of shuffles to compute significance.
    - n_jobs: Number of parallel jobs for shuffling. Default is -1 (use all available cores).

    Returns:
    - z_score: The z-score of the observed correlation compared to the shuffled distribution.
    """
    # set random seed for reproducibility
    np.random.seed(0)

    # Compute bias matrices for task epochs
    task_bias_matrix = compute_bias_matrix_optimized(
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
    basepath = r"U:\data\HMC\HMC1\day8"
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
