import numpy as np


def proximity(pv1: np.ndarray, pv2: np.ndarray) -> np.ndarray:
    """
    Proximity between two firing rate vector trajectories.

    Parameters
    ----------
    pv1 : numpy.ndarray
        Firing rate vector trajectory in one context.
        Shape: (num_bins, num_neurons)

    pv2 : numpy.ndarray
        Firing rate vector trajectory in another context.
        Shape: (num_bins, num_neurons)

    Returns
    -------
    numpy.ndarray
        Proximity between the two contexts.

    References
    ----------
    .. [1] Guidera, J. A., Gramling, D. P., Comrie, A. E., Joshi, A.,
        Denovellis, E. L., Lee, K. H., Zhou, J., Thompson, P., Hernandez, J.,
        Yorita, A., Haque, R., Kirst, C., & Frank, L. M. (2024). Regional
        specialization manifests in the reliability of neural population codes.
        bioRxiv : the preprint server for biology, 2024.01.25.576941.
        https://doi.org/10.1101/2024.01.25.576941
    """
    # Calculate the norms
    norm_diff = np.linalg.norm(pv1 - pv2, axis=1)

    norm_diff_mean = np.apply_along_axis(
        lambda e: np.mean(np.linalg.norm(e - pv2, axis=1)), arr=pv1, axis=1
    )

    # Calculate proximity
    prox = 1 - (norm_diff / norm_diff_mean)

    return prox
