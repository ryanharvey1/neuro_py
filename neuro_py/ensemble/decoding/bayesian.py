import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True)
def decode_2d(
    ct: np.ndarray,
    tc: np.ndarray,
    occupancy: np.ndarray,
    bin_size_s: float,
    uniform_prior: bool = False,
) -> np.ndarray:
    """
    Decode position from spike counts in a 2D spatial environment

    Parameters:
    - ct: 2D array, spike counts matrix with shape (n_bins, n_cells)
    - tc: 3D array, ratemap matrix with shape (n_xbins, n_ybins, n_cells)
    - occupancy: 2D array, occupancy matrix with shape (n_xbins, n_ybins)
    - bin_size_s: float, width of each time bin in seconds

    Returns:
    - p: 3D array, decoded position probabilities matrix with shape (n_bins, n_xbins, n_ybins)
    """

    # Ensure input arrays are contiguous for vectorization
    ct = np.ascontiguousarray(ct)
    tc = np.ascontiguousarray(tc)
    occupancy = np.ascontiguousarray(occupancy)

    # Flatten spatial dimensions
    n_xbins, n_ybins, n_cells = tc.shape
    n_spatial_bins = n_xbins * n_ybins
    tc_flat = tc.reshape(n_spatial_bins, n_cells)
    occupancy_flat = occupancy.flatten()

    if uniform_prior:
        # Use uniform prior
        occupancy_flat = np.ones_like(occupancy_flat)

    # Precompute log values
    log_tc_flat = np.log(tc_flat + 1e-10)  # add small value to avoid log(0)
    log_p1 = -tc_flat.sum(axis=1) * bin_size_s
    log_p2 = np.log(occupancy_flat / occupancy_flat.sum())

    # Initialize the probability matrix
    n_bins = ct.shape[0]
    p = np.zeros((n_bins, n_spatial_bins))

    # Vectorized calculation of log probabilities
    for i in prange(n_bins):  # prange for parallel loop
        log_likelihood = log_p1 + log_p2 + np.sum(log_tc_flat * ct[i, :], axis=1)
        p[i, :] = np.exp(
            log_likelihood - np.max(log_likelihood)
        )  # Subtract max for numerical stability

    # Normalize the probabilities along the spatial axis
    p_sum = p.sum(axis=1)  # Sum over spatial bins
    p = p / p_sum.reshape(-1, 1)  # Reshape p_sum to (n_bins, 1) for broadcasting

    # Reshape the probabilities to the 2D spatial dimensions
    p = p.reshape(n_bins, n_xbins, n_ybins)

    return p
