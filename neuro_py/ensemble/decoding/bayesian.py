import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True)
def decode(
    ct: np.ndarray,
    tc: np.ndarray,
    occupancy: np.ndarray,
    bin_size_s: float,
    uniform_prior: bool = False,
) -> np.ndarray:
    """
    Decode position from spike counts in an N-dimensional spatial environment

    Parameters
    ----------
    ct : ndarray
        2D array, spike counts matrix with shape (n_bins, n_cells)
    tc : ndarray
        ND array, ratemap matrix with shape (n_xbins, n_ybins, ..., n_cells)
    occupancy : ndarray
        (N-1)D array, occupancy matrix with shape (n_xbins, n_ybins, ...)
    bin_size_s : float
        float, width of each time bin in seconds
    uniform_prior : bool, optional
        bool, whether to use uniform prior, by default False

    Returns
    ----------
    p : ndarray
        (N+1)D array, decoded position probabilities matrix with shape (n_bins, n_xbins, n_ybins, ...)

    Examples
    ----------
    # 1D example
    >>> ct = np.random.rand(10, 5)
    >>> tc = np.random.rand(3, 5)
    >>> occupancy = np.random.rand(3)
    >>> bin_size_s = 0.1
    >>> p = decode(ct, tc, occupancy, bin_size_s)

    # 2D example
    >>> ct = np.random.rand(10, 5)
    >>> tc = np.random.rand(3, 3, 5)
    >>> occupancy = np.random.rand(3, 3)
    >>> bin_size_s = 0.1
    >>> p = decode(ct, tc, occupancy, bin_size_s)

    # 3D example
    >>> ct = np.random.rand(10, 5)
    >>> tc = np.random.rand(3, 3, 3, 5)
    >>> occupancy = np.random.rand(3, 3, 3)
    >>> bin_size_s = 0.1
    >>> p = decode(ct, tc, occupancy, bin_size_s)
    """

    # Ensure input arrays are contiguous for vectorization
    ct = np.ascontiguousarray(ct)
    tc = np.ascontiguousarray(tc)
    occupancy = np.ascontiguousarray(occupancy)

    # Validate input shapes
    assert ct.ndim == 2, "ct must be a 2D array with shape (n_bins, n_cells)"
    assert tc.ndim >= 2, (
        "tc must be at least a 2D array with shape (n_xbins, ..., n_cells)"
    )
    assert occupancy.ndim == tc.ndim - 1, (
        "occupancy must have one fewer dimension than tc"
    )
    assert ct.shape[1] == tc.shape[-1], "Number of cells in ct and tc must match"

    # Flatten spatial dimensions
    n_cells = tc.shape[-1]
    spatial_shape = tc.shape[:-1]  # Shape of spatial dimensions

    # Calculate the total number of spatial bins
    n_spatial_bins = 1
    for dim in spatial_shape:
        n_spatial_bins *= dim

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

    # Reshape the probabilities to the original spatial dimensions
    p = p.reshape((n_bins,) + spatial_shape)

    return p
