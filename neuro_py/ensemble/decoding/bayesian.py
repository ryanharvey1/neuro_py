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

@njit(parallel=True, fastmath=True, cache=True)
def decode_with_prior_fallback(
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
        if not np.any(ct[i, :]):
            p[i, :] = occupancy_flat / occupancy_flat.sum()
            continue
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

# @njit(parallel=True, fastmath=True, cache=True)
# def decode_with_prior_fallback(
#     ct: np.ndarray,
#     tc: np.ndarray,
#     occupancy: np.ndarray,
#     bin_size_s: float,
#     uniform_prior: bool = False,
# ) -> np.ndarray:
#     # Input validation
#     assert ct.ndim == 2, "ct must be 2D array (n_bins, n_cells)"
#     assert tc.ndim >= 2, "tc must be at least 2D array (n_xbins, ..., n_cells)"
#     assert occupancy.ndim == tc.ndim - 1
#     assert ct.shape[1] == tc.shape[-1]

#     # Flatten spatial dimensions
#     n_cells = tc.shape[-1]
#     spatial_shape = tc.shape[:-1]
#     n_spatial_bins = np.prod(np.array(spatial_shape))  # Faster than manual product
    
#     tc_flat = tc.reshape(n_spatial_bins, n_cells)
#     occupancy_flat = occupancy.ravel()  # Slightly faster than flatten()

#     if uniform_prior:
#         occupancy_flat = np.ones_like(occupancy_flat)

#     # Precompute terms
#     occupancy_sum = occupancy_flat.sum()
#     inv_occupancy_sum = 1.0 / occupancy_sum
#     uniform_prob = occupancy_flat * inv_occupancy_sum
#     log_tc_flat = np.log(tc_flat + 1e-10)
#     log_p1 = -tc_flat.sum(axis=1) * bin_size_s
#     log_p2 = np.log(occupancy_flat * inv_occupancy_sum)

#     # Initialize output
#     n_bins = ct.shape[0]
#     p = np.zeros((n_bins, n_spatial_bins))

#     # Main computation loop - optimized
#     for i in prange(n_bins):
#         current_ct = ct[i, :]
        
#         if not np.any(current_ct):
#             # Manual broadcasting without tile
#             p[i, :] = uniform_prob
#             continue
        
#         # Optimized dot product
#         log_likelihood = log_p1 + log_p2
#         for j in range(n_cells):
#             ct_val = current_ct[j]
#             if ct_val > 0:  # Skip zero counts
#                 log_likelihood += log_tc_flat[:, j] * ct_val
        
#         # Numerical stability
#         max_ll = log_likelihood.max()
#         p_exp = np.exp(log_likelihood - max_ll)
#         p_sum = p_exp.sum()
        
#         if p_sum > 0:
#             p[i, :] = p_exp / p_sum
#         else:
#             p[i, :] = uniform_prob

#     return p.reshape((n_bins,) + spatial_shape)

# @njit(parallel=True, fastmath=True, cache=True, boundscheck=False)
# def decode_with_prior_fallback(
#     ct: np.ndarray,
#     tc: np.ndarray,
#     occupancy: np.ndarray,
#     bin_size_s: float,
#     uniform_prior: bool = False,
# ) -> np.ndarray:
#     # Input dimensions
#     n_bins, n_cells = ct.shape
#     spatial_shape = tc.shape[:-1]
#     n_spatial_bins = np.prod(np.array(spatial_shape))
    
#     # Reshape arrays
#     tc_flat = tc.reshape(n_spatial_bins, n_cells)
#     occupancy_flat = occupancy.ravel()
    
#     # Precompute constants
#     if uniform_prior:
#         log_p2 = np.zeros(n_spatial_bins)
#         uniform_val_scalar = 1.0 / n_spatial_bins
#         uniform_val_array = np.empty(0)  # Dummy array
#     else:
#         inv_occupancy_sum = 1.0 / occupancy_flat.sum()
#         log_p2 = np.log(occupancy_flat * inv_occupancy_sum)
#         uniform_val_array = occupancy_flat * inv_occupancy_sum
#         uniform_val_scalar = 0.0  # Dummy scalar
    
#     log_tc_flat = np.log(tc_flat + 1e-10)
#     log_p1 = (-tc_flat.sum(axis=1) * bin_size_s)
    
#     # Output array
#     p = np.empty((n_bins, n_spatial_bins))
    
#     # Main computation loop
#     for i in prange(n_bins):
#         current_ct = ct[i, :]
#         any_nonzero = False
        
#         # Check for any spikes
#         for j in range(n_cells):
#             if current_ct[j] > 0:
#                 any_nonzero = True
#                 break
        
#         if not any_nonzero:
#             # Handle zero-count case
#             if uniform_prior:
#                 for k in range(n_spatial_bins):
#                     p[i, k] = uniform_val_scalar
#             else:
#                 for k in range(n_spatial_bins):
#                     p[i, k] = uniform_val_array[k]
#             continue
        
#         # Compute log-likelihood
#         log_likelihood = log_p1 + log_p2
#         for j in range(n_cells):
#             ct_val = current_ct[j]
#             if ct_val > 0:
#                 for k in range(n_spatial_bins):
#                     log_likelihood[k] += log_tc_flat[k, j] * ct_val
        
#         # Find max for numerical stability
#         max_ll = log_likelihood[0]
#         for k in range(1, n_spatial_bins):
#             if log_likelihood[k] > max_ll:
#                 max_ll = log_likelihood[k]
        
#         # Compute exp and sum
#         p_sum = 0.0
#         for k in range(n_spatial_bins):
#             p[i, k] = np.exp(log_likelihood[k] - max_ll)
#             p_sum += p[i, k]
        
#         # Normalize
#         inv_p_sum = 1.0 / p_sum
#         for k in range(n_spatial_bins):
#             p[i, k] *= inv_p_sum
    
#     return p.reshape((n_bins,) + spatial_shape)