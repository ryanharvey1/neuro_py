import numpy as np


def find_terminal_masked_indices(mask, axis):
    """
    Find the first and last indices of non-masked values along an axis.

    Parameters
    ----------
    mask : np.ndarray
        Mask of `arr`.
    axis : int
        Axis along which to find the first and last indices.

    Returns
    -------
    np.ndarray
        First index of non-masked values along `axis`.
    np.ndarray
        Last index of non-masked values along `axis`.
    """
    first_idx = np.argmax(mask, axis=axis)
    reversed_mask = np.flip(mask, axis=axis)
    last_idx = mask.shape[axis] - np.argmax(reversed_mask, axis=axis) - 1

    return first_idx, last_idx
