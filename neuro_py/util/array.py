from typing import Tuple

import numpy as np


def find_terminal_masked_indices(mask: np.ndarray, axis: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the first and last indices of non-masked values along an axis.

    Only tested upto 2D arrays. If `mask` is empty along `axis`, the first and
    last indices are set to 0 and the last index along the axis, respectively.

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

    Examples
    --------
    1D Example:
    >>> mask = np.array([0, 0, 1, 1, 0])
    >>> find_terminal_masked_indices(mask, axis=0)
    (2, 3)

    2D Example (along rows):
    >>> mask = np.array([[0, 0, 1],
    ...                  [1, 1, 0],
    ...                  [0, 0, 0]])
    >>> find_terminal_masked_indices(mask, axis=1)
    (array([2, 0, 0]), array([2, 1, -1]))

    2D Example (along columns):
    >>> find_terminal_masked_indices(mask, axis=0)
    (array([1, 1, 0]), array([1, 1, 0]))
    """
    first_idx = np.argmax(mask, axis=axis)
    reversed_mask = np.flip(mask, axis=axis)
    last_idx = mask.shape[axis] - np.argmax(reversed_mask, axis=axis) - 1

    return first_idx, last_idx

def replace_border_zeros_with_nan(arr: np.ndarray) -> np.ndarray:
    """Replace zero values at the borders of each dimension of a n-dimensional
    array with NaN.

    Parameters
    ----------
    arr : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        Array with zero values at the borders replaced with NaN.
    
    Examples
    --------
    >>> arr = np.arange(27).reshape(3, 3, 3)
    >>> arr[0, 2] = arr[2, 2] = arr[2, 0, 0] = arr[1, 1, 1] = 0
    >>> replace_border_zeros_with_nan(arr)
    array([[[nan,  1.,  2.],
            [ 3.,  4.,  5.],
            [nan, nan, nan]],

           [[ 9., 10., 11.],
            [12.,  0., 14.],
            [15., 16., 17.]],

           [[nan, 19., 20.],
            [21., 22., 23.],
            [nan, nan, nan]]])
    """
    arr = np.array(arr, dtype=float)
    dims = arr.shape

    for axis in range(len(dims)):      
        # Find indices where zero values start and stop
        for idx in np.ndindex(*[dims[i] for i in range(len(dims)) if i != axis]):
            slicer = list(idx)
            slicer.insert(axis, slice(None))  # Insert the full slice along the current axis

            # Check for first sequence of zeros
            subarray = arr[tuple(slicer)]
            first_zero_indices = np.where(np.cumsum(subarray != 0) == 0)[0]
            if len(first_zero_indices) > 0:
                subarray[first_zero_indices] = np.nan

            # Check for last sequence of zeros
            last_zero_indices = np.where(np.cumsum(subarray[::-1] != 0) == 0)[0]
            if len(last_zero_indices) > 0:
                subarray[-last_zero_indices-1] = np.nan

            arr[tuple(slicer)] = subarray  # Replace modified subarray

    return arr
