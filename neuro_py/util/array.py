from typing import Tuple

import numpy as np


def find_terminal_masked_indices(
    mask: np.ndarray, axis: int
) -> Tuple[np.ndarray, np.ndarray]:
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
            slicer.insert(
                axis, slice(None)
            )  # Insert the full slice along the current axis

            # Check for first sequence of zeros
            subarray = arr[tuple(slicer)]
            first_zero_indices = np.where(np.cumsum(subarray != 0) == 0)[0]
            if len(first_zero_indices) > 0:
                subarray[first_zero_indices] = np.nan

            # Check for last sequence of zeros
            last_zero_indices = np.where(np.cumsum(subarray[::-1] != 0) == 0)[0]
            if len(last_zero_indices) > 0:
                subarray[-last_zero_indices - 1] = np.nan

            arr[tuple(slicer)] = subarray  # Replace modified subarray

    return arr


def is_nested(array: np.ndarray) -> bool:
    """
    Check if an array is nested.

    Parameters
    ----------
    array : np.ndarray
        Input array.

    Returns
    -------
    bool
        True if the array is nested, False otherwise.

    Examples
    --------
    >>> is_nested(np.array([1, 2, 3]))
    False

    >>> is_nested(np.array([[1, 2], [3, 4]]))
    False

    >>> is_nested(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
    False

    >>> is_nested(np.array([np.array([1, 2]), np.array([3, 4, 5])], dtype=object))
    True
    """
    if array.dtype != object:
        return False
    return any(isinstance(item, np.ndarray) for item in array)


def circular_interp(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """
    Circular interpolation of data.
    This function performs interpolation on circular data, such as angles, using sine and cosine.

    Parameters
    ----------
    x : np.ndarray
        The x-coordinates at which to evaluate the interpolated values.
    xp : np.ndarray
        The x-coordinates of the data points, must be increasing.
    fp : np.ndarray
        The y-coordinates of the data points, same length as `xp`, [-π, π] or [0, 2π].

    Returns
    -------
    np.ndarray
        The interpolated values in radians, in the same range as the input fp data.
    """
    if len(xp) != len(fp):
        raise ValueError("xp and fp must have the same length.")
    if len(xp) < 2:
        raise ValueError("At least two points are required for interpolation.")
    if not np.all(np.diff(xp) > 0):
        raise ValueError("xp must be strictly increasing.")

    # interpolate sine and cosine components
    s = np.interp(x, xp, np.sin(fp))
    c = np.interp(x, xp, np.cos(fp))
    # return the angle formed by sine and cosine
    result = np.arctan2(s, c)
    
    # Detect if input data uses [0, 2π] convention instead of [-π, π]
    # Simple heuristic: if all input values are non-negative, assume [0, 2π]
    if np.all(fp >= 0):
        # Convert from [-π, π] to [0, 2π]
        result = (result + 2 * np.pi) % (2 * np.pi)
    
    return result
