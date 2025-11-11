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


def interp_max_gap(
    x: np.ndarray, xp: np.ndarray, fp: np.ndarray, max_gap: float, fallback=np.nan
) -> np.ndarray:
    """
    Interpolate with gap-aware masking.

    Perform a 1-D linear interpolation similar to :func:`numpy.interp`, but
    any points that fall into gaps between consecutive `xp` values that
    exceed `max_gap` are set to `fallback` instead of being interpolated.

    Parameters
    ----------
    x : array_like
        The x-coordinates at which to evaluate the interpolated values.
    xp : 1-D sequence of floats
        The x-coordinates of the data points, must be increasing.
    fp : 1-D sequence of floats
        The y-coordinates of the data points, same length as `xp`.
    max_gap : float
        Maximum allowed gap between consecutive points in `xp`. If the gap
        xp[i+1] - xp[i] is greater than `max_gap`, then any `x` that falls in
        the open interval (xp[i], xp[i+1]) will be assigned `fallback`.
    fallback : scalar, optional
        Value to use for points falling inside a large gap or outside the
        extremes of `xp` (passed to ``numpy.interp`` as left/right). Default
        is ``np.nan``.

    Returns
    -------
    y_interpolated : ndarray
        Interpolated values. Same shape as `x`.

    Notes
    -----
    This function delegates to :func:`numpy.interp` for the initial
    interpolation and then post-processes the output to mask values that lie
    within large gaps of the original `xp` grid.

    Examples
    --------
    >>> xp = np.array([0, 1, 3])
    >>> fp = np.array([0, 10, 30])
    >>> x = np.array([0.5, 1.5, 2.5])
    >>> interp_max_gap(x, xp, fp, max_gap=1.5)
    array([ 5., nan, nan])
    """

    y_interpolated = np.interp(x, xp, fp, left=fallback, right=fallback)

    # Identify gaps in original data that exceed max_gap
    diff_x = np.diff(xp)

    # Find indices in x_new that fall within large gaps
    for i in range(len(diff_x)):
        if diff_x[i] > max_gap:
            # Get the range of x_new that falls in this large gap
            mask = (x > xp[i]) & (x < xp[i + 1])
            y_interpolated[mask] = fallback  # Set to fallback value if gap is too large

    return y_interpolated


def shrink(matrix: np.ndarray, row_bin_size: int, column_bin_size: int) -> np.ndarray:
    """
    Shrink a 2D matrix by averaging non-overlapping blocks.

    This reduces the resolution of the matrix by taking the mean of
    (row_bin_size x column_bin_size)-sized blocks. If the matrix size
    is not divisible by the bin sizes, NaNs are appended as evenly as possible.

    Parameters
    ----------
    matrix : np.ndarray
        2D input array.
    row_bin_size : int
        Number of rows to average together.
    column_bin_size : int
        Number of columns to average together.

    Returns
    -------
    shrunk : np.ndarray
        The downsampled (shrunk) matrix.

    Examples
    --------
    >>> matrix = np.array([[1, 2, 3, 4],
    ...                    [4, 5, 6, 7],
    ...                    [7, 8, 9, 10]])
    >>> shrink(matrix, 2, 2)
    array([[3., 5.],
           [7.5, 9.5]])
    """
    matrix = np.asarray(matrix, dtype=float)

    # Input validation
    if not (isinstance(row_bin_size, int) and row_bin_size >= 1):
        raise ValueError("row_bin_size must be a positive integer (>= 1)")
    if not (isinstance(column_bin_size, int) and column_bin_size >= 1):
        raise ValueError("column_bin_size must be a positive integer (>= 1)")
    if matrix.ndim != 2:
        raise ValueError("matrix must be a 2D array")

    n_rows, n_cols = matrix.shape

    # Determine padded size
    new_rows = int(np.ceil(n_rows / row_bin_size) * row_bin_size)
    new_cols = int(np.ceil(n_cols / column_bin_size) * column_bin_size)

    # Pad with NaNs as evenly as possible
    if new_rows > n_rows:
        pad_rows = new_rows - n_rows
        top = pad_rows // 2
        bottom = pad_rows - top
        matrix = np.pad(matrix, ((top, bottom), (0, 0)), constant_values=np.nan)

    if new_cols > n_cols:
        pad_cols = new_cols - n_cols
        left = pad_cols // 2
        right = pad_cols - left
        matrix = np.pad(matrix, ((0, 0), (left, right)), constant_values=np.nan)

    # Reshape into block structure
    r_blocks = new_rows // row_bin_size
    c_blocks = new_cols // column_bin_size

    # Efficiently compute block means ignoring NaNs
    shrunk = np.nanmean(
        matrix.reshape(r_blocks, row_bin_size, c_blocks, column_bin_size)
        .swapaxes(1, 2)
        .reshape(r_blocks, c_blocks, -1),
        axis=2,
    )

    return shrunk
