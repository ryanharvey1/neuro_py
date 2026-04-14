from typing import Tuple

import numpy as np
import pandas as pd


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


def zscore_columns(df: pd.DataFrame, ddof: int = 0) -> pd.DataFrame:
    """
    Z-score each column of a DataFrame.

    Each column is normalized independently by subtracting its mean
    and dividing by its standard deviation:

        z = (x - mean) / std

    Parameters
    ----------
    df : pandas.DataFrame
        Input data. Z-scoring is performed independently for each column.
    ddof : int, optional
        Delta degrees of freedom used when computing the standard deviation.
        The divisor used is ``N - ddof``. Default is 0 (population standard deviation).

    Returns
    -------
    pandas.DataFrame
        A new DataFrame with each column z-scored. The index and columns
        are preserved.

    Notes
    -----
    - Columns with zero variance will result in NaN values.
    - The input DataFrame is not modified.

    Examples
    --------
    Basic usage:

    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "a": [1, 2, 3],
    ...     "b": [10, 20, 30],
    ... })
    >>> z = zscore_columns(df, ddof=0)
    >>> z.round(3)
           a      b
    0 -1.225 -1.225
    1  0.000  0.000
    2  1.225  1.225

    Means are approximately zero:

    >>> z.mean().round(6)
    a    0.0
    b    0.0
    dtype: float64

    Standard deviations are approximately one:

    >>> z.std(ddof=0).round(6)
    a    1.0
    b    1.0
    dtype: float64

    Zero-variance columns produce NaNs:

    >>> df = pd.DataFrame({"constant": [1, 1, 1]})
    >>> zscore_columns(df)
       constant
    0       NaN
    1       NaN
    2       NaN
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if not numeric_cols.equals(df.columns):
        non_numeric = df.columns.difference(numeric_cols)
        raise TypeError(
            "zscore_columns expects all columns to be numeric; "
            f"non-numeric columns found: {list(non_numeric)}"
        )
    return (df - df.mean(axis=0)) / df.std(axis=0, ddof=ddof)


def smooth_peth(
    peth: np.ndarray | pd.Series | pd.DataFrame,
    smooth_window: float = 0.1,
    smooth_std: float = 1.0,
    dt: float | None = None,
) -> np.ndarray | pd.Series | pd.DataFrame:
    """
    Fast Gaussian smoothing for PETH-like data.

    Parameters
    ----------
    peth : np.ndarray, pandas.Series, or pandas.DataFrame
        Shape (time, units), (time,), or Series with time index.
    smooth_window : float
        Window size in same units as time axis
    smooth_std : float
        Gaussian std in same units as time axis.
        Internally converted to sample units for pandas Gaussian rolling.
    dt : float, optional
        Time step (required if peth is ndarray)

    Returns
    -------
    same type as input

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd

    >>> # --- Simulated PETH data ---
    >>> # 1000 time bins (e.g., -0.5 to 0.5 seconds), 10 neurons
    >>> n_time = 1000
    >>> n_neurons = 10

    >>> time = np.linspace(-0.5, 0.5, n_time)
    >>> dt = time[1] - time[0]

    >>> # Simulate noisy firing rates
    >>> rng = np.random.default_rng(0)
    >>> peth_array = rng.poisson(lam=5, size=(n_time, n_neurons)).astype(float)

    >>> # --- Example 1: NumPy input ---
    >>> smoothed_array = smooth_peth(
    ...     peth_array,
    ...     smooth_window=0.05,   # 50 ms window
    ...     smooth_std=0.01,      # 10 ms std
    ...     dt=dt,
    ... )

    >>> print(smoothed_array.shape)
    (1000, 10)


    >>> # --- Example 2: pandas DataFrame input ---
    >>> peth_df = pd.DataFrame(peth_array, index=time)

    >>> smoothed_df = smooth_peth(
    ...     peth_df,
    ...     smooth_window=0.05,
    ...     smooth_std=0.01,
    ... )

    >>> print(smoothed_df.shape)
    (1000, 10)


    >>> # --- Example 3: single neuron (1D input) ---
    >>> single_unit = peth_array[:, 0]

    >>> smoothed_single = smooth_peth(
    ...    single_unit,
    ...    smooth_window=0.05,
    ...    smooth_std=0.01,
    ...    dt=dt,
    ... )

    >>> print(smoothed_single.shape)
    (1000,)
    """
    if smooth_window <= 0:
        raise ValueError("smooth_window must be positive")
    if smooth_std <= 0:
        raise ValueError("smooth_std must be positive")

    # Determine input type and preserve metadata
    is_series = isinstance(peth, pd.Series)
    is_df = isinstance(peth, pd.DataFrame)
    return_ndarray_1d = False

    if is_series:
        # Convert Series to DataFrame for rolling, preserve name and index
        peth_df = peth.to_frame()
        index = peth.index
        series_name = peth.name
    elif is_df:
        peth_df = peth.copy()
        index = peth.index
        columns = peth.columns
    else:
        # ndarray input
        if dt is None:
            raise ValueError("dt must be provided for ndarray input")
        if dt <= 0:
            raise ValueError("dt must be positive for ndarray input")
        values = np.asarray(peth, dtype=float)
        if values.shape[0] < 2:
            raise ValueError("peth must contain at least 2 time samples")
        if values.ndim == 1:
            dt_local = dt
            time_index = np.arange(len(values)) * dt_local
            peth_df = pd.Series(values, index=time_index).to_frame()
            return_ndarray_1d = True
        else:
            dt_local = dt
            time_index = np.arange(values.shape[0]) * dt_local
            peth_df = pd.DataFrame(values, index=time_index)
        index = peth_df.index

    if len(peth_df.index) < 2:
        raise ValueError("peth must contain at least 2 time samples")

    # Get time step from index
    time_vals = np.asarray(peth_df.index, dtype=float)
    diffs = np.diff(time_vals)
    if not np.all(diffs > 0):
        raise ValueError("peth time/index must be strictly increasing")
    dt_local = float(diffs[0])
    if not np.allclose(diffs, dt_local, rtol=1e-6, atol=1e-12):
        raise ValueError("peth time/index must be uniformly sampled")

    # Convert smooth_window to samples
    window_samples = max(1, int(round(smooth_window / dt_local)))

    # pandas Gaussian window std is specified in sample units
    smooth_std_samples = smooth_std / dt_local

    # Use pandas rolling with Gaussian window
    smoothed_df = (
        peth_df.rolling(
            window=window_samples,
            win_type="gaussian",
            center=True,
            min_periods=1,
        )
        .mean(std=smooth_std_samples)
        .copy()
    )

    # Convert back to original type
    if is_series:
        return pd.Series(smoothed_df.iloc[:, 0].values, index=index, name=series_name)
    elif is_df:
        return pd.DataFrame(smoothed_df.values, index=index, columns=columns)
    else:
        # ndarray: return as array
        values = smoothed_df.values
        if return_ndarray_1d:
            return values[:, 0]
        else:
            return values
