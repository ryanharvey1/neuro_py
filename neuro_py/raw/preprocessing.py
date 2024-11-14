import os
import warnings
from typing import List, Tuple

import numpy as np


def zero_intervals_in_file(
    filepath: str,
    n_channels: int,
    zero_intervals: List[Tuple[int, int]],
    precision: str = "int16",
) -> None:
    """
    Zero out specified intervals in a binary file.

    Parameters
    ----------
    filepath : str
        Path to the binary file.
    n_channels : int
        Number of channels in the file.
    zero_intervals : List[Tuple[int, int]]
        List of intervals (start, end) in sample indices to zero out.
    precision : str, optional
        Data precision, by default "int16".

    Returns
    -------
    None

    Examples
    --------
    >>> fs = 20_000
    >>> zero_intervals_in_file(
    >>>     "U:\data\hpc_ctx_project\HP13\HP13_day12_20241112\HP13_day12_20241112.dat",
    >>>     n_channels=128,
    >>>     zero_intervals = (bad_intervals.data * fs).astype(int)
    >>>     )
    """
    # Check if file exists
    if not os.path.exists(filepath):
        warnings.warn("File does not exist.")
        return

    # Open the file in memory-mapped mode for read/write
    bytes_size = np.dtype(precision).itemsize
    with open(filepath, "rb") as f:
        startoffile = f.seek(0, 0)
        endoffile = f.seek(0, 2)
        n_samples = int((endoffile - startoffile) / n_channels / bytes_size)

    # Map the file to memory in read-write mode
    data = np.memmap(
        filepath, dtype=precision, mode="r+", shape=(n_samples, n_channels)
    )

    # Zero out the specified intervals
    zero_value = np.zeros((1, n_channels), dtype=precision)
    for start, end in zero_intervals:
        if 0 <= start < n_samples and 0 < end <= n_samples:
            data[start:end, :] = zero_value
        else:
            warnings.warn(
                f"Interval ({start}, {end}) is out of bounds and was skipped."
            )

    # Ensure changes are written to disk
    data.flush()

    # save log file with intervals zeroed out
    log_file = filepath.replace(".dat", "_zeroed_intervals.log")
    with open(log_file, "w") as f:
        for start, end in zero_intervals:
            f.write(f"{start} {end}\n")
