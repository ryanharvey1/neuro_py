import os
from typing import Union

import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat

import neuro_py as npy


def filter_tracker_jumps(
    beh_df: pd.DataFrame, max_speed: Union[int, float] = 100
) -> pd.DataFrame:
    """
    Filter out tracker jumps (to NaN) in the behavior data.

    Parameters
    ----------
    beh_df : pd.DataFrame
        Behavior data with columns x, y, and ts.
    max_speed : Union[int,float], optional
        Maximum allowed speed in pixels per second.

    Returns
    -------
    pd.DataFrame

    Notes
    -----
    Will force dtypes of x and y to float64
    """

    # Calculate the Euclidean distance between consecutive points
    beh_df["dx"] = beh_df["x"].diff()
    beh_df["dy"] = beh_df["y"].diff()
    beh_df["distance"] = np.sqrt(beh_df["dx"] ** 2 + beh_df["dy"] ** 2)

    # Calculate the time difference between consecutive timestamps
    beh_df["dt"] = beh_df["ts"].diff()

    # Calculate the speed between consecutive points (distance / time)
    beh_df["speed"] = beh_df["distance"] / beh_df["dt"]

    # Identify the start of each jump
    # A jump starts when the speed exceeds the threshold, and the previous speed did not
    jump_starts = (beh_df["speed"] > max_speed) & (
        beh_df["speed"].shift(1) <= max_speed
    )

    # Mark x and y as NaN only for the first frame of each jump
    beh_df.loc[jump_starts, ["x", "y"]] = np.nan

    beh_df = beh_df.drop(columns=["dx", "dy", "distance", "dt", "speed"])

    return beh_df


def filter_tracker_jumps_in_file(
    basepath: str, epoch_number=None, epoch_interval=None
) -> None:
    """
    Filter out tracker jumps in the behavior data (to NaN) and save the filtered data back to the file.

    Parameters
    ----------
    basepath : str
        Basepath to the behavior file.
    epoch_number : int, optional
        Epoch number to filter the behavior data to.
    epoch_interval : tuple, optional
        Epoch interval to filter the behavior data to.

    Returns
    -------
    None

    Examples
    --------
    >>> basepath = "path/to/behavior/file"
    >>> filter_tracker_jumps_in_file(basepath, epoch_number=1)
    """

    # Load the behavior data
    file = os.path.join(basepath, os.path.basename(basepath) + "animal.behavior.mat")

    behavior = loadmat(file, simplify_cells=True)

    # Filter the behavior data to remove tracker jumps
    if epoch_number is not None:
        epoch_df = npy.io.load_epoch(basepath)
        idx = (
            behavior["behavior"]["timestamps"] > epoch_df.loc[epoch_number].startTime
        ) & (behavior["behavior"]["timestamps"] < epoch_df.loc[epoch_number].stopTime)
    elif epoch_interval is not None:
        idx = (behavior["behavior"]["timestamps"] > epoch_interval[0]) & (
            behavior["behavior"]["timestamps"] < epoch_interval[1]
        )
    else:
        # bool length of the same length as the number of timestamps
        idx = np.ones(len(behavior["behavior"]["timestamps"]), dtype=bool)

    # Filter the behavior data and add to dataframe
    x = behavior["behavior"]["position"]["x"][idx]
    y = behavior["behavior"]["position"]["y"][idx]
    ts = behavior["behavior"]["timestamps"][idx]
    beh_df = pd.DataFrame({"x": x, "y": y, "ts": ts})

    # Filter out tracker jumps
    beh_df = filter_tracker_jumps(beh_df)

    # Save the filtered behavior data back to the file
    behavior["behavior"]["position"]["x"][idx] = beh_df.x.values
    behavior["behavior"]["position"]["y"][idx] = beh_df.y.values

    savemat(file, behavior, long_field_names=True)
