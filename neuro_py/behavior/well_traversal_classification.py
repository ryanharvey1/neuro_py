# https://github.com/Eden-Kramer-Lab/loren_frank_data_processing/blob/master/loren_frank_data_processing/well_traversal_classification.py

import numpy as np
import pandas as pd
from scipy.ndimage.measurements import label

_WELL_NAMES = {1: "Center", 2: "Left", 3: "Right"}


def paired_distances(x, y):
    """Euclidean distance between x and y at each time point.

    Parameters
    ----------
    x,y : ndarray, shape (n_time, n_space)

    Returns
    -------
    distance : ndarray, shape (n_time,)

    """
    x, y = np.array(x), np.array(y)
    x = np.atleast_2d(x).T if x.ndim < 2 else x
    y = np.atleast_2d(y).T if y.ndim < 2 else y
    return np.linalg.norm(x - y, axis=1)


def enter_exit_target(position, target, max_distance=1):
    """Marks when a position has reached a target ("enter")
    and when it has left a target ("exit").

    The position is considered to have reached a target when it is less than
    the `max_distance` from the target.

    Enter and exit times are marked as follows:
     1: entered the target radius
     0: neither
    -1: exited the target radius

    Works for 1D position and 2D position.

    Parameters
    ----------
    position : ndarray, shape (n_time, n_space)
    target : ndarray, shape (1, n_space)
    max_distance : float, optional
        How close the position is to the target to be considered at the target.

    Returns
    -------
    enter_exit : ndarray, shape (n_time,)
    at_target : ndarray, shape (n_time,)

    """
    distance_from_target = paired_distances(position, target)
    at_target = distance_from_target < max_distance
    enter_exit = np.r_[0, np.diff(at_target.astype(float))]
    return enter_exit, at_target


def enter_exit_target_dio(dio_indicator):
    at_target = (dio_indicator > 0).astype(np.float16)
    enter_exit = np.r_[0, np.diff(at_target)]
    return enter_exit, at_target


def shift_well_enters(enter_exit):
    """Shifts the enter times back one time point."""
    shifted_enter_exit = enter_exit.copy()
    old_ind = np.where(enter_exit > 0)[0]  # positive entries are well-entries
    new_ind = old_ind - 1
    shifted_enter_exit[new_ind] = enter_exit[old_ind]
    shifted_enter_exit[old_ind] = 0
    return shifted_enter_exit


def segment_path(
    time, position, well_locations, max_distance_from_well=10
):
    """Label traversals between each well location.

    Parameters
    ----------
    time : ndarray, shape (n_time,)
    position : ndarray, shape (n_time, n_space)
    well_locations : array_like, shape (n_wells, n_space)
    max_distance_from_well : float, optional
        The animal is considered at a well location if its position is closer
        than this value.

    Returns
    -------
    segments_df : pandas DataFrame, shape (n_segments, 6)
    labeled_segments : pandas DataFrame, shape (n_time,)

    """

    well_enter_exit, at_target = np.stack(
        [
            enter_exit_target(position, np.atleast_2d(well), max_distance_from_well)
            for well in well_locations
        ],
        axis=1,
    )
    n_wells = len(well_locations)
    well_labels = np.arange(n_wells) + 1
    well_enter_exit = np.sum(well_enter_exit.T * well_labels, axis=1)
    shifted_well_enter_exit = shift_well_enters(well_enter_exit)
    is_segment = ~(np.sum(at_target, axis=0) > 0)
    labeled_segments, n_segment_labels = label(is_segment)
    segment_labels = np.arange(n_segment_labels) + 1

    start_time, end_time, duration = [], [], []
    distance_traveled, from_well, to_well = [], [], []

    for segment_label in segment_labels:
        is_seg = np.in1d(labeled_segments, segment_label)
        segment_time = time[is_seg]
        start_time.append(segment_time.min())
        end_time.append(segment_time.max())
        duration.append(segment_time.max() - segment_time.min())
        try:
            start, _, end = np.unique(shifted_well_enter_exit[is_seg])
        except ValueError:
            start, end = np.nan, np.nan

        from_well.append(np.abs(start))
        to_well.append(np.abs(end))
        p = position[is_seg]
        distance_traveled.append(np.sum(paired_distances(p[1:], p[:-1])))

    data = [
        ("start_time", start_time),
        ("end_time", end_time),
        ("duration", duration),
        ("from_well", from_well),
        ("to_well", to_well),
        ("distance_traveled", distance_traveled),
    ]
    index = pd.Index(segment_labels, name="segment")
    return (
        pd.DataFrame.from_dict(dict(data)).set_index(index),
        pd.DataFrame(dict(labeled_segments=labeled_segments), index=time),
    )


def find_last_non_center_well(segments_df, segment_ind):
    last_wells = segments_df.iloc[:segment_ind].to_well
    try:
        return last_wells[last_wells != "Center"].iloc[-1]
    except IndexError:
        # There are no non-center wells. Just return current well.
        return ""


def get_correct_inbound_outbound(segments_df):
    n_segments = segments_df.shape[0]
    task = np.empty((n_segments,), dtype=object)
    turn = np.empty((n_segments,), dtype=object)
    is_correct = np.zeros((n_segments,), dtype=bool)

    for segment_ind in np.arange(n_segments):
        cur_segment = segments_df.iloc[segment_ind]
        if cur_segment.from_well == "Center":
            task[segment_ind] = "Outbound"
            last_non_center_well = find_last_non_center_well(segments_df, segment_ind)
            is_correct[segment_ind] = (cur_segment.to_well != last_non_center_well) & (
                cur_segment.to_well != "Center"
            )
            if (last_non_center_well != "") | ~is_correct[segment_ind]:
                turn[segment_ind] = last_non_center_well
            else:
                is_left_turn = (
                    (cur_segment.from_well == "Left")
                    & (cur_segment.to_well == "Center")
                ) | (
                    (cur_segment.from_well == "Center")
                    & (cur_segment.to_well == "Right")
                )

                turn[segment_ind] = "Left" if is_left_turn else "Right"
        else:
            task[segment_ind] = "Inbound"
            is_correct[segment_ind] = segments_df.iloc[segment_ind].to_well == "Center"
            turn[segment_ind] = cur_segment.from_well

    segments_df["task"] = task
    segments_df["is_correct"] = is_correct
    segments_df["turn"] = turn

    return segments_df


def score_inbound_outbound(
    segments_df, min_distance_traveled=50, well_names=_WELL_NAMES
):
    """In the alternating arm task, determines whether the trial should be
    inbound (running to the center arm) or outbound (running to the opposite
    outer arm as before) and if the trial was performed correctly.

    Parameters
    ----------
    segments_df : pandas DataFrame
        Output of `segment_path` function.
    min_distance_traveled : float, optional
        Minimum path length (in cm) while outside of the well radius for
        a segment to be considered as a trial.

    Returns
    -------
    segments_df : pandas DataFrame
        Same as the input dataframe but with the wells labeled
        (left, right, center) and columns for `task` (inbound/outbound) and
        `is_correct` (True/False).
    """
    segments_df = (
        segments_df.copy()
        .loc[segments_df.distance_traveled > min_distance_traveled]
        .dropna()
    )
    segments_df = segments_df.assign(
        to_well=lambda df: df.to_well.map(well_names),
        from_well=lambda df: df.from_well.map(well_names),
    )
    return get_correct_inbound_outbound(segments_df)
