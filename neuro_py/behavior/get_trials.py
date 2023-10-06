from neuro_py.behavior import linear_positions
from neuro_py.behavior import well_traversal_classification
import numpy as np
from scipy import stats
import nelpy as nel
from neuro_py.io import loading
import pandas as pd
import os
import scipy.io as sio
from scipy.signal import medfilt
from neuro_py.process.intervals import find_interval
from typing import Tuple, List, Union
import logging

# linear track
def get_linear_maze_trials(basepath, epoch_input=None):
    """
    Get trials for linear maze
    Locates inbound and outbound laps for each linear track in session
    Input:
        basepath: str
        epoch_input: None, deprecated
    Output:
        pos: PositionArray
        inbound_laps: EpochArray
        outbound_laps: EpochArray

    """
    if epoch_input is not None:
        logging.warning("epoch_input is no longer supported")

    position_df = loading.load_animal_behavior(basepath)
    position_df_no_nan = position_df.query("not x.isnull() & not y.isnull()")
    
    if position_df_no_nan.shape[0] == 0:
        return None, None, None

    if "linearized" not in position_df_no_nan.columns:
        return None, None, None

    pos = nel.PositionArray(
        data=position_df_no_nan["linearized"].values.T,
        timestamps=position_df_no_nan.timestamps.values,
    )

    epoch_df = loading.load_epoch(basepath)
    epoch = nel.EpochArray(
        [np.array([epoch_df.startTime, epoch_df.stopTime]).T]
    )
    
    inbound_laps_temp = []
    outbound_laps_temp = []
    maze_idx = np.where(epoch_df.environment == "linear")[0]
    for idx in maze_idx:
        current_position = pos[epoch[int(idx)]]

        # get outbound and inbound epochs
        outbound_laps, inbound_laps = linear_positions.get_linear_track_lap_epochs(
            current_position.abscissa_vals, current_position.data[0], newLapThreshold=20
        )

        inbound_laps = linear_positions.find_good_lap_epochs(current_position, inbound_laps, min_laps=5)
        outbound_laps = linear_positions.find_good_lap_epochs(
            current_position, outbound_laps, min_laps=5
        )
        if not inbound_laps.isempty:
            inbound_laps_temp.append(inbound_laps.data)
        if not outbound_laps.isempty:
            outbound_laps_temp.append(outbound_laps.data)

    inbound_laps = nel.EpochArray(np.vstack(inbound_laps_temp))
    outbound_laps = nel.EpochArray(np.vstack(outbound_laps_temp))

    return pos, inbound_laps, outbound_laps


# tmaze
def get_t_maze_trials(basepath, epoch):
    def dissociate_laps_by_states(states, dir_epoch, states_of_interest=[1, 2]):
        # unique_states = np.unique(states.data[~np.isnan(states.data)])
        lap_id = []
        for ep in dir_epoch:
            state_count = []
            for us in states_of_interest:
                state_count.append(np.nansum(states[ep].data == us))
            lap_id.append(states_of_interest[np.argmax(state_count)])
        return np.array(lap_id).astype(int)

    position_df = loading.load_animal_behavior(basepath)
    position_df_no_nan = position_df.query("not x.isnull() & not y.isnull()")

    if position_df_no_nan.shape[0] == 0:
        return None, None, None

    if "linearized" not in position_df_no_nan.columns:
        return None, None, None

    if "states" not in position_df_no_nan.columns:
        return None, None, None

    pos = nel.PositionArray(
        data=position_df_no_nan["linearized"].values.T,
        timestamps=position_df_no_nan.timestamps.values,
    )

    pos = pos[epoch]
    if pos.isempty:
        return None, None, None
    
    states = nel.AnalogSignalArray(
        data=position_df_no_nan["states"].values.T,
        timestamps=position_df_no_nan.timestamps.values,
    )
    states = states[epoch]

    # get outbound and inbound epochs
    outbound_laps, inbound_laps = linear_positions.get_linear_track_lap_epochs(
        pos.abscissa_vals, pos.data[0], newLapThreshold=20
    )

    inbound_laps = linear_positions.find_good_lap_epochs(pos, inbound_laps, min_laps=5)
    outbound_laps = linear_positions.find_good_lap_epochs(
        pos, outbound_laps, min_laps=5
    )

    if outbound_laps.isempty:
        return None, None, None
    
    if not inbound_laps.isempty:
        logging.warning("inbound_laps should be empty for tmaze")

    if inbound_laps.n_intervals > outbound_laps.n_intervals:
        raise TypeError("inbound_laps should be less than outbound_laps for tmaze")

    # locate laps with the majority in state 1 or 2
    lap_id = dissociate_laps_by_states(states, outbound_laps, states_of_interest=[1, 2])

    right_epochs = nel.EpochArray(data=outbound_laps.data[lap_id == 1, :])
    left_epochs = nel.EpochArray(data=outbound_laps.data[lap_id == 2, :])

    position_df_no_nan = position_df_no_nan[
        position_df_no_nan["time"].between(epoch.start, epoch.stop)
    ]
    return pos, right_epochs, left_epochs


# wmaze
def get_w_maze_trials(
    basepath: str, max_distance_from_well: int = 20, min_distance_traveled: int = 50
):
    """
    Get trials for w maze
    :param basepath: basepath to session
    :param max_distance_from_well: maximum distance from well to be considered a trial
    :param min_distance_traveled: minimum distance traveled to be considered a trial
    :return: pos, trials, right_trials, left_trials

    metadata dependencies:
    animal.behavior.mat
        center, left, right x y coordinates *

    * can label these with label_key_locations_wmaze.m or manually
    """

    # load position and key location metadata
    filename = os.path.join(
        basepath, os.path.basename(basepath) + ".animal.behavior.mat"
    )
    data = sio.loadmat(filename, simplify_cells=True)

    # load epochs and place in array
    epoch_df = loading.load_epoch(basepath)
    epoch = nel.EpochArray(
        [np.array([epoch_df.startTime, epoch_df.stopTime]).T], label="session_epochs"
    )

    # load position and place in array
    position_df = loading.load_animal_behavior(basepath)
    position_df_no_nan = position_df.query("not x.isnull() & not y.isnull()")
    pos = nel.PositionArray(
        data=position_df_no_nan[["x", "y"]].values.T,
        timestamps=position_df_no_nan.timestamps.values,
    )
    wmaze_idx = np.where(epoch_df.environment == "wmaze")[0]
    for idx in wmaze_idx:
        # get key locations
        right_x = data["behavior"]["epochs"][idx]["right_x"]
        right_y = data["behavior"]["epochs"][idx]["right_y"]

        center_x = data["behavior"]["epochs"][idx]["center_x"]
        center_y = data["behavior"]["epochs"][idx]["center_y"]

        left_x = data["behavior"]["epochs"][idx]["left_x"]
        left_y = data["behavior"]["epochs"][idx]["left_y"]

        well_locations = np.array(
            [[center_x, center_y], [left_x, left_y], [right_x, right_y]]
        )

        # temp_df = position_df[~np.isnan(position_df.x)]
        segments_df, _ = well_traversal_classification.segment_path(
            position_df_no_nan["timestamps"].values,
            position_df_no_nan[["projected_x", "projected_y"]].values,
            well_locations,
            max_distance_from_well=max_distance_from_well,
        )

        segments_df = well_traversal_classification.score_inbound_outbound(
            segments_df, min_distance_traveled=min_distance_traveled
        )
        conditions = [
            "from_well == 'Center' & to_well == 'Left'",
            "from_well == 'Left' & to_well == 'Center'",
            "from_well == 'Center' & to_well == 'Right'",
            "from_well == 'Right' & to_well == 'Center'",
        ]
        condition_labels = [
            "center_left",
            "left_center",
            "center_right",
            "right_center",
        ]
        trajectories = {}
        for con, con_label in zip(conditions, condition_labels):
            trajectories[con_label] = nel.EpochArray(
                np.array(
                    [segments_df.query(con).start_time, segments_df.query(con).end_time]
                ).T
            )

    return pos, trajectories


# cheeseboard
def get_cheeseboard_trials(
    basepath: str,
    min_distance_from_home: int = 15,
    max_trial_time: int = 60 * 10,
    min_trial_time: int = 5,
    kernel_size: int = 2,
    min_std_away_from_home: int = 6,
):
    """
    get_cheeseboard_trials: get epochs of cheeseboard trials
    Input:
        basepath: basepath of session
        min_distance_from_home: minimum distance from home to be considered a trial
        max_trial_time: maximum time of a trial
        min_trial_time: minimum time of a trial
        kernel_size: size of kernel to use for smoothing
        min_std_away_from_home: minimum standard deviation away from home to be considered a trial
    Output:
        trial_epochs: epochs of trials

    metadata dependencies:
        animal.behavior.mat
            homebox_x within epochs *
            homebox_y within epochs *

        * can label these with label_key_locations_cheeseboard.m or manually
    """

    # load position and key location metadata
    filename = os.path.join(
        basepath, os.path.basename(basepath) + ".animal.behavior.mat"
    )
    data = sio.loadmat(filename, simplify_cells=True)

    # load epochs and place in array
    epoch_df = loading.load_epoch(basepath)
    epoch = nel.EpochArray(
        [np.array([epoch_df.startTime, epoch_df.stopTime]).T], label="session_epochs"
    )

    # load position and place in array
    position_df = loading.load_animal_behavior(basepath)
    position_df_no_nan = position_df.query("not x.isnull() & not y.isnull()")
    pos = nel.PositionArray(
        data=position_df_no_nan[["x", "y"]].values.T,
        timestamps=position_df_no_nan.timestamps.values,
    )
    # calculate kernel samples size based on sampling rate for x seconds
    kernel_size = int(pos.fs * kernel_size)
    # check if even number
    if kernel_size % 2 == 0:
        kernel_size += 1

    cheeseboard_idx = np.where(epoch_df.environment == "cheeseboard")[0]
    trials_temp = []
    stddev = []
    for idx in cheeseboard_idx:
        # get homebox location
        homebox_x = data["behavior"]["epochs"][idx]["homebox_x"]
        homebox_y = data["behavior"]["epochs"][idx]["homebox_y"]

        # get position during epoch
        current_pos = pos[epoch[int(idx)]]
        x, y = current_pos.data

        # calculate distance from homebox
        distance = np.sqrt((x - homebox_x) ** 2 + (y - homebox_y) ** 2)

        # median filter distance to remove noise (jumps in position)
        distance = medfilt(distance, kernel_size=kernel_size)

        # find intervals where distance is greater than min_distance_from_home
        dist_intervals = np.array(find_interval((distance > min_distance_from_home)))

        close_distances = distance[distance < min_distance_from_home]
        for trial in dist_intervals:
            far_distances = distance[trial[0] : trial[1]].mean()

            stddev.append(
                (np.abs(far_distances) - np.nanmean(np.abs(close_distances), axis=0))
                / np.nanstd(np.abs(close_distances), axis=0)
            )

        # get start and stop times of intervals
        if len(dist_intervals) > 0:
            trials_temp.append(current_pos.time[dist_intervals])

    # concatenate trials and place in EpochArray
    trials = nel.EpochArray(np.vstack(trials_temp))

    # remove trials that are too long or too short
    trials._data = trials.data[
        (trials.durations < max_trial_time)
        & (trials.durations > min_trial_time)
        & (np.array(stddev) > min_std_away_from_home)
    ]

    return pos, trials


# open field
def get_openfield_trials(
    basepath,
    spatial_binsize: int = 3,
    trial_time_bin_size: Union[int, float] = 1 * 60,
    prop_trial_sampled: float = 0.5,
    environments: List[str] = [
        "box",
        "bigSquare",
        "midSquare",
        "bigSquarePlus",
        "plus",
    ],
    minimum_correlation=0.6,
    method="correlation",
) -> Tuple[nel.PositionArray, nel.EpochArray]:
    """
    get_openfield_trials: get epochs of openfield trials

    The logic here is to find trials that have a minimum and even amount of
        spatial sampling (prop_trial_sampled) in order to assess spatial
        stability, population correlations, and other things.

    Input:
        basepath: basepath of session
        spatial_binsize: size of spatial bins to use for occupancy
        trial_time_bin_size: size of time bins to use for occupancy
        prop_trial_sampled: proportion of trials to sample
        environments: list of environments to include as openfield
        minimum_correlation: minimum correlation between trials to be considered a trial
        method: method to use (correlation,proportion)
            correlation - use correlation between the trial map and the overall map to determine if it is a trial
            proportion - use the proportion of the trial map that is sampled to determine if it is a trial
    Output:
        pos: position array
        trials: epochs of trials
    """

    def compute_occupancy_2d(
        pos_run: object, x_edges: list, y_edges: list
    ) -> np.ndarray:
        """
        compute_occupancy_2d: compute occupancy of 2d position
        Input:
            pos_run: position array
            x_edges: x edges of bins
            y_edges: y edges of bins
        Output:
            occupancy: occupancy of 2d position
        """
        occupancy, _, _ = np.histogram2d(
            pos_run.data[0, :], pos_run.data[1, :], bins=(x_edges, y_edges)
        )
        return occupancy / pos_run.fs

    # load position and place in array
    position_df = loading.load_animal_behavior(basepath)
    position_df_no_nan = position_df.query("not x.isnull() & not y.isnull()")
    pos = nel.PositionArray(
        data=position_df_no_nan[["x", "y"]].values.T,
        timestamps=position_df_no_nan.timestamps.values,
    )
    
    if pos.isempty:
        return pos, nel.EpochArray([], label="session_epochs")
    
    # load epochs and place in array
    epoch_df = loading.load_epoch(basepath)
    epoch = nel.EpochArray(
        [np.array([epoch_df.startTime, epoch_df.stopTime]).T], label="session_epochs"
    )

    # find epochs that are these environments
    openfield_idx = np.where(np.isin(epoch_df.environment, environments))[0]
    trials = []
    # loop through epochs
    for idx in openfield_idx:
        # get position during epoch
        current_position = pos[epoch[int(idx)]]

        if current_position.isempty:
            continue

        # get the edges of the position
        ext_xmin, ext_xmax = (
            np.floor(np.nanmin(current_position.data[0, :])),
            np.ceil(np.nanmax(current_position.data[0, :])),
        )
        ext_ymin, ext_ymax = (
            np.floor(np.nanmin(current_position.data[1, :])),
            np.ceil(np.nanmax(current_position.data[1, :])),
        )
        # create bin edges for occupancy map at spatial_binsize
        x_edges = np.arange(ext_xmin, ext_xmax + spatial_binsize, spatial_binsize)
        y_edges = np.arange(ext_ymin, ext_ymax + spatial_binsize, spatial_binsize)

        # compute occupancy map and get proportion of environment sampled
        occupancy = compute_occupancy_2d(current_position, x_edges, y_edges)
        overall_prop_sampled = sum(occupancy.flatten() > 0) / (
            (len(x_edges) - 1) * (len(y_edges) - 1)
        )
        # create possible trials based on trial_time_bin_size
        # these will be iterated over to find trials that are sampled enough
        duration = epoch_df.iloc[idx].stopTime - epoch_df.iloc[idx].startTime
        bins = np.linspace(
            epoch_df.iloc[idx].startTime,
            epoch_df.iloc[idx].stopTime,
            int(np.ceil(duration / (trial_time_bin_size))),
        )
        trials_temp = nel.EpochArray(np.array([bins[:-1], bins[1:]]).T)

        trial_i = 0
        # loop through possible trials and find when sampled enough
        for i_interval in range(trials_temp.n_intervals):
            # compute occupancy map and get proportion of environment sampled for trial
            trial_occupancy = compute_occupancy_2d(
                current_position[trials_temp[trial_i : i_interval + 1]],
                x_edges,
                y_edges,
            )

            if method == "correlation":
                # correlate trial_occupancy with overall occupancy
                r = np.corrcoef(
                    occupancy.flatten() > 0,
                    trial_occupancy.flatten() > 0,
                )[0, 1]

                # if sampled enough, add to trials
                if r > minimum_correlation:
                    trials.append(
                        [
                            trials_temp[trial_i : i_interval + 1].start,
                            trials_temp[trial_i : i_interval + 1].stop,
                        ]
                    )
                    # update trial_i to next interval to start from
                    trial_i = i_interval + 1

            elif method == "proportion":
                trial_prop_sampled = sum(trial_occupancy.flatten() > 0) / (
                    (len(x_edges) - 1) * (len(y_edges) - 1)
                )
                if trial_prop_sampled > prop_trial_sampled * overall_prop_sampled:
                    trials.append(
                        [
                            trials_temp[trial_i : i_interval + 1].start,
                            trials_temp[trial_i : i_interval + 1].stop,
                        ]
                    )
                    # update trial_i to next interval to start from
                    trial_i = i_interval + 1
            else:
                raise ValueError("method must be correlation or proportion")

    # concatenate trials and place in EpochArray
    trials = nel.EpochArray(np.vstack(trials))

    return pos, trials
