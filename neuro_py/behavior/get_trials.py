import logging
import os
from typing import List, Tuple, Union

import nelpy as nel
import numpy as np
import scipy.io as sio
from scipy.signal import medfilt

from neuro_py.behavior import linear_positions, well_traversal_classification
from neuro_py.io import loading
from neuro_py.process.intervals import find_interval


def get_linear_maze_trials(basepath: str, epoch_input: None = None) -> Tuple[
    Union[nel.PositionArray, None],
    Union[nel.EpochArray, None],
    Union[nel.EpochArray, None],
]:
    """Get trials for linear maze.

    Locates inbound and outbound laps for each linear track in the session.

    Parameters
    ----------
    basepath : str
        The path to the base directory of the session data.
    epoch_input : None, optional
        Deprecated parameter. This is no longer supported.

    Returns
    -------
    pos : PositionArray or None
        The position data for the linear maze trials.
    inbound_laps : EpochArray or None
        The epochs corresponding to inbound laps.
    outbound_laps : EpochArray or None
        The epochs corresponding to outbound laps.

    Notes
    -----
    If no valid position data is found, None values are returned for all
    outputs.
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
    epoch = nel.EpochArray([np.array([epoch_df.startTime, epoch_df.stopTime]).T])

    domain = nel.EpochArray(
        [np.array([epoch_df.startTime.iloc[0], epoch_df.stopTime.iloc[-1]]).T]
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
        if not inbound_laps.isempty:
            inbound_laps = linear_positions.find_good_lap_epochs(
                current_position, inbound_laps, min_laps=5
            )

        if not outbound_laps.isempty:
            outbound_laps = linear_positions.find_good_lap_epochs(
                current_position, outbound_laps, min_laps=5
            )

        if not inbound_laps.isempty:
            inbound_laps_temp.append(inbound_laps.data)
        if not outbound_laps.isempty:
            outbound_laps_temp.append(outbound_laps.data)

    inbound_laps = nel.EpochArray(np.vstack(inbound_laps_temp), domain=domain)
    outbound_laps = nel.EpochArray(np.vstack(outbound_laps_temp), domain=domain)

    return pos, inbound_laps, outbound_laps


def get_t_maze_trials(
    basepath: str, epoch: nel.EpochArray, bypass_standard_behavior: bool = False
) -> Tuple[
    Union[nel.PositionArray, None],
    Union[nel.EpochArray, None],
    Union[nel.EpochArray, None],
]:
    """
    Get trials for T maze.

    This function retrieves position data and epochs for right and left trials
    based on the specified epoch. It checks if the number of outbound laps exceeds
    the number of inbound laps unless bypassed.

    Parameters
    ----------
    basepath : str
        The base path to the session data.
    epoch : nel.EpochArray
        The epoch to get trials for.
    bypass_standard_behavior : bool, optional
        If True, allows for more outbound than inbound trials. Default is False.

    Returns
    -------
    pos : PositionArray or None
        The position data for the T maze trials.
    right_epochs : EpochArray or None
        The epochs corresponding to right trials.
    left_epochs : EpochArray or None
        The epochs corresponding to left trials.

    Raises
    ------
    TypeError
        If inbound laps exceed outbound laps and bypass_standard_behavior is False.

    Notes
    -----
    If there are no valid positions or states in the session data, None is returned
    for all outputs.
    """

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

    if (
        inbound_laps.n_intervals > outbound_laps.n_intervals
    ) and not bypass_standard_behavior:
        raise TypeError("inbound_laps should be less than outbound_laps for tmaze")

    # locate laps with the majority in state 1 or 2
    lap_id = dissociate_laps_by_states(states, outbound_laps, states_of_interest=[1, 2])

    right_epochs = nel.EpochArray(data=outbound_laps.data[lap_id == 1, :])
    left_epochs = nel.EpochArray(data=outbound_laps.data[lap_id == 2, :])

    position_df_no_nan = position_df_no_nan[
        position_df_no_nan["time"].between(epoch.start, epoch.stop)
    ]
    return pos, right_epochs, left_epochs


def get_w_maze_trials(
    basepath: str, max_distance_from_well: int = 20, min_distance_traveled: int = 50
) -> Tuple[
    Union[nel.PositionArray, None],
    Union[np.ndarray, None],
    Union[np.ndarray, None],
    Union[np.ndarray, None],
]:
    """
    Get trials for W maze.

    This function retrieves position data and identifies trials for the W maze
    based on specified distance criteria.

    Parameters
    ----------
    basepath : str
        The base path to the session data.
    max_distance_from_well : int, optional
        The maximum distance from the well to be considered a trial. Default is 20.
    min_distance_traveled : int, optional
        The minimum distance traveled to be considered a trial. Default is 50.

    Returns
    -------
    pos : PositionArray or None
        The position data for the W maze trials.
    trials : ndarray or None
        The indices of the trials.
    right_trials : ndarray or None
        The indices of the right trials.
    left_trials : ndarray or None
        The indices of the left trials.

    Notes
    -----
    This function requires the following metadata dependencies:

    - `animal.behavior.mat`: contains center, left, and right x y coordinates.

    You can label these with `label_key_locations_wmaze.m` or manually.
    """

    # load position and key location metadata
    filename = os.path.join(
        basepath, os.path.basename(basepath) + ".animal.behavior.mat"
    )
    data = sio.loadmat(filename, simplify_cells=True)

    # load epochs and place in array
    epoch_df = loading.load_epoch(basepath)

    # load position and place in array
    position_df = loading.load_animal_behavior(basepath)
    position_df_no_nan = position_df.query("not x.isnull() & not y.isnull()")

    pos = nel.PositionArray(
        data=position_df_no_nan["linearized"].values.T,
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

        current_ts_idx = position_df_no_nan["timestamps"].between(
            epoch_df.iloc[idx].startTime, epoch_df.iloc[idx].stopTime
        )

        # temp_df = position_df[~np.isnan(position_df.x)]
        segments_df, _ = well_traversal_classification.segment_path(
            position_df_no_nan["timestamps"].values[current_ts_idx],
            position_df_no_nan[["x", "y"]].values[current_ts_idx],
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


def get_cheeseboard_trials(
    basepath: str,
    min_distance_from_home: int = 15,
    max_trial_time: int = 600,  # Default is 60 * 10
    min_trial_time: int = 5,
    kernel_size: int = 2,
    min_std_away_from_home: int = 6,
) -> Tuple[nel.PositionArray, nel.EpochArray]:
    """
    Get epochs of cheeseboard trials.

    This function retrieves epochs for cheeseboard trials based on specified 
    distance and time criteria.

    Parameters
    ----------
    basepath : str
        The base path to the session data.
    min_distance_from_home : int, optional
        The minimum distance from home to be considered a trial. Default is 15.
    max_trial_time : int, optional
        The maximum duration of a trial in seconds. Default is 600 (10 minutes).
    min_trial_time : int, optional
        The minimum duration of a trial in seconds. Default is 5.
    kernel_size : int, optional
        The size of the kernel to use for smoothing. Default is 2.
    min_std_away_from_home : int, optional
        The minimum standard deviation away from home to be considered a trial.
        Default is 6.

    Returns
    -------
    pos : PositionArray
        The position data for the cheeseboard trials.
    trials : EpochArray
        The epochs of the trials.

    Notes
    -----
    This function requires the following metadata dependencies:

    - `animal.behavior.mat`: contains homebox_x and homebox_y coordinates within epochs.

    You can label these with `label_key_locations_cheeseboard.m` or manually.
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


def get_openfield_trials(
    basepath: str,
    epoch_type: str = "epochs",
    spatial_binsize: int = 3,
    n_time_bins: int = 1,  # for bin_method = "fixed", not used for bin_method = "dynamic"
    bin_method: str = "dynamic",
    trial_time_bin_size: Union[int, float] = 60,  # in seconds for bin_method = "dynamic", not used for bin_method = "fixed"
    prop_trial_sampled: float = 0.5,
    environments: List[str] = [
        "box",
        "bigSquare",
        "midSquare",
        "bigSquarePlus",
        "plus",
    ],
    minimum_correlation: float = 0.6,
    method: str = "correlation",
) -> Tuple[nel.PositionArray, nel.EpochArray]:
    """
    Get epochs of openfield trials.

    This function identifies trials in an open field environment that meet
    specific criteria for spatial sampling to assess spatial stability and
    population correlations.

    Parameters
    ----------
    basepath : str
        The base path to the session data.
    epoch_type : str, optional
        The type of epoch to use ('trials' or 'epochs'). Default is 'epochs'.
    spatial_binsize : int, optional
        The size of spatial bins to use for occupancy. Default is 3.
    n_time_bins : int, optional
        The number of time bins to use for occupancy for fixed bin method. 
        Default is 1.
    bin_method : str, optional
        The method to use for binning time ('dynamic' or 'fixed'). 
        Default is 'dynamic'.
    trial_time_bin_size : Union[int, float], optional
        The size of time bins to use for occupancy for dynamic bin method 
        (in seconds). Default is 60.
    prop_trial_sampled : float, optional
        The proportion of trials to sample. Default is 0.5.
    environments : List[str], optional
        A list of environments to include as open field. Default includes 
        several environments such as 'box' and 'plus'.
    minimum_correlation : float, optional
        The minimum correlation between trials to be considered a trial. 
        Default is 0.6.
    method : str, optional
        The method to use ('correlation' or 'proportion'). Default is 
        'correlation'. `correlation` - use correlation between the trial map and
        the overall map to determine if it is a trial. `proportion` - use the
        proportion of the trial map that is sampled to determine if it is a
        trial

    Returns
    -------
    pos : PositionArray
        The position data for the open field trials.
    trials : EpochArray
        The epochs of the identified trials.

    Raises
    ------
    ValueError
        If the method is not 'correlation' or 'proportion'.

    Notes
    -----
    This function requires the loading of animal behavior and epoch data
    from the specified base path.
    """

    def compute_occupancy_2d(
        pos_run: object, x_edges: list, y_edges: list
    ) -> np.ndarray:
        """Compute occupancy of 2D position
        
        Parameters
        ----------
        pos_run : object
            Position data for the run
        x_edges : list
            Bin edges of x position
        y_edges : list
            Bin edges of y position
        
        Returns
        -------
        np.ndarray
            Occupancy map of the position
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
    if epoch_type == "trials":
        epoch_df = loading.load_trials(basepath)
        openfield_idx = np.arange(
            0, len(epoch_df)
        )  # assume trials make up all epochs associated with position
        trialsID = epoch_df.trialsID.values
    elif epoch_type == "epochs":
        epoch_df = loading.load_epoch(basepath)
        # find epochs that are these environments
        openfield_idx = np.where(np.isin(epoch_df.environment, environments))[0]

    epoch = nel.EpochArray([np.array([epoch_df.startTime, epoch_df.stopTime]).T])

    # find epochs that are these environments
    trials = []
    if epoch_type == "trials":
        trial_ID = []

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

        if bin_method == "dynamic":
            bins = np.linspace(
                epoch_df.iloc[idx].startTime,
                epoch_df.iloc[idx].stopTime,
                int(np.ceil(duration / (trial_time_bin_size))),
            )
        elif bin_method == "fixed":
            bins = np.arange(
                epoch_df.iloc[idx].startTime,
                epoch_df.iloc[idx].stopTime,
                int(np.floor(epoch[int(idx)].duration / n_time_bins)),
            )
        trials_temp = nel.EpochArray(np.array([bins[:-1], bins[1:]]).T)
        if epoch_type == "trials":
            temp_ID = trialsID[idx]

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
                    if epoch_type == "trials":
                        trial_ID.append(temp_ID + "_" + str(idx))
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
                    if epoch_type == "trials":
                        trial_ID.append(temp_ID + "_" + str(idx))
                    # update trial_i to next interval to start from
                    trial_i = i_interval + 1
            else:
                raise ValueError("method must be correlation or proportion")

    # concatenate trials and place in EpochArray
    if epoch_type == "trials":
        trials = nel.EpochArray(np.vstack(trials), label=np.vstack(trial_ID))
    else:
        trials = nel.EpochArray(np.vstack(trials), label="session_epochs")

    return pos, trials
