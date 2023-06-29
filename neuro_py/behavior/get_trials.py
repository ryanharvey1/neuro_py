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


# linear track
def get_linear_maze_trials(basepath, epoch):
    position_df = loading.load_animal_behavior(basepath)
    position_df_no_nan = position_df.query("not x.isnull() & not y.isnull()")
    if position_df_no_nan.shape[0] == 0:
        return None, None, None, None

    if "linearized" not in position_df_no_nan.columns:
        return None, None, None

    pos = nel.PositionArray(
        data=position_df_no_nan["linearized"].values.T,
        timestamps=position_df_no_nan.timestamps.values,
    )

    pos = pos[epoch]

    # get outbound and inbound epochs
    outbound_laps, inbound_laps = linear_positions.get_linear_track_lap_epochs(
        pos.abscissa_vals, pos.data[0], newLapThreshold=20
    )

    inbound_laps = linear_positions.find_good_lap_epochs(pos, inbound_laps, min_laps=5)
    outbound_laps = linear_positions.find_good_lap_epochs(
        pos, outbound_laps, min_laps=5
    )

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
        return None, None, None, None

    if "linearized" not in position_df_no_nan.columns:
        return None, None, None, None

    if "states" not in position_df_no_nan.columns:
        return None, None, None, None

    pos = nel.PositionArray(
        data=position_df_no_nan["linearized"].values.T,
        timestamps=position_df_no_nan.timestamps.values,
    )

    pos = pos[epoch]

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

    if not inbound_laps.isempty:
        raise TypeError("inbound_laps should be empty for tmaze")

    if outbound_laps.isempty:
        return None, None, None, None, None

    # locate laps with the majority in state 1 or 2
    lap_id = dissociate_laps_by_states(states, outbound_laps, states_of_interest=[1, 2])

    right_epochs = nel.EpochArray(data=outbound_laps.data[lap_id == 1, :])
    left_epochs = nel.EpochArray(data=outbound_laps.data[lap_id == 2, :])

    position_df_no_nan = position_df_no_nan[
        position_df_no_nan["time"].between(epoch.start, epoch.stop)
    ]
    return pos, right_epochs, left_epochs


# wmaze
def get_w_maze_trials(basepath, max_distance_from_well=20, min_distance_traveled=50):
    def flip_pos_within_epoch(pos, dir_epoch):
        """
        flip_pos_within_epoch: flips x coordinate within epoch
            Made to reverse x coordinate within nelpy array for replay analysis
        Input:
            pos: nelpy analog array with single dim
            dir_epoch: epoch to flip
        Output:
            pos: original pos, but fliped by epoch
        """

        def flip_x(x):
            return (x * -1) - np.nanmin(x * -1)

        # make pos df
        pos_df = pd.DataFrame()
        pos_df["ts"] = pos.abscissa_vals
        pos_df["x"] = pos.data.T
        pos_df["dir"] = False

        # make index within df of epoch
        for ep in dir_epoch:
            pos_df.loc[pos_df["ts"].between(ep.starts[0], ep.stops[0]), "dir"] = True

        # flip x within epoch
        pos_df.loc[pos_df.dir == True, "x"] = flip_x(pos_df[pos_df.dir == True].x)

        # add position back to input pos
        pos._data = np.expand_dims(pos_df.x.values, axis=0)

        return pos

    position_df = loading.load_animal_behavior(basepath)
    position_df_no_nan = position_df.query("not x.isnull() & not y.isnull()")

    well_locations = np.array(
        [
            [
                position_df.query("states == 0").projected_x.mean(),
                position_df.query("states == 0").projected_y.max(),
            ],
            [
                position_df.query("states == 2").projected_x.mean(),
                position_df.query("states == 2").projected_y.max(),
            ],
            [
                position_df.query("states == 1").projected_x.mean(),
                position_df.query("states == 1").projected_y.max(),
            ],
        ]
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

    # flip the x coordinate so it is always increasing
    for con in trajectories.keys():
        x_slope = []
        for pos_seg in pos[trajectories[con]]:
            # use regression to find slope of x coordinate
            b1, _, _, _, _ = stats.linregress(
                np.arange(len(pos_seg.data[0])), pos_seg.data[0]
            )
            x_slope.append(b1)
        # if the majority (>.5) of laps have x coords that decrease
        if np.mean(np.array(x_slope) < 0) > 0.5:
            pos = flip_pos_within_epoch(pos, trajectories[con])

    return pos, trajectories


# cheeseboard
def get_cheeseboard_trials(
    basepath: str,
    min_distance_from_home: int = 15,
    max_trial_time: int = 60*10,
    min_trial_time: int = 5,
    kernel_size: int = 2,
    min_std_away_from_home: int = 6,
) -> nel.EpochArray():
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

        * can label these with label_key_locations.m or manually
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

    return trials, pos

    # trials variable is not reliable, so calculate based on position
    # trials = data["behavior"]["trials"]

    # trials = nel.EpochArray(trials,domain=epoch.domain)

    # position_df = loading.load_animal_behavior(basepath)
    # position_df_no_nan = position_df.query("not x.isnull() & not y.isnull()")
    # pos = nel.PositionArray(
    #     data=position_df_no_nan[["x", "y"]].values.T,
    #     timestamps=position_df_no_nan.timestamps.values,
    #     support=epoch.domain
    # )
    # return pos[epoch], trials[epoch]


# open field (8 min bins)
