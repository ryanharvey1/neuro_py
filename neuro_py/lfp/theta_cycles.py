import os
import sys

import numpy as np
import pandas as pd
import nelpy as nel

from typing import Union

from lazy_loader import attach as _attach
from neurodsp.filt import filter_signal
from scipy.io import savemat

from neuro_py.process.intervals import find_interval
from neuro_py.io import loading

__all__ = (
    "get_theta_channel",
    "process_lfp",
    "get_ep_from_df",
    "save_theta_cycles",
    "get_theta_cycles",
)
__getattr__, __dir__, __all__ = _attach(f"{__name__}", submodules=__all__)
del _attach

def get_theta_channel(basepath: str, tag: str = "CA1so") -> int:
    brain_region = loading.load_brain_regions(basepath)

    channel_tags = loading.load_channel_tags(basepath)

    if tag in brain_region.keys():
        theta_chan = brain_region[tag]["channels"]
    else:       
        return None
    
    bad_ch = channel_tags["Bad"]["channels"]
    for ch in theta_chan:
        if np.any(ch != bad_ch):
            theta_chan = ch
            break
    return ch - 1  # return in base 0


def process_lfp(basepath: str) -> tuple:
    nChannels, fs, _, _ = loading.loadXML(basepath)

    lfp, ts = loading.loadLFP(
        basepath, n_channels=nChannels, channel=None, frequency=fs
    )
    return lfp, ts, fs

def get_ep_from_df(df: pd.DataFrame,ts: np.ndarray):
# inputs a dataframe from bycycle and ts from lfp and returns a nelpy array of theta epochs 

    index_for_oscilation_epoch = find_interval(df.is_burst)
    start = []
    stop = []
    for idx in index_for_oscilation_epoch:
        start.append(df.sample_peak[idx[0]])
        stop.append(df.sample_peak[idx[1]])

    # convert list to array
    start = np.array(start)
    stop = np.array(stop)

    # index ts get get start and end ts for each oscillation epoch

    start_ts = ts[start]
    stop_ts = ts[stop]

    theta_epoch = nel.EpochArray([np.array([start_ts, stop_ts]).T])

    return theta_epoch

def save_theta_cycles(
    df: pd.DataFrame,
    ts: np.ndarray,
    basepath: str,
    detection_params: dict,
    ch: int,
    event_name: str = "thetacycles",
    detection_name: str = "bycycle",
) -> None:
    """
    Save theta cycles detected using bycycle to a .mat file in the cell explorer format.

    Parameters
    ----------
    df : bycycle dataframe (df_features)
    ts: array, timestamps of lfp
    basepath : str
        Basepath to save the file to.
    event_name : str
        Name of the events.
    detection_name : Union[None, str], optional
        Name of the detection, by default None
    detection_params : dictionary of detection parameters, by default thresholds
    ch: int, channel used for theta detection
    """
    filename = os.path.join(
        basepath, os.path.basename(basepath) + "." + event_name + ".events.mat"
    )
    data = {}
    data[event_name] = {}

    # create variables that will be saved
    timestamps = np.array(
        [ts[df.sample_peak.values[:-1]], ts[df.sample_peak.values[1:]]]
    )
    peaks = ts[df.sample_last_trough.values[1:]]
    amplitudes = df.band_amp.values[1:]
    duration = np.diff(
        np.array([ts[df.sample_peak.values[:-1]], ts[df.sample_peak.values[1:]]]),
        axis=0,
    )
    center = np.median(
        np.array([ts[df.sample_peak.values[:-1]], ts[df.sample_peak.values[1:]]]),
        axis=0,
    )

    # limit to cycles using is_burst
    timestamps = timestamps[:, df.is_burst.values[1:]]
    peaks = peaks[df.is_burst.values[1:]]
    amplitudes = amplitudes[df.is_burst.values[1:]]
    duration = duration[:, df.is_burst.values[1:]]
    center = center[df.is_burst.values[1:]]

    # save start_ts and stop_ts as 2d array
    data[event_name]["timestamps"] = timestamps.T
    data[event_name]["peaks"] = peaks.T
    data[event_name]["amplitudes"] = amplitudes.T
    data[event_name]["amplitudeUnits"] = "mV"
    data[event_name]["eventID"] = []
    data[event_name]["eventIDlabels"] = []
    data[event_name]["eventIDbinary"] = []

    # check if only single epoch
    data[event_name]["duration"] = duration.T

    data[event_name]["center"] = center.T
    data[event_name]["detectorinfo"] = {}
    if detection_name is None:
        data[event_name]["detectorinfo"]["detectorname"] = []
    else:
        data[event_name]["detectorinfo"]["detectorname"] = detection_name
    data[event_name]["detectorinfo"]["detectionparms"] = detection_params
    data[event_name]["detectorinfo"]["detectionintervals"] = []
    data[event_name]["detectorinfo"]["theta_channel"] = ch

    savemat(filename, data, long_field_names=True)


def get_theta_cycles(
    basepath: str,
    theta_freq: tuple[int] = (6, 10),
    lowpass: int = 48,
    detection_params: Union[dict, None] = None,
    ch: Union[int, None] = None,
):
    from bycycle import Bycycle
    # load lfp as memmap
    lfp, ts, fs = process_lfp(basepath)

    # get theta channel - default chooses CA1so
    if ch is None:
        ch = get_theta_channel(basepath,tag="CA1so")

    if ch is None:
        ch = get_theta_channel(basepath,tag="CA1sp")

    if ch is None:
        Warning("No theta channel found")
        return None
    
    # per bycycle documentation, low-pass filter signal before running bycycle 4x the frequency of interest
    filt_sig = filter_signal(lfp[:, ch], fs, "lowpass", lowpass, remove_edges=False)

    # for detecting theta epochs
    if detection_params is None:
        thresholds = {
            "amp_fraction": 0.1,
            "amp_consistency": 0.4,
            "period_consistency": 0.5,
            "monotonicity": 0.6,
            "min_n_cycles": 3,
        }
    else:
        thresholds = detection_params

    # initialize bycycle object
    bm = Bycycle(thresholds=thresholds)
    bm.fit(filt_sig, fs, theta_freq)

    save_theta_cycles(bm.df_features, ts, basepath, detection_params=thresholds, ch=ch)

# to run on cmd
if __name__ == "__main__":
    print(len(sys.argv))
    if len(sys.argv) == 2:
        run(sys.argv[1])
    elif len(sys.argv) == 3:
        run(sys.argv[1], epoch=int(sys.argv[2]))