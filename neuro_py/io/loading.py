"""Loading functions for cell explorer format"""

import glob
import multiprocessing
import os
import sys
import warnings
from itertools import chain
from typing import Any, Dict, List, Tuple, Union
from xml.dom import minidom

import nelpy as nel
import numpy as np
import pandas as pd
import scipy.io as sio
from joblib import Parallel, delayed
from scipy import signal

from neuro_py.behavior.kinematics import get_speed
from neuro_py.process.intervals import find_interval, in_intervals
from neuro_py.process.peri_event import get_participation


def loadXML(basepath: str) -> Union[Tuple[int, int, int, Dict[int, list]], None]:
    """
    Load XML file and extract relevant information.

    Parameters
    ----------
    basepath : str
        Path to the folder session containing the XML file.

    Returns
    -------
    Union[Tuple[int, int, int, Dict[int, list]], None]
        A tuple containing:
        - The number of channels (int)
        - The sampling frequency of the dat file (int)
        - The sampling frequency of the eeg file (int)
        - The mappings shanks to channels as a dict (Dict[int, list])
    """
    # check if saved file exists
    try:
        basename = os.path.basename(basepath)
        filename = glob.glob(os.path.join(basepath, basename + ".xml"))[0]
    except Exception:
        warnings.warn("xml file does not exist")
        return

    xmldoc = minidom.parse(filename)
    nChannels = (
        xmldoc.getElementsByTagName("acquisitionSystem")[0]
        .getElementsByTagName("nChannels")[0]
        .firstChild.data
    )
    fs_dat = (
        xmldoc.getElementsByTagName("acquisitionSystem")[0]
        .getElementsByTagName("samplingRate")[0]
        .firstChild.data
    )
    fs = (
        xmldoc.getElementsByTagName("fieldPotentials")[0]
        .getElementsByTagName("lfpSamplingRate")[0]
        .firstChild.data
    )

    shank_to_channel = {}
    groups = (
        xmldoc.getElementsByTagName("anatomicalDescription")[0]
        .getElementsByTagName("channelGroups")[0]
        .getElementsByTagName("group")
    )
    for i in range(len(groups)):
        shank_to_channel[i] = [
            int(child.firstChild.data)
            for child in groups[i].getElementsByTagName("channel")
        ]
    return int(nChannels), int(fs), int(fs_dat), shank_to_channel


def loadLFP(
    basepath: str,
    n_channels: int = 90,
    channel: Union[int, None] = None,
    frequency: float = 1250.0,
    precision: str = "int16",
    ext: str = "lfp",
    filename: str = None,  # name of file to load, located in basepath
):
    """
    Load LFP data from a specified file.

    Parameters
    ----------
    basepath : str
        Path to the folder containing the LFP file.
    n_channels : int, optional
        Number of channels, by default 90.
    channel : Optional[Union[int, list]], optional
        Specific channel(s) to load, by default None.
    frequency : float, optional
        Sampling frequency, by default 1250.0.
    precision : str, optional
        Data precision, by default "int16".
    ext : str, optional
        File extension, by default "lfp".
    filename : Optional[str], optional
        Name of the file to load, located in basepath, by default None.

    Returns
    -------
    Optional[Tuple[np.ndarray, np.ndarray]]
        Data and corresponding timestamps.

    Notes
    -----
    If both .lfp and .eeg files are present, .lfp file is prioritized.
    If neither are present, returns None.
    """
    if filename is not None:
        path = os.path.join(basepath, filename)
    else:
        path = ""
        if ext == "lfp":
            path = os.path.join(basepath, os.path.basename(basepath) + ".lfp")
            if not os.path.exists(path):
                path = os.path.join(basepath, os.path.basename(basepath) + ".eeg")
        if ext == "dat":
            path = os.path.join(basepath, os.path.basename(basepath) + ".dat")

    # check if saved file exists
    if not os.path.exists(path):
        warnings.warn("file does not exist")
        return
    if channel is None:
        n_channels = int(n_channels)

        f = open(path, "rb")
        startoffile = f.seek(0, 0)
        endoffile = f.seek(0, 2)
        bytes_size = 2
        n_samples = int((endoffile - startoffile) / n_channels / bytes_size)
        f.close()
        data = np.memmap(path, np.int16, "r", shape=(n_samples, n_channels))
        timestep = np.arange(0, n_samples) / frequency
        return data, timestep

    if type(channel) is not list:
        f = open(path, "rb")
        startoffile = f.seek(0, 0)
        endoffile = f.seek(0, 2)
        bytes_size = 2
        n_samples = int((endoffile - startoffile) / n_channels / bytes_size)
        f.close()
        with open(path, "rb") as f:
            data = np.fromfile(f, np.int16).reshape((n_samples, n_channels))[:, channel]
            timestep = np.arange(0, len(data)) / frequency
            # check if lfp time stamps exist
            lfp_ts_path = os.path.join(
                os.path.dirname(os.path.abspath(path)), "lfp_ts.npy"
            )
            if os.path.exists(lfp_ts_path):
                timestep = np.load(lfp_ts_path).reshape(-1)

            return data, timestep

    elif type(channel) is list:
        f = open(path, "rb")
        startoffile = f.seek(0, 0)
        endoffile = f.seek(0, 2)
        bytes_size = 2

        n_samples = int((endoffile - startoffile) / n_channels / bytes_size)
        f.close()
        with open(path, "rb") as f:
            data = np.fromfile(f, np.int16).reshape((n_samples, n_channels))[:, channel]
            timestep = np.arange(0, len(data)) / frequency
            # check if lfp time stamps exist
            lfp_ts_path = os.path.join(
                os.path.dirname(os.path.abspath(path)), "lfp_ts.npy"
            )
            if os.path.exists(lfp_ts_path):
                timestep = np.load(lfp_ts_path).reshape(-1)
            return data, timestep


class LFPLoader(object):
    """
    Simple class to load LFP or wideband data from a recording folder.

    Parameters
    ----------
    basepath : str
        Path to the recording folder.
    channels : Union[int, list, None], optional
        Channel number or list of channel numbers, by default None (load all channels memmap).
    ext : str, optional
        File extension, by default "lfp".
    epoch : Union[np.ndarray, nel.EpochArray, None], optional
        Epoch array or ndarray, by default None (load all data).

    Returns
    -------
    nelpy.AnalogSignalArray
        Analog signal array of shape (n_channels, n_samples).

    Examples
    --------
    >>> # load lfp file
    >>> basepath = r"X:/data/Barrage/NN10/day10"
    >>> lfp = loading.LFPLoader(basepath,ext="lfp")
    >>> lfp
        <AnalogSignalArray at 0x25ba1576640: 128 signals> for a total of 5:33:58:789 hours

    >>> # Loading dat file
    >>> dat = loading.LFPLoader(basepath,ext="dat")
    >>> dat
        <AnalogSignalArray at 0x25ba4fedc40: 128 signals> for a total of 5:33:58:790 hours
    >>> dat.lfp.data.shape
        (128, 400775808)
    >>> type(dat.lfp.data)
        numpy.memmap
    """

    def __init__(
        self,
        basepath: str,
        channels: Union[int, list, None] = None,
        ext: str = "lfp",
        epoch: Union[np.ndarray, nel.EpochArray, None] = None,
    ) -> None:
        self.basepath = basepath  # path to the recording folder
        self.channels = channels  # channel number or list of channel numbers
        self.ext = ext  # lfp or dat
        self.epoch = epoch

        # get xml data
        self.get_xml_data()

        # set sampling rate based on the extension of the file (lfp or dat)
        if self.ext == "dat":
            self.fs = self.fs_dat

        # load lfp
        self.load_lfp()

    def get_xml_data(self) -> None:
        nChannels, fs, fs_dat, shank_to_channel = loadXML(self.basepath)
        self.nChannels = nChannels
        self.fs = fs
        self.fs_dat = fs_dat
        self.shank_to_channel = shank_to_channel

    def load_lfp(self) -> None:
        lfp, timestep = loadLFP(
            self.basepath,
            n_channels=self.nChannels,
            channel=self.channels,
            frequency=self.fs,
            ext=self.ext,
        )

        if isinstance(self.epoch, nel.EpochArray):
            intervals = self.epoch.data
        elif isinstance(self.epoch, np.ndarray):
            intervals = self.epoch
            if intervals.ndim == 1:
                intervals = intervals[np.newaxis, :]
        else:
            intervals = np.array([0, timestep.shape[0] / self.fs])[np.newaxis, :]

        idx = in_intervals(timestep, intervals)

        # if loading all, don't index as to preserve memmap
        if idx.all():
            self.lfp = nel.AnalogSignalArray(
                data=lfp.T,
                timestamps=timestep,
                fs=self.fs,
                support=nel.EpochArray(intervals),
            )
        else:
            self.lfp = nel.AnalogSignalArray(
                data=lfp[idx, None].T,
                timestamps=timestep[idx],
                fs=self.fs,
                support=nel.EpochArray(
                    np.array([min(timestep[idx]), max(timestep[idx])])
                ),
            )

    def __repr__(self) -> None:
        return self.lfp.__repr__()

    def get_phase(self, band2filter: list = [6, 12], ford: int = 3) -> np.ndarray:
        """
        Get the phase of the LFP signal using a bandpass filter and Hilbert transform.

        Parameters
        ----------
        band2filter : list, optional
            The frequency band to filter, by default [6, 12].
        ford : int, optional
            The order of the Butterworth filter, by default 3.

        Returns
        -------
        np.ndarray
            The phase of the LFP signal.
        """
        band2filter = np.array(band2filter, dtype=float)
        b, a = signal.butter(ford, band2filter / (self.fs / 2), btype="bandpass")
        filt_sig = signal.filtfilt(b, a, self.lfp.data, padtype="odd")
        return np.angle(signal.hilbert(filt_sig))

    def get_freq_phase_amp(
        self, band2filter: list = [6, 12], ford: int = 3, kernel_size: int = 13
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the filtered signal, phase, amplitude, and filtered amplitude of the LFP signal.

        Parameters
        ----------
        band2filter : list, optional
            The frequency band to filter, by default [6, 12].
        ford : int, optional
            The order of the Butterworth filter, by default 3.
        kernel_size : int, optional
            The kernel size for the median filter, by default 13.

        Returns
        -------
        filt_sig : np.ndarray
            The filtered signal.
        phase : np.ndarray
            The phase of the LFP signal.
        amplitude : np.ndarray
            The amplitude of the LFP signal.
        amplitude_filtered : np.ndarray
            The filtered amplitude of the LFP signal.
        frequency : np.ndarray
            The instantaneous frequency of the LFP signal.
        """

        band2filter = np.array(band2filter, dtype=float)

        b, a = signal.butter(ford, band2filter / (self.fs / 2), btype="bandpass")

        filt_sig = signal.filtfilt(b, a, self.lfp.data, padtype="odd")
        phase = np.angle(signal.hilbert(filt_sig))
        amplitude = np.abs(signal.hilbert(filt_sig))
        amplitude_filtered = signal.filtfilt(b, a, amplitude, padtype="odd")

        # calculate the frequency
        # median filter to smooth the unwrapped phase (this is to avoid jumps in the frequency)
        filtered_signal = signal.medfilt2d(
            np.unwrap(phase), kernel_size=[1, kernel_size]
        )

        # Calculate the derivative of the unwrapped phase to get frequency
        dt = np.diff(self.lfp.abscissa_vals)
        if np.allclose(dt, dt[0]):  # Check if sampling is uniform
            dt = dt[0]  # Use a single scalar for uniform sampling
        else:
            dt = np.hstack((dt[0], dt))  # Use an array for non-uniform sampling
        derivative = np.gradient(filtered_signal, dt, axis=-1)
        frequency = derivative / (2 * np.pi)

        return filt_sig, phase, amplitude, amplitude_filtered, frequency
        

# Alias for backwards compatibility
class __init__(LFPLoader):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "Class 'LoadLfp' is deprecated, please use 'LFPLoader' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


def load_position(basepath: str, fs: float = 39.0625) -> Tuple[pd.DataFrame, float]:
    """
    Load position data from a .whl file in the specified directory.

    Parameters
    ----------
    basepath : str
        Path to the directory containing the .whl file.
    fs : float, optional
        Sampling frequency, by default 39.0625.

    Returns
    -------
    Tuple[pd.DataFrame, float]
        DataFrame containing position data and the sampling frequency.

    Notes
    -----
    If the directory does not exist or contains no .whl files, the function will exit.
    """
    if not os.path.exists(basepath):
        print("The path " + basepath + " doesn't exist; Exiting ...")
        sys.exit()
    listdir = os.listdir(basepath)
    whlfiles = [f for f in listdir if f.endswith(".whl")]
    if not len(whlfiles):
        print("Folder contains no whl files; Exiting ...")
        sys.exit()
    new_path = os.path.join(basepath, whlfiles[0])
    df = pd.read_csv(new_path, delimiter="\t", header=0, names=["x1", "y1", "x2", "y2"])
    df[df == -1] = np.nan
    return df, fs


def writeNeuroscopeEvents(path: str, ep: Any, name: str) -> None:
    """
    Write events to a Neuroscope-compatible file.

    Parameters
    ----------
    path : str
        Path to the output file.
    ep : Any
        Epoch data containing start and end times.
    name : str
        Name of the event.

    Returns
    -------
    None
    """
    f = open(path, "w")
    for i in range(len(ep)):
        f.writelines(
            str(ep.as_units("ms").iloc[i]["start"])
            + " "
            + name
            + " start "
            + str(1)
            + "\n"
        )
        # f.writelines(str(ep.as_units('ms').iloc[i]['peak']) + " "+name+" start "+ str(1)+"\n")
        f.writelines(
            str(ep.as_units("ms").iloc[i]["end"]) + " " + name + " end " + str(1) + "\n"
        )
    f.close()


def load_all_cell_metrics(basepaths: List[str]) -> pd.DataFrame:
    """
    Load cell metrics from multiple sessions.

    Parameters
    ----------
    basepaths : List[str]
        List of basepaths, can be a pandas column.

    Returns
    -------
    pd.DataFrame
        Concatenated pandas DataFrame with metrics.

    Notes
    -----
    To get waveforms, spike times, etc., use load_cell_metrics.
    """

    # to speed up, use parallel
    num_cores = multiprocessing.cpu_count()
    cell_metrics = Parallel(n_jobs=num_cores)(
        delayed(load_cell_metrics)(basepath, True) for basepath in basepaths
    )

    return pd.concat(cell_metrics, ignore_index=True)


def load_cell_metrics(
    basepath: str, only_metrics: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, dict]]:
    """
    Loader of cell-explorer cell_metrics.cellinfo.mat.

    Parameters
    ----------
    basepath : str
        Path to folder with cell_metrics.cellinfo.mat.
    only_metrics : bool, optional
        If True, only metrics are loaded, by default False.

    Returns
    -------
    Union[pd.DataFrame, Tuple[pd.DataFrame, dict]]
        DataFrame of single unit features and a dictionary with data that does not fit nicely into a DataFrame (waveforms, acgs, epochs, etc.).

    Notes
    -----
    See https://cellexplorer.org/datastructure/standard-cell-metrics/ for details.

    TODO: Extract all fields from cell_metrics.cellinfo. There are more items that can be extracted.
    """

    def extract_epochs(data):
        startTime = [
            ep["startTime"][0][0][0][0]
            for ep in data["cell_metrics"]["general"][0][0]["epochs"][0][0][0]
        ]
        stopTime = [
            ep["stopTime"][0][0][0][0]
            for ep in data["cell_metrics"]["general"][0][0]["epochs"][0][0][0]
        ]
        name = [
            ep["name"][0][0][0]
            for ep in data["cell_metrics"]["general"][0][0]["epochs"][0][0][0]
        ]

        epochs = pd.DataFrame()
        epochs["name"] = name
        epochs["startTime"] = startTime
        epochs["stopTime"] = stopTime
        return epochs

    def extract_events(data):
        psth = {}
        for dt in data["cell_metrics"]["events"][0][0].dtype.names:
            psth[dt] = pd.DataFrame(
                index=data["cell_metrics"]["general"][0][0][0]["events"][0][dt][0][0][
                    "x_bins"
                ][0][0].T[0]
                / 1000,
                data=np.hstack(data["cell_metrics"]["events"][0][0][dt][0][0][0]),
            )
        return psth

    def extract_general(data):
        # extract fr per unit with lag zero to ripple
        try:
            ripple_fr = [
                ev.T[0]
                for ev in data["cell_metrics"]["events"][0][0]["ripples"][0][0][0]
            ]
        except Exception:
            ripple_fr = []
        # extract spikes times
        spikes = [
            spk.T[0] for spk in data["cell_metrics"]["spikes"][0][0]["times"][0][0][0]
        ]
        # extract epochs
        try:
            epochs = extract_epochs(data)
        except Exception:
            epochs = []

        # extract events
        try:
            events_psth = extract_events(data)
        except Exception:
            events_psth = []

        # extract avg waveforms
        try:
            waveforms = np.vstack(
                data["cell_metrics"]["waveforms"][0][0]["filt"][0][0][0]
            )
        except Exception:
            try:
                waveforms = [
                    w.T for w in data["cell_metrics"]["waveforms"][0][0][0][0][0][0]
                ]
            except Exception:
                waveforms = [w.T for w in data["cell_metrics"]["waveforms"][0][0][0]]
        # extract chanCoords
        try:
            chanCoords_x = data["cell_metrics"]["general"][0][0]["chanCoords"][0][0][0][
                0
            ]["x"].T[0]
            chanCoords_y = data["cell_metrics"]["general"][0][0]["chanCoords"][0][0][0][
                0
            ]["y"].T[0]
        except Exception:
            chanCoords_x = []
            chanCoords_y = []

        # add to dictionary
        data_ = {
            "acg_wide": data["cell_metrics"]["acg"][0][0]["wide"][0][0],
            "acg_narrow": data["cell_metrics"]["acg"][0][0]["narrow"][0][0],
            "acg_log10": data["cell_metrics"]["acg"][0][0]["log10"][0][0],
            "ripple_fr": ripple_fr,
            "chanCoords_x": chanCoords_x,
            "chanCoords_y": chanCoords_y,
            "epochs": epochs,
            "spikes": spikes,
            "waveforms": waveforms,
            "events_psth": events_psth,
        }
        return data_

    def un_nest_df(df):
        # Un-nest some strings are nested within brackets (a better solution exists...)
        # locate and iterate objects in df
        for item in df.keys()[df.dtypes == "object"]:
            # if you can get the size of the first item with [0], it is nested
            # otherwise it fails and is not nested
            try:
                df[item][0][0].size
                # the below line is from: https://www.py4u.net/discuss/140913
                df[item] = df[item].str.get(0)
            except Exception:
                continue
        return df

    filename = os.path.join(
        basepath, os.path.basename(basepath) + ".cell_metrics.cellinfo.mat"
    )
    # filename = glob.glob(os.path.join(basepath, "*.cell_metrics.cellinfo.mat"))[0]

    # check if saved file exists
    if not os.path.exists(filename):
        warnings.warn("file does not exist")
        if only_metrics:
            return None
        return None, None

    # load cell_metrics file
    data = sio.loadmat(filename)

    # construct data frame with features per neuron
    df = {}
    # count units
    n_cells = data["cell_metrics"]["UID"][0][0][0].size
    dt = data["cell_metrics"].dtype
    for dn in dt.names:
        # check if var has the right n of units and is a vector
        try:
            if (data["cell_metrics"][dn][0][0][0][0].size == 1) & (
                data["cell_metrics"][dn][0][0][0].size == n_cells
            ):
                # check if nested within brackets
                try:
                    df[dn] = [
                        value[0] if len(value) == 1 else value
                        for value in data["cell_metrics"][dn][0][0][0]
                    ]
                except Exception:
                    df[dn] = data["cell_metrics"][dn][0][0][0]
        except Exception:
            continue

    df = pd.DataFrame(df)

    # load in tag
    # check if tags exist within cell_metrics
    if "tags" in data.get("cell_metrics").dtype.names:
        # get names of each tag
        dt = data["cell_metrics"]["tags"][0][0].dtype
        if len(dt) > 0:
            # iter through each tag
            for dn in dt.names:
                # set up column for tag
                df["tags_" + dn] = [False] * df.shape[0]
                # iter through uid
                for uid in data["cell_metrics"]["tags"][0][0][dn][0][0].flatten():
                    df.loc[df.UID == uid, "tags_" + dn] = True

    # add bad unit tag for legacy
    df["bad_unit"] = [False] * df.shape[0]
    if "tags_Bad" in df.keys():
        df.bad_unit = df.tags_Bad
        df.bad_unit = df.bad_unit.replace({np.nan: False})

    # add data from general metrics
    df["basename"] = data["cell_metrics"]["general"][0][0]["basename"][0][0][0]
    df["basepath"] = basepath
    df["sex"] = data["cell_metrics"]["general"][0][0]["animal"][0][0]["sex"][0][0][0]
    df["species"] = data["cell_metrics"]["general"][0][0]["animal"][0][0]["species"][0][
        0
    ][0]
    df["strain"] = data["cell_metrics"]["general"][0][0]["animal"][0][0]["strain"][0][
        0
    ][0]
    try:
        df["geneticLine"] = data["cell_metrics"]["general"][0][0]["animal"][0][0][
            "geneticLine"
        ][0][0][0]
    except Exception:
        pass
    df["cellCount"] = data["cell_metrics"]["general"][0][0]["cellCount"][0][0][0][0]

    # fix nesting issue for strings
    df = un_nest_df(df)

    # convert nans within tags columns to false
    cols = df.filter(regex="tags_").columns
    df[cols] = df[cols].replace({np.nan: False})

    if only_metrics:
        return df

    # extract other general data and put into dict
    data_ = extract_general(data)

    return df, data_


def load_SWRunitMetrics(basepath: str) -> pd.DataFrame:
    """
    Load SWRunitMetrics.mat into a pandas DataFrame.

    Parameters
    ----------
    basepath : str
        Path to the folder containing the SWRunitMetrics.mat file.

    Returns
    -------
    pd.DataFrame
        DataFrame with the following fields:
        - particip: the probability of participation into ripples for each unit
        - FRall: mean firing rate during ripples
        - FRparticip: mean firing rate for ripples with at least 1 spike
        - nSpkAll: mean number of spikes in all ripples
        - nSpkParticip: mean number of spikes in ripples with at least 1 spike
        - epoch: behavioral epoch label
    """

    def extract_swr_epoch_data(data, epoch):
        # get var names
        dt = data["SWRunitMetrics"][epoch][0][0].dtype

        df2 = pd.DataFrame()

        # get n units
        # there might be other fields within here like the epoch timestamps
        # skip those by returning empty df
        try:
            n_cells = data["SWRunitMetrics"][epoch][0][0][0]["particip"][0].shape[0]
        except Exception:
            return df2

        for dn in dt.names:
            if (data["SWRunitMetrics"][epoch][0][0][0][dn][0].shape[1] == 1) & (
                data["SWRunitMetrics"][epoch][0][0][0][dn][0].shape[0] == n_cells
            ):
                df2[dn] = data["SWRunitMetrics"][epoch][0][0][0][dn][0].T[0]
        df2["epoch"] = epoch
        return df2

    try:
        filename = glob.glob(os.path.join(basepath, "*.SWRunitMetrics.mat"))[0]
    except Exception:
        warnings.warn("file does not exist")
        return pd.DataFrame()

    # load file
    data = sio.loadmat(filename)

    df2 = pd.DataFrame()
    # loop through each available epoch and pull out contents
    for epoch in data["SWRunitMetrics"].dtype.names:
        if data["SWRunitMetrics"][epoch][0][0].size > 0:  # not empty
            # call content extractor
            df_ = extract_swr_epoch_data(data, epoch)

            # append conents to overall data frame
            if df_.size > 0:
                df2 = pd.concat([df2, df_], ignore_index=True)

    return df2


def _add_manual_events(df: pd.DataFrame, added_ts: list) -> pd.DataFrame:
    """
    Add new rows to a dataframe representing manual events (from Neuroscope2)
    with durations equal to the mean duration of the existing events.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe, with at least two columns called 'start' and 'stop',
        representing the start and stop times of the events.
    added_ts : list
        A list of timestamps representing the peaks of the new events to be added
        to the dataframe.

    Returns
    -------
    pd.DataFrame
        The modified dataframe with the new rows added and sorted by the 'peaks' column.
    """
    # Calculate the mean duration of the existing events
    mean_duration = (df["stop"] - df["start"]).mean()

    # Create a new dataframe with a 'peaks' column equal to the added_ts values
    df_added = pd.DataFrame()
    df_added["peaks"] = added_ts

    # Calculate the start and stop times of the new events based on the mean duration
    df_added["start"] = added_ts - mean_duration / 2
    df_added["stop"] = added_ts + mean_duration / 2

    # Calculate the duration of the new events as the mean duration
    df_added["duration"] = df_added.stop.values - df_added.start.values

    # Append the new events to the original dataframe
    df = pd.concat([df, df_added], ignore_index=True)

    # Sort the dataframe by the 'peaks' column
    df.sort_values(by=["peaks"], ignore_index=True, inplace=True)

    return df


def load_ripples_events(
    basepath: str, return_epoch_array: bool = False, manual_events: bool = True
) -> Union[pd.DataFrame, nel.EpochArray]:
    """
    Load info from ripples.events.mat and store within a DataFrame.

    Parameters
    ----------
    basepath : str
        Path to your session where ripples.events.mat is located.
    return_epoch_array : bool, optional
        If True, the output will be an EpochArray, by default False.
    manual_events : bool, optional
        If True, add manually added events from Neuroscope2 (interval will be calculated from mean event duration), by default True.

    Returns
    -------
    Union[pd.DataFrame, nel.EpochArray]
        DataFrame with the following fields:
        - start: start time of ripple
        - stop: end time of ripple
        - peaks: peak time of ripple
        - amplitude: envelope value at peak time
        - duration: ripple duration
        - frequency: instant frequency at peak
        - detectorName: the name of ripple detector used
        - event_spk_thres: 1 or 0 for if a mua threshold was used
        - basepath: path name
        - basename: session id
        - animal: animal id

    Notes
    -----
    * Note that basepath/basename/animal relies on specific folder structure and may be incorrect for some data structures.
    """

    # locate .mat file
    filename = os.path.join(
        basepath, os.path.basename(basepath) + ".ripples.events.mat"
    )
    if not os.path.exists(filename):
        warnings.warn("file does not exist")
        return pd.DataFrame()

    # load matfile
    data = sio.loadmat(filename)

    # make data frame of known fields
    df = pd.DataFrame()
    try:
        df["start"] = data["ripples"]["timestamps"][0][0][:, 0]
        df["stop"] = data["ripples"]["timestamps"][0][0][:, 1]
    except Exception:
        df["start"] = data["ripples"]["times"][0][0][:, 0]
        df["stop"] = data["ripples"]["times"][0][0][:, 1]

    for name in ["peaks", "amplitude", "duration", "frequency", "peakNormedPower"]:
        try:
            df[name] = data["ripples"][name][0][0]
        except Exception:
            df[name] = np.nan

    if df.duration.isna().all():
        df["duration"] = df.stop - df.start

    try:
        df["detectorName"] = data["ripples"]["detectorinfo"][0][0]["detectorname"][0][
            0
        ][0]
    except Exception:
        try:
            df["detectorName"] = data["ripples"]["detectorName"][0][0][0]
        except Exception:
            df["detectorName"] = "unknown"

    # find ripple channel (this can be in several places depending on the file)
    try:
        df["ripple_channel"] = data["ripples"]["detectorinfo"][0][0]["detectionparms"][
            0
        ][0]["Channels"][0][0][0][0]
    except Exception:
        try:
            df["ripple_channel"] = data["ripples"]["detectorParams"][0][0]["channel"][
                0
            ][0][0][0]
        except Exception:
            try:
                df["ripple_channel"] = data["ripples"]["detectorinfo"][0][0][
                    "detectionparms"
                ][0][0]["channel"][0][0][0][0]
            except Exception:
                try:
                    df["ripple_channel"] = data["ripples"]["detectorinfo"][0][0][
                        "detectionparms"
                    ][0][0]["ripple_channel"][0][0][0][0]
                except Exception:
                    try:
                        df["ripple_channel"] = data["ripples"]["detectorinfo"][0][0][
                            "detectionchannel1"
                        ][0][0][0][0]
                    except Exception:
                        df["ripple_channel"] = np.nan

    # remove flagged ripples, if exist
    try:
        df.drop(
            labels=np.array(data["ripples"]["flagged"][0][0]).T[0] - 1,
            axis=0,
            inplace=True,
        )
        df.reset_index(inplace=True)
    except Exception:
        pass

    # adding manual events
    if manual_events:
        try:
            df = _add_manual_events(df, data["ripples"]["added"][0][0].T[0])
        except Exception:
            pass

    # adding if ripples were restricted by spikes
    dt = data["ripples"].dtype
    if "eventSpikingParameters" in dt.names:
        df["event_spk_thres"] = 1
    else:
        df["event_spk_thres"] = 0

    # get basename and animal
    normalized_path = os.path.normpath(filename)
    path_components = normalized_path.split(os.sep)
    df["basepath"] = basepath
    df["basename"] = path_components[-2]
    df["animal"] = path_components[-3]

    if return_epoch_array:
        return nel.EpochArray([np.array([df.start, df.stop]).T], label="ripples")

    return df


def load_theta_cycles(
    basepath: str, return_epoch_array: bool = False
) -> Union[pd.DataFrame, nel.EpochArray]:
    """
    Load theta cycles calculated from auto_theta_cycles.m.

    Parameters
    ----------
    basepath : str
        Path to your session where thetacycles.events.mat is located.
    return_epoch_array : bool, optional
        If True, the output will be an EpochArray, by default False.

    Returns
    -------
    Union[pd.DataFrame, nel.EpochArray]
        DataFrame with the following fields:
        - start: start time of theta cycle
        - stop: end time of theta cycle
        - duration: theta cycle duration
        - center: center time of theta cycle
        - trough: trough time of theta cycle
        - theta_channel: the theta channel used for detection
    """
    filename = os.path.join(
        basepath, os.path.basename(basepath) + ".thetacycles.events.mat"
    )
    if not os.path.exists(filename):
        warnings.warn("file does not exist")
        if return_epoch_array:
            return nel.EpochArray()
        return pd.DataFrame()

    data = sio.loadmat(filename, simplify_cells=True)
    df = pd.DataFrame()
    df["start"] = data["thetacycles"]["timestamps"][:, 0]
    df["stop"] = data["thetacycles"]["timestamps"][:, 1]
    df["duration"] = data["thetacycles"]["duration"]
    df["center"] = data["thetacycles"]["center"]
    df["trough"] = data["thetacycles"]["peaks"]
    df["theta_channel"] = data["thetacycles"]["detectorinfo"]["theta_channel"]

    if return_epoch_array:
        return nel.EpochArray([np.array([df.start, df.stop]).T], label="theta_cycles")
    return df


def load_barrage_events(
    basepath: str,
    return_epoch_array: bool = False,
    restrict_to_nrem: bool = True,
    min_duration: float = 0.0,
) -> Union[pd.DataFrame, nel.EpochArray]:
    """
    Load barrage events from the .HSEn2.events.mat file.

    Parameters
    ----------
    basepath : str
        Basepath to the session folder.
    return_epoch_array : bool, optional
        If True, return an EpochArray instead of a DataFrame, by default False
    restrict_to_nrem : bool, optional
        If True, restrict to NREM sleep, by default True
    min_duration : float, optional
        Minimum duration of a barrage, by default 0.0

    Returns
    -------
    Union[pd.DataFrame, nel.EpochArray]
        DataFrame with barrage events.
    """

    # locate barrage file
    filename = os.path.join(basepath, os.path.basename(basepath) + ".HSEn2.events.mat")

    # check if file exists
    if os.path.exists(filename) is False:
        warnings.warn("No barrage file found for {}".format(basepath))
        if return_epoch_array:
            return nel.EpochArray()
        return pd.DataFrame()

    # load data from file and extract relevant data
    data = sio.loadmat(filename, simplify_cells=True)
    data = data["HSEn2"]

    # convert to DataFrame
    df = pd.DataFrame()
    df["start"] = data["timestamps"][:, 0]
    df["stop"] = data["timestamps"][:, 1]
    df["peaks"] = data["peaks"]
    df["duration"] = data["timestamps"][:, 1] - data["timestamps"][:, 0]

    # restrict to NREM sleep
    if restrict_to_nrem:
        state_dict = load_SleepState_states(basepath)
        nrem_epochs = nel.EpochArray(state_dict["NREMstate"]).expand(2)
        idx = in_intervals(df["start"].values, nrem_epochs.data)
        df = df[idx].reset_index(drop=True)

    # restrict to barrages with a minimum duration
    df = df[df.duration > min_duration].reset_index(drop=True)

    # make sure each barrage has some ca2 activity
    # load ca2 pyr cells
    st, _ = load_spikes(basepath, putativeCellType="Pyr", brainRegion="CA2")
    # bin spikes into barrages
    bst = get_participation(st.data, df["start"].values, df["stop"].values)
    # keep only barrages with some activity
    df = df[np.sum(bst > 0, axis=0) > 0].reset_index(drop=True)

    if return_epoch_array:
        return nel.EpochArray([np.array([df.start, df.stop]).T], label="barrage")

    # get basename and animal
    normalized_path = os.path.normpath(filename)
    path_components = normalized_path.split(os.sep)
    df["basepath"] = basepath
    df["basename"] = path_components[-2]
    df["animal"] = path_components[-3]

    return df


def load_ied_events(
    basepath: str, manual_events: bool = True, return_epoch_array: bool = False
) -> Union[pd.DataFrame, nel.EpochArray]:
    """
    Load info from ripples.events.mat and store within a DataFrame.

    Parameters
    ----------
    basepath : str
        Path to your session where ripples.events.mat is located.
    return_epoch_array : bool, optional
        If True, the output will be an EpochArray, by default False.
    manual_events : bool, optional
        If True, add manually added events from Neuroscope2 (interval will be calculated from mean event duration), by default True.

    Returns
    -------
    Union[pd.DataFrame, nel.EpochArray]
        DataFrame with the following fields:
        - start: start time of ripple
        - stop: end time of ripple
        - center: center time of ripple
        - peaks: peak time of ripple

    Notes
    -----
    * Note that basepath/basename/animal relies on specific folder structure and may be incorrect for some data structures.
    """

    # locate .mat file
    try:
        filename = glob.glob(basepath + os.sep + "*IED.events.mat")[0]
    except Exception:
        try:
            filename = glob.glob(basepath + os.sep + "*interictal_spikes.events.mat")[0]
        except Exception:
            # warnings.warn("file does not exist")
            return pd.DataFrame()

    df = pd.DataFrame()

    data = sio.loadmat(filename, simplify_cells=True)
    struct_name = list(data.keys())[-1]
    df["start"] = data[struct_name]["timestamps"][:, 0]
    df["stop"] = data[struct_name]["timestamps"][:, 1]
    df["center"] = data[struct_name]["peaks"]
    df["peaks"] = data[struct_name]["peaks"]

    # remove flagged ripples, if exist
    try:
        df.drop(
            labels=np.array(data[struct_name]["flagged"]).T - 1,
            axis=0,
            inplace=True,
        )
        df.reset_index(inplace=True)
    except Exception:
        pass

    # adding manual events
    if manual_events:
        try:
            df = _add_manual_events(df, data[struct_name]["added"])
        except Exception:
            pass

    if return_epoch_array:
        return nel.EpochArray([np.array([df.start, df.stop]).T], label="ied")

    return df


def load_dentate_spikes(
    basepath: str,
    dentate_spike_type: List[str] = ["DS1", "DS2"],
    manual_events: bool = True,
    return_epoch_array: bool = False,
) -> Union[pd.DataFrame, nel.EpochArray]:
    """
    Load info from DS*.events.mat and store within a DataFrame.

    Parameters
    ----------
    basepath : str
        Path to your session where DS*.events.mat is located.
    dentate_spike_type : List[str], optional
        List of DS types to load, by default ["DS1", "DS2"].
    manual_events : bool, optional
        If True, add manually added events from Neuroscope2 (interval will be calculated from mean event duration), by default True.
    return_epoch_array : bool, optional
        If True, the output will be an EpochArray, by default False.

    Returns
    -------
    Union[pd.DataFrame, nel.EpochArray]
        DataFrame with the following fields:
        - start: start time of DS
        - stop: end time of DS
        - peaks: peak time of DS
        - amplitude: envelope value at peak time
        - duration: DS duration
        - detectorName: the name of DS detector used
        - basepath: path name
        - basename: session id
        - animal: animal id

    Notes
    -----
    * Note that basepath/basename/animal relies on specific folder structure and may be incorrect for some data structures.
    """

    def extract_data(s_type, data, manual_events):
        # make data frame of known fields
        df = pd.DataFrame()
        df["start"] = data[s_type]["timestamps"][:, 0]
        df["stop"] = data[s_type]["timestamps"][:, 1]
        df["peaks"] = data[s_type]["peaks"]
        df["event_label"] = s_type
        df["amplitude"] = data[s_type]["amplitudes"]
        df["duration"] = data[s_type]["duration"]
        df["amplitudeUnits"] = data[s_type]["amplitudeUnits"]
        df["detectorName"] = data[s_type]["detectorinfo"]["detectorname"]
        df["ml_channel"] = data[s_type]["detectorinfo"]["ml_channel"]
        df["h_channel"] = data[s_type]["detectorinfo"]["h_channel"]

        # remove flagged ripples, if exist
        try:
            df.drop(
                labels=np.array(data[s_type]["flagged"]).T - 1,
                axis=0,
                inplace=True,
            )
            df.reset_index(inplace=True)
        except Exception:
            pass

        # adding manual events
        if manual_events:
            try:
                df = _add_manual_events(df, data[s_type]["added"])
            except Exception:
                pass
        return df

    # locate .mat file
    df = pd.DataFrame()
    for s_type in dentate_spike_type:
        filename = glob.glob(basepath + os.sep + "*" + s_type + ".events.mat")
        if len(filename) == 0:
            continue
        # load matfile
        filename = filename[0]
        data = sio.loadmat(filename, simplify_cells=True)
        # pull out data
        df = pd.concat(
            [df, extract_data(s_type, data, manual_events)], ignore_index=True
        )

    if df.shape[0] == 0:
        return df

    if return_epoch_array:
        return nel.EpochArray([np.array([df.start, df.stop]).T], label="dentate_spike")

    # get basename and animal
    normalized_path = os.path.normpath(filename)
    path_components = normalized_path.split(os.sep)
    df["basepath"] = basepath
    df["basename"] = path_components[-2]
    df["animal"] = path_components[-3]

    return df


def load_theta_rem_shift(basepath: str) -> Tuple[pd.DataFrame, dict]:
    """
    Load theta REM shift data from get_rem_shift.m.

    Parameters
    ----------
    basepath : str
        Path to your session where theta_rem_shift.mat is located.

    Returns
    -------
    Tuple[pd.DataFrame, dict]
        DataFrame with the following fields:
        - UID: unique identifier for each unit
        - circ_dist: circular distance
        - rem_shift: REM shift
        - non_rem_shift: non-REM shift
        - m_rem: mean phase locking value during REM
        - r_rem: resultant vector length during REM
        - k_rem: concentration parameter during REM
        - p_rem: p-value of phase locking during REM
        - mode_rem: mode of phase locking during REM
        - m_wake: mean phase locking value during wake
        - r_wake: resultant vector length during wake
        - k_wake: concentration parameter during wake
        - p_wake: p-value of phase locking during wake
        - mode_wake: mode of phase locking during wake

    dict
        Dictionary with phase distributions and spike phases for REM and wake states.
    """
    try:
        filename = glob.glob(basepath + os.sep + "*theta_rem_shift.mat")[0]
    except Exception:
        warnings.warn("file does not exist")
        return pd.DataFrame(), np.nan

    data = sio.loadmat(filename)

    df = pd.DataFrame()

    df["UID"] = data["rem_shift_data"]["UID"][0][0][0]
    df["circ_dist"] = data["rem_shift_data"]["circ_dist"][0][0][0]
    df["rem_shift"] = data["rem_shift_data"]["rem_shift"][0][0][0]
    df["non_rem_shift"] = data["rem_shift_data"]["non_rem_shift"][0][0][0]

    # rem metrics
    df["m_rem"] = data["rem_shift_data"]["PhaseLockingData_rem"][0][0]["phasestats"][0][
        0
    ]["m"][0][0][0]
    df["r_rem"] = data["rem_shift_data"]["PhaseLockingData_rem"][0][0]["phasestats"][0][
        0
    ]["r"][0][0][0]
    df["k_rem"] = data["rem_shift_data"]["PhaseLockingData_rem"][0][0]["phasestats"][0][
        0
    ]["k"][0][0][0]
    df["p_rem"] = data["rem_shift_data"]["PhaseLockingData_rem"][0][0]["phasestats"][0][
        0
    ]["p"][0][0][0]
    df["mode_rem"] = data["rem_shift_data"]["PhaseLockingData_rem"][0][0]["phasestats"][
        0
    ][0]["mode"][0][0][0]

    # wake metrics
    df["m_wake"] = data["rem_shift_data"]["PhaseLockingData_wake"][0][0]["phasestats"][
        0
    ][0]["m"][0][0][0]
    df["r_wake"] = data["rem_shift_data"]["PhaseLockingData_wake"][0][0]["phasestats"][
        0
    ][0]["r"][0][0][0]
    df["k_wake"] = data["rem_shift_data"]["PhaseLockingData_wake"][0][0]["phasestats"][
        0
    ][0]["k"][0][0][0]
    df["p_wake"] = data["rem_shift_data"]["PhaseLockingData_wake"][0][0]["phasestats"][
        0
    ][0]["p"][0][0][0]
    df["mode_wake"] = data["rem_shift_data"]["PhaseLockingData_wake"][0][0][
        "phasestats"
    ][0][0]["mode"][0][0][0]

    def get_distros(data, state):
        return np.vstack(data["rem_shift_data"][state][0][0]["phasedistros"][0][0].T)

    def get_spikephases(data, state):
        return data["rem_shift_data"][state][0][0]["spkphases"][0][0][0]

    # add to dictionary
    data_dict = {
        "rem": {
            "phasedistros": get_distros(data, "PhaseLockingData_rem"),
            "spkphases": get_spikephases(data, "PhaseLockingData_rem"),
        },
        "wake": {
            "phasedistros": get_distros(data, "PhaseLockingData_wake"),
            "spkphases": get_spikephases(data, "PhaseLockingData_wake"),
        },
    }

    return df, data_dict


def load_SleepState_states(
    basepath: str,
    return_epoch_array: bool = False,
    states_list: list = ["WAKEstate", "NREMstate", "REMstate", "THETA", "nonTHETA"],
) -> dict:
    """
    Loader of SleepState.states.mat.

    Parameters
    ----------
    basepath : str
        Path to the folder containing the SleepState.states.mat file.
    return_epoch_array : bool, optional
        If True, return an dict of EpochArrays, by default False.
    states_list : list, optional
        List of states to load, by default ["WAKEstate", "NREMstate", "REMstate", "THETA", "nonTHETA"].

    Returns
    -------
    dict
        Dictionary containing the contents of the SleepState.states.mat file.
    """
    filename = os.path.join(
        basepath, os.path.basename(basepath) + ".SleepState.states.mat"
    )
    if not os.path.exists(filename):
        warnings.warn("file does not exist")
        return None

    # load cell_metrics file
    data = sio.loadmat(filename)

    # get epoch id
    wake_id = (
        np.where(data["SleepState"]["idx"][0][0]["statenames"][0][0][0] == "WAKE")[0][0]
        + 1
    )
    rem_id = (
        np.where(data["SleepState"]["idx"][0][0]["statenames"][0][0][0] == "REM")[0][0]
        + 1
    )
    nrem_id = (
        np.where(data["SleepState"]["idx"][0][0]["statenames"][0][0][0] == "NREM")[0][0]
        + 1
    )

    # get states and timestamps vectors
    states = data["SleepState"]["idx"][0][0]["states"][0][0]
    timestamps = data["SleepState"]["idx"][0][0]["timestamps"][0][0]

    # set up dict
    dict_ = {
        "wake_id": wake_id,
        "rem_id": rem_id,
        "nrem_id": nrem_id,
        "states": states,
        "timestamps": timestamps,
    }

    # iter through states and add to dict
    dt = data["SleepState"]["ints"][0][0].dtype
    for dn in dt.names:
        dict_[dn] = data["SleepState"]["ints"][0][0][dn][0][0]

    if not return_epoch_array:
        return dict_
    else:
        epoch_df = load_epoch(basepath)
        # get session bounds to provide support
        session_domain = nel.EpochArray(
            [epoch_df.startTime.iloc[0], epoch_df.stopTime.iloc[-1]]
        )
        states_dict = {}
        for state in states_list:
            states_dict[state] = nel.EpochArray(
                dict_.get(state, []), domain=session_domain
            )
        return states_dict


def load_animal_behavior(
    basepath: str, alternative_file: Union[str, None] = None
) -> pd.DataFrame:
    """
    load_animal_behavior loads basename.animal.behavior.mat files created by general_behavior_file.m
    The output is a pandas data frame with [time,x,y,z,linearized,speed,acceleration,trials,epochs]

    Parameters
    ----------
    basepath : str
        Path to the session folder.
    alternative_file : Union[str, None], optional
        Alternative file name to load, by default None.

    Returns
    -------
    pd.DataFrame
        DataFrame with the following fields:
        - time: timestamps
        - x: x-coordinate
        - y: y-coordinate
        - z: z-coordinate
        - linearized: linearized position
        - speed: speed of the animal
        - acceleration: acceleration of the animal
        - trials: trial numbers
        - epochs: epoch names
        - environment: environment names
    """
    df = pd.DataFrame()

    if alternative_file is None:
        try:
            filename = glob.glob(os.path.join(basepath, "*.animal.behavior.mat"))[0]
        except Exception:
            warnings.warn("file does not exist")
            return df
    else:
        try:
            filename = glob.glob(
                os.path.join(basepath, "*" + alternative_file + ".mat")
            )[0]
        except Exception:
            warnings.warn("file does not exist")
            return df

    data = sio.loadmat(filename, simplify_cells=True)

    # add timestamps first which provide the correct shape of df
    # here, I'm naming them time, but this should be deprecated
    df["time"] = data["behavior"]["timestamps"]

    # add all other position coordinates to df (will add everything it can within position)
    for key in data["behavior"]["position"].keys():
        values = data["behavior"]["position"][key]
        if isinstance(values, (list, np.ndarray)) and len(values) == 0:
            continue
        df[key] = values

    # add other fields from behavior to df (acceleration,speed,states)
    for key in data["behavior"].keys():
        values = data["behavior"][key]
        if isinstance(values, (list, np.ndarray)) and len(values) != len(df):
            continue
        df[key] = values

    # add speed and acceleration
    if "speed" not in df.columns:
        df["speed"] = get_speed(df[["x", "y"]].values, df.time.values)
    if "acceleration" not in df.columns:  # using backward difference
        df.loc[1:, "acceleration"] = np.diff(df["speed"]) / np.diff(df["time"])
        df.loc[0, "acceleration"] = 0  # assuming no acceleration at start

    trials = data["behavior"]["trials"]
    try:
        for t in range(trials.shape[0]):
            idx = (df.time >= trials[t, 0]) & (df.time <= trials[t, 1])
            df.loc[idx, "trials"] = t
    except Exception:
        pass

    epochs = load_epoch(basepath)
    for t in range(epochs.shape[0]):
        idx = (df.time >= epochs.startTime.iloc[t]) & (
            df.time <= epochs.stopTime.iloc[t]
        )
        df.loc[idx, "epochs"] = epochs.name.iloc[t]
        df.loc[idx, "environment"] = epochs.environment.iloc[t]
    return df


def load_epoch(basepath: str) -> pd.DataFrame:
    """
    Loads epoch info from cell explorer basename.session and stores in a DataFrame.

    Parameters
    ----------
    basepath : str
        Path to the session folder.

    Returns
    -------
    pd.DataFrame
        DataFrame with the following fields:
        - name: name of the epoch
        - startTime: start time of the epoch
        - stopTime: stop time of the epoch
        - environment: environment during the epoch
        - manipulation: manipulation during the epoch
        - behavioralParadigm: behavioral paradigm during the epoch
        - stimuli: stimuli during the epoch
        - notes: notes about the epoch
        - basepath: path to the session folder
    """

    filename = os.path.join(basepath, os.path.basename(basepath) + ".session.mat")

    if not os.path.exists(filename):
        warnings.warn(f"file {filename} does not exist")
        return pd.DataFrame()

    # load file
    data = sio.loadmat(filename, simplify_cells=True)

    def add_columns(df):
        """add columns to df if they don't exist"""
        needed_columns = [
            "name",
            "startTime",
            "stopTime",
            "environment",
            "manipulation",
            "behavioralParadigm",
            "stimuli",
            "notes",
        ]
        for col in needed_columns:
            if col not in df.columns:
                df[col] = np.nan
        return df

    try:
        epoch_df = pd.DataFrame(data["session"]["epochs"])
        epoch_df = add_columns(epoch_df)
        epoch_df["basepath"] = basepath
        return epoch_df
    except Exception:
        epoch_df = pd.DataFrame([data["session"]["epochs"]])
        epoch_df = add_columns(epoch_df)
        epoch_df["basepath"] = basepath
        return epoch_df


def load_trials(basepath: str) -> pd.DataFrame:
    """
    Loads trials from cell explorer basename.session.behavioralTracking and stores in a DataFrame.

    Parameters
    ----------
    basepath : str
        Path to the session folder.

    Returns
    -------
    pd.DataFrame
        DataFrame with the following fields:
        - startTime: start time of the trial
        - stopTime: stop time of the trial
        - trialsID: ID of the trial
    """
    try:
        filename = glob.glob(os.path.join(basepath, "*.animal.behavior.mat"))[0]
    except Exception:
        warnings.warn("file does not exist")
        return pd.DataFrame()

    # load file
    data = sio.loadmat(filename, simplify_cells=True)

    try:
        df = pd.DataFrame(data=data["behavior"]["trials"])
        df.columns = ["startTime", "stopTime"]
        df["trialsID"] = data["behavior"]["trialsID"]
        return df
    except Exception:
        df = pd.DataFrame(data=[data["behavior"]["trials"]])
        df.columns = ["startTime", "stopTime"]
        df["trialsID"] = data["behavior"]["trialsID"]
        return df


def load_brain_regions(
    basepath: str, out_format: str = "dict"
) -> Union[dict, pd.DataFrame]:
    """
    Loads brain region info from cell explorer basename.session and stores in dict (default) or DataFrame.

    Parameters
    ----------
    basepath : str
        Path to the session folder.
    out_format : str, optional
        Output format, either 'dict' or 'DataFrame', by default 'dict'.

    Returns
    -------
    Union[dict, pd.DataFrame]
        Dictionary or DataFrame with brain region information.

    Examples
    -------
    >>> brainRegions = load_epoch("Z:\\Data\\GirardeauG\\Rat09\\Rat09-20140327")
    >>> print(brainRegions.keys())
    dict_keys(['CA1', 'Unknown', 'blv', 'bmp', 'ven'])
    >>> print(brainRegions['CA1'].keys())
    dict_keys(['channels', 'electrodeGroups'])
    >>> print(brainRegions['CA1']['channels'])
    [145 146 147 148 149 153 155 157 150 151 154 159 156 152 158 160 137 140
    129 136 138 134 130 132 142 143 144 141 131 139 133 135]
    >>> print(brainRegions['CA1']['electrodeGroups'])
        [17 18 19 20]
    """
    filename = glob.glob(os.path.join(basepath, "*.session.mat"))[0]
    _, _, _, shank_to_channel = loadXML(basepath)

    # load file
    data = sio.loadmat(filename)
    data = data["session"]

    brainRegions = {}
    for dn in data["brainRegions"][0][0].dtype.names:
        channels = data["brainRegions"][0][0][dn][0][0][0][0][0][0]
        try:
            electrodeGroups = data["brainRegions"][0][0][dn][0][0][0][0][1][0]
        except Exception:
            electrodeGroups = np.nan

        brainRegions[dn] = {
            "channels": channels,
            "electrodeGroups": electrodeGroups,
        }

    if out_format == "DataFrame":  # return as DataFrame
        region_df = pd.DataFrame(columns=["channels", "region"])
        for key in brainRegions.keys():
            temp_df = pd.DataFrame(brainRegions[key]["channels"], columns=["channels"])
            temp_df["region"] = key

            region_df = pd.concat([region_df, temp_df])

            # # sort channels by shank
        mapped_channels = []
        mapped_shanks = []
        for key, shank_i in enumerate(shank_to_channel.keys()):
            mapped_channels.append(shank_to_channel[key])
            mapped_shanks.append(np.repeat(shank_i, len(shank_to_channel[key])))
        #  unpack to listssss
        idx = list(chain(*mapped_channels))
        shanks = list(chain(*mapped_shanks))

        mapped_df = region_df.sort_values("channels").reset_index(drop=True).iloc[idx]
        mapped_df["shank"] = shanks
        mapped_df["channels"] = (
            mapped_df["channels"] - 1
        )  # save channel as zero-indexed

        return mapped_df.reset_index(drop=True)

    elif out_format == "dict":
        return brainRegions


def get_animal_id(basepath: str) -> str:
    """
    Return animal ID from basepath using basename.session.mat.

    Parameters
    ----------
    basepath : str
        Path to session folder.

    Returns
    -------
    str
        Animal ID.
    """
    try:
        filename = glob.glob(os.path.join(basepath, "*.session.mat"))[0]
    except Exception:
        warnings.warn("file does not exist")
        return pd.DataFrame()

    # load file
    data = sio.loadmat(filename)
    return data["session"][0][0]["animal"][0][0]["name"][0]


def add_animal_id(df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    """
    Add animal_id column to a dataframe based on the basepath column.

    Parameters
    ----------
    df : pd.core.frame.DataFrame
        Dataframe with a basepath column.

    Returns
    -------
    pd.core.frame.DataFrame
        Dataframe with an additional animal_id column.
    """
    df["animal_id"] = df.basepath.map(
        dict([(basepath, get_animal_id(basepath)) for basepath in df.basepath.unique()])
    )
    return df


def load_basic_data(basepath: str) -> Tuple[pd.DataFrame, dict, pd.DataFrame, float]:
    """
    Load basic data from the specified basepath.

    Parameters
    ----------
    basepath : str
        Path to the session folder.

    Returns
    -------
    Tuple[pd.DataFrame, dict, pd.DataFrame, float]
        - cell_metrics: DataFrame containing cell metrics.
        - data: Dictionary containing additional data.
        - ripples: DataFrame containing ripple events.
        - fs_dat: Sampling rate of the data.
    """
    try:
        nChannels, fs, fs_dat, shank_to_channel = loadXML(basepath)
    except Exception:
        fs_dat = load_extracellular_metadata(basepath).get("sr")

    ripples = load_ripples_events(basepath)
    cell_metrics, data = load_cell_metrics(basepath)

    return cell_metrics, data, ripples, fs_dat


def load_spikes(
    basepath: str,
    putativeCellType: List[str] = [],
    brainRegion: List[str] = [],
    remove_bad_unit: bool = True,
    brain_state: List[str] = [],
    other_metric: Union[str, None] = None,
    other_metric_value: Union[str, None] = None,
    support: Union[nel.EpochArray, None] = None,
) -> Tuple[Union[nel.SpikeTrainArray, None], Union[pd.DataFrame, None]]:
    """
    Load specific cells' spike times.

    Parameters
    ----------
    basepath : str
        Path to the session folder.
    putativeCellType : List[str], optional
        List of putative cell types to restrict spikes to, by default [].
    brainRegion : List[str], optional
        List of brain regions to restrict spikes to, by default [].
    remove_bad_unit : bool, optional
        If True, do not load bad cells (tagged in CE), by default True.
    brain_state : List[str], optional
        List of brain states to restrict spikes to, by default [].
    other_metric : Union[str, None], optional
        Metric to restrict spikes to, by default None.
    other_metric_value : Union[str, None], optional
        Value of the metric to restrict spikes to, by default None.
    support : Union[nel.EpochArray, None], optional
        Time support to provide, by default None.

    Returns
    -------
    Tuple[Union[nel.SpikeTrainArray, None], Union[pd.DataFrame, None]]
        Spike train array and cell metrics DataFrame.
    """
    if not isinstance(putativeCellType, list):
        putativeCellType = [putativeCellType]
    if not isinstance(brainRegion, list):
        brainRegion = [brainRegion]

    # get sample rate from xml or session
    try:
        _, _, fs_dat, _ = loadXML(basepath)
    except Exception:
        fs_dat = load_extracellular_metadata(basepath).get("sr", None)

    if fs_dat is None:
        return None, None

    # load cell metrics and spike data
    cell_metrics, data = load_cell_metrics(basepath)

    if cell_metrics is None or data is None:
        return None, None

    # put spike data into array st
    st = np.array(data["spikes"], dtype=object)

    # restrict cell metrics
    if len(putativeCellType) > 0:
        restrict_idx = []
        for cell_type in putativeCellType:
            restrict_idx.append(
                cell_metrics.putativeCellType.str.contains(cell_type).values
            )
        restrict_idx = np.any(restrict_idx, axis=0)
        cell_metrics = cell_metrics[restrict_idx]
        st = st[restrict_idx]

    if len(brainRegion) > 0:
        restrict_idx = []
        for brain_region in brainRegion:
            restrict_idx.append(
                cell_metrics.brainRegion.str.contains(brain_region).values
            )
        restrict_idx = np.any(restrict_idx, axis=0)
        cell_metrics = cell_metrics[restrict_idx]
        st = st[restrict_idx]

    # restrict cell metrics by arbitrary metric
    if other_metric is not None:
        # make other_metric_value a list if not already
        if not isinstance(other_metric, list):
            other_metric = [other_metric]
        if not isinstance(other_metric_value, list):
            other_metric_value = [other_metric_value]
        # check that other_metric_value is the same length as other_metric
        if len(other_metric) != len(other_metric_value):
            raise ValueError(
                "other_metric and other_metric_value must be of same length"
            )

        restrict_idx = []
        for metric, value in zip(other_metric, other_metric_value):
            restrict_idx.append(cell_metrics[metric].str.contains(value).values)
        restrict_idx = np.any(restrict_idx, axis=0)
        cell_metrics = cell_metrics[restrict_idx]
        st = st[restrict_idx]

    if remove_bad_unit:
        # bad units will be tagged true, so only keep false values
        restrict_idx = ~cell_metrics.bad_unit.values
        cell_metrics = cell_metrics[restrict_idx]
        st = st[restrict_idx]

    # get spike train array
    try:
        if support is not None:
            st = nel.SpikeTrainArray(timestamps=st, fs=fs_dat, support=support)
        else:
            st = nel.SpikeTrainArray(timestamps=st, fs=fs_dat)
    except Exception:  # if only single cell... should prob just skip session
        if support is not None:
            st = nel.SpikeTrainArray(timestamps=st[0], fs=fs_dat, support=support)
        else:
            st = nel.SpikeTrainArray(timestamps=st[0], fs=fs_dat)

    if len(brain_state) > 0:
        # get brain states
        brain_states = ["WAKEstate", "NREMstate", "REMstate", "THETA", "nonTHETA"]
        if brain_state not in brain_states:
            assert print("not correct brain state. Pick one", brain_states)
        else:
            state_dict = load_SleepState_states(basepath)
            state_epoch = nel.EpochArray(state_dict[brain_state])
            st = st[state_epoch]

    return st, cell_metrics


def load_deepSuperficialfromRipple(
    basepath: str, bypass_mismatch_exception: bool = False
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Load deepSuperficialfromRipple file created by classification_DeepSuperficial.m.

    Parameters
    ----------
    basepath : str
        Path to the session folder.
    bypass_mismatch_exception : bool, optional
        If True, bypass the mismatch exception, by default False.

    Returns
    -------
    Tuple[pd.DataFrame, np.ndarray, np.ndarray]
        - channel_df: DataFrame containing channel information.
        - ripple_average: Array containing average ripple traces.
        - ripple_time_axis: Array containing ripple time axis.
    """
    # locate .mat file
    file_type = "*.deepSuperficialfromRipple.channelinfo.mat"
    filename = glob.glob(basepath + os.sep + file_type)[0]

    # load matfile
    data = sio.loadmat(filename)

    channel_df = pd.DataFrame()
    name = "deepSuperficialfromRipple"

    # sometimes more channels positons will be in deepSuperficialfromRipple than in xml
    #   this is because they used channel id as an index.
    channel_df = pd.DataFrame()
    channels = np.hstack(data[name]["channel"][0][0]) * np.nan
    shanks = np.hstack(data[name]["channel"][0][0]) * np.nan

    channels_, shanks_ = zip(
        *[
            (values[0], np.tile(shank, len(values[0])))
            for shank, values in enumerate(data[name]["ripple_channels"][0][0][0])
        ]
    )
    channel_sort_idx = np.hstack(channels_) - 1
    channels[channel_sort_idx] = np.hstack(channels_)
    shanks[channel_sort_idx] = np.hstack(shanks_) + 1

    channel_df["channel"] = channels
    channel_df.loc[np.arange(len(channel_sort_idx)), "channel_sort_idx"] = (
        channel_sort_idx
    )
    channel_df["shank"] = shanks

    # add distance from pyr layer (will only be accurate if polarity rev)
    channel_df["channelDistance"] = data[name]["channelDistance"][0][0].T[0]

    # add channel class (deep or superficial)
    channelClass = []
    for item in data[name]["channelClass"][0][0]:
        try:
            channelClass.append(item[0][0])
        except Exception:
            channelClass.append("unknown")
    channel_df["channelClass"] = channelClass

    # add if shank has polarity reversal
    for shank in channel_df.shank.unique():
        if channel_df[channel_df.shank == shank].channelClass.unique().shape[0] == 2:
            channel_df.loc[channel_df.shank == shank, "polarity_reversal"] = True
        else:
            channel_df.loc[channel_df.shank == shank, "polarity_reversal"] = False

    # add ripple and sharp wave features
    labels = ["ripple_power", "ripple_amplitude", "SWR_diff", "SWR_amplitude"]
    for label in labels:
        try:
            channel_df.loc[channel_sort_idx, label] = np.hstack(
                data[name][label][0][0][0]
            )[0]
        except Exception:
            x = np.arange(len(channel_sort_idx)) * np.nan
            x[0 : len(np.hstack(data[name][label][0][0][0])[0])] = np.hstack(
                data[name][label][0][0][0]
            )[0]
            channel_df.loc[channel_sort_idx, label] = x

    # pull put avg ripple traces and ts
    ripple_time_axis = data[name]["ripple_time_axis"][0][0][0]
    ripple_average = np.ones([channel_df.shape[0], len(ripple_time_axis)]) * np.nan

    rip_map = []
    for ch, values in zip(channels_, data[name]["ripple_average"][0][0][0]):
        if values.shape[1] > 0:
            rip_map.append(values)
        else:
            rip_map.append(np.zeros([len(ripple_time_axis), len(ch)]) * np.nan)

    ripple_average[channel_sort_idx] = np.hstack(rip_map).T

    brainRegions = load_brain_regions(basepath)
    for key, value in brainRegions.items():
        if ("ca1" in key.lower()) | ("ca2" in key.lower()):
            for shank in value["electrodeGroups"]:
                channel_df.loc[channel_df.shank == shank, "ca1_shank"] = True

    if (ripple_average.shape[0] != channel_df.shape[0]) & (~bypass_mismatch_exception):
        raise Exception(
            "size mismatch "
            + str(np.hstack(ripple_average).shape[1])
            + " and "
            + str(channel_df.shape[0])
        )

    channel_df["basepath"] = basepath

    return channel_df, ripple_average, ripple_time_axis


def load_mua_events(basepath: str) -> pd.DataFrame:
    """
    Loads the MUA data from the basepath.
    Meant to load .mat file created by find_HSE.m.

    Parameters
    ----------
    basepath : str
        The path to the folder containing the MUA data.

    Returns
    -------
    pd.DataFrame
        The pandas DataFrame containing the MUA data.

    TODO
    ----
    If none exist in basepath, create one.
    """

    # locate .mat file
    try:
        filename = glob.glob(basepath + os.sep + "*mua_ca1_pyr.events.mat")[0]
    except Exception:
        warnings.warn("file does not exist")
        return pd.DataFrame()

    # load matfile
    data = sio.loadmat(filename)

    # pull out and package data
    df = pd.DataFrame()
    df["start"] = data["HSE"]["timestamps"][0][0][:, 0]
    df["stop"] = data["HSE"]["timestamps"][0][0][:, 1]
    df["peaks"] = data["HSE"]["peaks"][0][0]
    df["center"] = data["HSE"]["center"][0][0]
    df["duration"] = data["HSE"]["duration"][0][0]
    df["amplitude"] = data["HSE"]["amplitudes"][0][0]
    df["amplitudeUnits"] = data["HSE"]["amplitudeUnits"][0][0][0]
    df["detectorName"] = data["HSE"]["detectorinfo"][0][0]["detectorname"][0][0][0]

    # get basename and animal
    normalized_path = os.path.normpath(filename)
    path_components = normalized_path.split(os.sep)
    df["basepath"] = basepath
    df["basename"] = path_components[-2]
    df["animal"] = path_components[-3]

    return df


def load_manipulation(
    basepath: str,
    struct_name: Union[str, None] = None,
    return_epoch_array: bool = True,
    merge_gap: Union[int, None] = None,
) -> Union[pd.DataFrame, nel.EpochArray]:
    """
    Loads the data from the basename.eventName.manipulations.mat file and returns a pandas dataframe.

    file structure defined here:
        https://cellexplorer.org/datastructure/data-structure-and-format/#manipulations

    Parameters
    ----------
    basepath : str
        Path to the basename.eventName.manipulations.mat file.
    struct_name : Union[str, None], optional
        Name of the structure in the mat file to load. If None, loads all the manipulation files, by default None.
    return_epoch_array : bool, optional
        If True, returns only the epoch array, by default True.
    merge_gap : Union[int, None], optional
        If not None, merges the epochs that are separated by less than merge_gap (sec). return_epoch_array must be True, by default None.

    Returns
    -------
    Union[pd.DataFrame, nel.EpochArray]
        DataFrame or EpochArray with the manipulation data.

    Examples
    -------
    >>> basepath = r"Z:\Data\Can\OML22\day8"
    >>> df_manipulation = load_manipulation(basepath, struct_name="optoStim", return_epoch_array=False)
    >>> df_manipulation.head(2)

    .. table:: Manipulation Data
        :widths: auto

        ====== ========== ========== ========== ========== ========== ========================
            start      stop       peaks      center     duration    amplitude     amplitudeUnits
        ====== ========== ========== ========== ========== ========== ========================
        8426.83650  8426.84845  8426.842475  8426.842475  0.01195   19651       pulse_respect_baseline
        8426.85245  8426.86745  8426.859950  8426.859950  0.01500   17516       pulse_respect_baseline
        ====== ========== ========== ========== ========== ========== ========================

    >>> basepath = r"Z:\Data\Can\OML22\day8"
    >>> df_manipulation = load_manipulation(basepath, struct_name="optoStim", return_epoch_array=True)
    >>> df_manipulation

    <EpochArray at 0x1faba577520: 5,774 epochs> of length 1:25:656 minutes
    """
    try:
        if struct_name is None:
            filename = glob.glob(basepath + os.sep + "*manipulation.mat")
            print(filename)
            if len(filename) > 1:
                raise ValueError(
                    "multi-file not implemented yet...than one manipulation file found"
                )
            filename = filename[0]
        else:
            filename = glob.glob(
                basepath + os.sep + "*" + struct_name + ".manipulation.mat"
            )[0]
    except Exception:
        return None
    # load matfile
    data = sio.loadmat(filename)

    if struct_name is None:
        struct_name = list(data.keys())[-1]

    df = pd.DataFrame()
    df["start"] = data[struct_name]["timestamps"][0][0][:, 0]
    df["stop"] = data[struct_name]["timestamps"][0][0][:, 1]
    df["peaks"] = data[struct_name]["peaks"][0][0]
    df["center"] = data[struct_name]["center"][0][0]
    df["duration"] = data[struct_name]["duration"][0][0]
    df["amplitude"] = data[struct_name]["amplitude"][0][0]
    df["amplitudeUnits"] = data[struct_name]["amplitudeUnits"][0][0][0]

    # extract event label names
    eventIDlabels = []
    for name in data[struct_name]["eventIDlabels"][0][0][0]:
        eventIDlabels.append(name[0])

    # extract numeric category labels associated with label names
    eventID = np.array(data[struct_name]["eventID"][0][0]).ravel()

    # add eventIDlabels and eventID to df
    for ev_label, ev_num in zip(eventIDlabels, np.unique(eventID)):
        df.loc[eventID == ev_num, "ev_label"] = ev_label

    if return_epoch_array:
        # get session epochs to add support for epochs
        epoch_df = load_epoch(basepath)
        # get session bounds to provide support
        session_bounds = nel.EpochArray(
            [epoch_df.startTime.iloc[0], epoch_df.stopTime.iloc[-1]]
        )
        # if many types of manipulations, add them to dictinary
        if df.ev_label.unique().size > 1:
            manipulation_epoch = {}
            for label in df.ev_label.unique():
                manipulation_epoch_ = nel.EpochArray(
                    np.array(
                        [
                            df[df.ev_label == label]["start"],
                            df[df.ev_label == label]["stop"],
                        ]
                    ).T,
                    domain=session_bounds,
                )
                if merge_gap is not None:
                    manipulation_epoch_ = manipulation_epoch_.merge(gap=merge_gap)

                manipulation_epoch[label] = manipulation_epoch_
        else:
            manipulation_epoch = nel.EpochArray(
                np.array([df["start"], df["stop"]]).T, domain=session_bounds
            )
            if merge_gap is not None:
                manipulation_epoch = manipulation_epoch.merge(gap=merge_gap)

        return manipulation_epoch
    else:
        return df


def load_channel_tags(basepath: str) -> dict:
    """
    Load channel tags from session file.

    Parameters
    ----------
    basepath : str
        The base path to the directory containing the session file.

    Returns
    -------
    dict
        A dictionary of channel tags from the session file.
    """
    filename = glob.glob(os.path.join(basepath, "*.session.mat"))[0]
    data = sio.loadmat(filename, simplify_cells=True)
    return data["session"]["channelTags"]


def load_extracellular_metadata(basepath: str) -> dict:
    """
    Load extracellular metadata from session file.

    Parameters
    ----------
    basepath : str
        The base path to the directory containing the session file.

    Returns
    -------
    dict
        A dictionary of extracellular metadata from the session file.
    """
    filename = os.path.join(basepath, os.path.basename(basepath) + ".session.mat")
    # check if filename exist
    if not os.path.exists(filename):
        return {}
    data = sio.loadmat(filename, simplify_cells=True)
    return data["session"]["extracellular"]


def load_probe_layout(basepath: str) -> pd.DataFrame:
    """
    Load electrode coordinates and grouping from the session.extracellular.mat file.

    Parameters
    ----------
    basepath : str
        Path to the session folder.

    Returns
    -------
    probe_layout : pd.DataFrame
        DataFrame with x, y coordinates and shank number.
    """

    # load session file
    filename = glob.glob(os.path.join(basepath, "*.session.mat"))[0]

    # load file
    data = sio.loadmat(filename, simplify_cells=True)
    x = data["session"]['extracellular']['chanCoords']['x']
    y = data["session"]['extracellular']['chanCoords']['y']
    electrode_groups = data["session"]["extracellular"]["electrodeGroups"]['channels']

    # for each group in electrodeGroups
    mapped_shanks = []
    mapped_channels = []
    for shank_i in np.arange(len(electrode_groups)):
        mapped_channels.append(
            electrode_groups[shank_i] - 1
        )  # -1 to make 0 indexed
        mapped_shanks.append(np.repeat(shank_i, len(electrode_groups[shank_i])))

    #  unpack to lists
    mapped_channels = list(chain(*mapped_channels))
    shanks = list(chain(*mapped_shanks))

    # get shank in same dimension as channels
    shanks = np.expand_dims(shanks, axis=1)

    probe_layout = (
        pd.DataFrame({"x": x.flatten(), "y": y.flatten()})
        .iloc[mapped_channels]
        .reset_index(drop=True)
    )
    probe_layout["shank"] = shanks
    probe_layout["channels"] = mapped_channels

    return probe_layout


def load_emg(
    basepath: str, threshold: float = 0.9
) -> Tuple[nel.AnalogSignalArray, nel.EpochArray, nel.EpochArray]:
    """
    Load EMG data from basename.EMGFromLFP.LFP.mat.

    Parameters
    ----------
    basepath : str
        Path to the session folder.
    threshold : float, optional
        Threshold for high epochs (low will be < threshold). Default is 0.9.

    Returns
    -------
    emg : nel.AnalogSignalArray
        EMG data.
    high_emg_epoch : nel.EpochArray
        High EMG epochs.
    low_emg_epoch : nel.EpochArray
        Low EMG epochs.
    """
    # locate .mat file
    filename = os.path.join(
        basepath, os.path.basename(basepath) + ".EMGFromLFP.LFP.mat"
    )

    # load matfile
    data = sio.loadmat(filename, simplify_cells=True)

    # put emg data into AnalogSignalArray
    emg = nel.AnalogSignalArray(
        data=data["EMGFromLFP"]["data"], timestamps=data["EMGFromLFP"]["timestamps"]
    )

    # get high and low emg epochs
    high_emg_epoch = find_interval(emg.data.flatten() > threshold)
    high_emg_epoch = nel.EpochArray(emg.abscissa_vals[high_emg_epoch])

    low_emg_epoch = find_interval(emg.data.flatten() < threshold)
    low_emg_epoch = nel.EpochArray(emg.abscissa_vals[low_emg_epoch])

    return emg, high_emg_epoch, low_emg_epoch


def load_events(basepath: str, epoch_name: str) -> Union[nel.EpochArray, None]:
    """
    Load events from basename.epoch_name.events.mat.

    Parameters
    ----------
    basepath : str
        Path to the session folder.
    epoch_name : str
        Name of epoch to load.

    Returns
    -------
    events : nel.EpochArray or None
        Events, or None if the file does not exist.
    """
    filename = os.path.join(
        basepath, os.path.basename(basepath) + "." + epoch_name + ".events.mat"
    )
    # check if filename exist
    if not os.path.exists(filename):
        return None

    data = sio.loadmat(filename, simplify_cells=True)
    return nel.EpochArray(data[epoch_name]["timestamps"])
