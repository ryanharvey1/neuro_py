import os
import sys
from copy import deepcopy
from typing import Union

import pickle
import nelpy as nel
import numpy as np
from scipy.io import savemat
from scipy.signal import cheby2, filtfilt, find_peaks

from neuro_py.io import loading
from neuro_py.lfp.preprocessing import clean_lfp
from neuro_py.process.intervals import find_intersecting_intervals, in_intervals
from neuro_py.process.peri_event import event_triggered_average_fast


class DetectDS(object):
    """
    Class for detecting dentate spikes

    Parameters
    ----------
    basepath : str
        Path to the folder containing the data
    hilus_ch : int
        Channel number of the hilus signal (0 indexing)
    mol_ch : int
        Channel number of the mol signal (0 indexing)
    noise_ch : int, optional
        Channel number of the noise signal or signal far from dentate (0 indexing)
    lowcut : float, optional
        Low cut frequency for the signal filter
    highcut : float, optional
        High cut frequency for the signal filter
    filter_signal_bool : bool, optional
        If True, the signal will be filtered
    primary_threshold : float, optional
        Primary threshold for detecting the dentate spikes (difference method only)
    secondary_threshold : float, optional
        Secondary threshold for detecting the dentate spikes (difference method only)
    primary_thres_mol : float, optional
        Primary threshold for detecting the dentate spikes in the mol signal
    primary_thres_hilus : float, optional
        Primary threshold for detecting the dentate spikes in the hilus signal
    min_duration : float, optional
        Minimum duration of the dentate spikes
    max_duration : float, optional
        Maximum duration of the dentate spikes
    filter_order : int, optional
        Order of the filter
    filter_rs : int, optional
        Resonance frequency of the filter
    method : str, optional
        Method for detecting the dentate spikes.
        "difference" for detecting the dentate spikes by difference between the hilus and mol signal
        "seperately" for detecting the dentate spikes by the hilus and mol signal separately
    clean_lfp : bool, optional
        If True, the LFP signal will be cleaned
    emg_threshold : float, optional
        Threshold for the EMG signal to remove dentate spikes


    Attributes
    ----------
    lfp : nelpy.AnalogSignalArray
        LFP signal
    filtered_lfp : nelpy.AnalogSignalArray
        Filtered LFP signal
    mol_hilus_diff : nelpy.AnalogSignalArray
        Difference between the hilus and mol signal
    ds_epoch : nelpy.EpochArray
        EpochArray with the dentate spikes
    peak_val : np.ndarray
        Peak value of the dentate spikes


    Methods
    -------
    load_lfp()
        Load the LFP signal
    filter_signal()
        Filter the LFP signal
    get_filtered_lfp()
        Get the filtered LFP signal
    get_lfp_diff()
        Get the difference between the hilus and mol signal
    detect_ds_difference()
        Detect the dentate spikes by difference between the hilus and mol signal
    detect_ds_seperately()
        Detect the dentate spikes by the hilus and mol signal separately
    save_ds_epoch()
        Save the dentate spikes as an EpochArray

    Examples
    --------
    In IDE or python console

    >>> from ds_swr.detection.detect_dentate_spike import DetectDS
    >>> from neuro_py.io import loading
    >>> channel_tags = loading.load_channel_tags(basepath)
    >>> dds = DetectDS(
        basepath,
        channel_tags["hilus"]["channels"] - 1,
        channel_tags["mol"]["channels"] - 1
    )
    >>> dds.detect_ds()
    >>> dds.save_ds_epoch()
    >>> dds
    <DetectDS at 0x17fe787c640: dentate spikes 5,769> of length 1:11:257 minutes


    In command line

    >>> python detect_dentate_spike.py Z:\Data\Can\OML22\day20
    """

    def __init__(
        self,
        basepath: str,
        hilus_ch: int,
        mol_ch: int,
        noise_ch: Union[int, None] = None,
        lowcut: int = 10,
        highcut: int = 250,
        filter_signal_bool: bool = True,
        primary_threshold: Union[int, float] = 5,
        primary_thres_mol: Union[int, float] = 2,
        primary_thres_hilus: Union[int, float] = 5,
        min_duration: float = 0.005,
        max_duration: float = 0.05,
        filter_order: int = 4,
        filter_rs: int = 20,
        method: str = "seperately",
        clean_lfp: bool = False,
        emg_threshold: float = 0.9,
    ) -> None:
        # adding all the parameters to the class
        self.__dict__.update(locals())
        del self.__dict__["self"]
        # setting the type name
        self.type_name = self.__class__.__name__
        self.get_xml_data()

    def get_xml_data(self):
        """
        Load the XML file to get the number of channels, sampling frequency and shank to channel mapping
        """
        nChannels, fs, fs_dat, shank_to_channel = loading.loadXML(self.basepath)
        self.nChannels = nChannels
        self.fs = fs
        self.fs_dat = fs_dat
        self.shank_to_channel = shank_to_channel

    def load_lfp(self):
        """
        Load the LFP signal
        """

        lfp, timestep = loading.loadLFP(
            self.basepath,
            n_channels=self.nChannels,
            frequency=self.fs,
            ext="lfp",
        )

        if self.noise_ch is None:
            channels = [self.hilus_ch, self.mol_ch]
        else:
            channels = [self.hilus_ch, self.mol_ch, self.noise_ch]

        self.lfp = nel.AnalogSignalArray(
            data=lfp[:, channels].T,
            timestamps=timestep,
            fs=self.fs,
            support=nel.EpochArray(np.array([min(timestep), max(timestep)])),
        )
        if self.clean_lfp:
            self.lfp._data = np.array(
                [
                    clean_lfp(self.lfp.signals[0]),
                    clean_lfp(self.lfp.signals[1]),
                ]
            )

    def filter_signal(self):
        """
        Filter the LFP signal

        Returns
        -------
        np.ndarray
            Filtered LFP signal
        """
        if not hasattr(self, "lfp"):
            self.load_lfp()

        b, a = cheby2(
            self.filter_order,
            self.filter_rs,
            [self.lowcut, self.highcut],
            fs=self.fs,
            btype="bandpass",
        )
        return filtfilt(b, a, self.lfp.data)

    def get_filtered_lfp(self):
        if not hasattr(self, "lfp"):
            self.load_lfp()

        self.filtered_lfp = deepcopy(self.lfp)
        self.filtered_lfp._data = self.filter_signal()

    def get_lfp_diff(self):
        if self.filter_signal_bool:
            y = self.filter_signal()
        else:
            if not hasattr(self, "lfp"):
                self.load_lfp()
            y = self.lfp.data

        self.mol_hilus_diff = nel.AnalogSignalArray(
            data=y[0, :] - y[1, :],
            timestamps=self.lfp.abscissa_vals,
            fs=self.fs,
            support=nel.EpochArray(
                np.array([min(self.lfp.abscissa_vals), max(self.lfp.abscissa_vals)])
            ),
        )

    def detect_ds_difference(self):
        if not hasattr(self, "mol_hilus_diff"):
            self.get_lfp_diff()

        PrimaryThreshold = (
            self.mol_hilus_diff.mean()
            + self.primary_threshold * self.mol_hilus_diff.std()
        )
        SecondaryThreshold = (
            self.mol_hilus_diff.mean()
            + self.secondary_threshold * self.mol_hilus_diff.std()
        )
        bounds, self.peak_val, _ = nel.utils.get_events_boundaries(
            x=self.mol_hilus_diff.data,
            PrimaryThreshold=PrimaryThreshold,
            SecondaryThreshold=SecondaryThreshold,
            minThresholdLength=0,
            minLength=self.min_duration,
            maxLength=self.max_duration,
            ds=1 / self.mol_hilus_diff.fs,
        )
        # convert bounds to time in seconds
        timebounds = self.mol_hilus_diff.time[bounds]
        # add 1/fs to stops for open interval
        timebounds[:, 1] += 1 / self.mol_hilus_diff.fs
        # create EpochArray with bounds
        self.ds_epoch = nel.EpochArray(timebounds)

        # remove ds in high emg
        _, high_emg_epoch, _ = loading.load_emg(self.basepath, self.emg_threshold)
        if not high_emg_epoch.isempty:
            idx = find_intersecting_intervals(self.ds_epoch, high_emg_epoch)
            self.ds_epoch._data = self.ds_epoch.data[~idx]
            self.peak_val = self.peak_val[~idx]

    def detect_ds_seperately(self):
        if not hasattr(self, "filtered_lfp"):
            self.get_filtered_lfp()

        # min and max time width of ds (converted to samples for find_peaks)
        time_widths = [
            int(self.min_duration * self.filtered_lfp.fs),
            int(self.max_duration * self.filtered_lfp.fs),
        ]

        # detect ds in hilus
        PrimaryThreshold = (
            self.filtered_lfp.data[0, :].mean()
            + self.primary_thres_hilus * self.filtered_lfp.data[0, :].std()
        )

        peaks, properties = find_peaks(
            self.filtered_lfp.data[0, :],
            height=PrimaryThreshold,
            width=time_widths,
        )
        self.peaks = peaks / self.filtered_lfp.fs
        self.peak_val = properties["peak_heights"]

        # create EpochArray with bounds
        hilus_epoch = nel.EpochArray(
            np.array([properties["left_ips"], properties["right_ips"]]).T
            / self.filtered_lfp.fs
        )

        # detect ds in mol
        PrimaryThreshold = (
            self.filtered_lfp.data[1, :].mean()
            + self.primary_thres_mol * self.filtered_lfp.data[1, :].std()
        )

        peaks, properties = find_peaks(
            -self.filtered_lfp.data[1, :],
            height=PrimaryThreshold,
            width=time_widths,
        )
        mol_epoch_peak = peaks / self.filtered_lfp.fs
        # create EpochArray with bounds
        mol_epoch = nel.EpochArray(
            np.array([properties["left_ips"], properties["right_ips"]]).T
            / self.filtered_lfp.fs
        )

        # detect ds in noise channel
        if self.noise_ch is not None:
            PrimaryThreshold = (
                self.filtered_lfp.data[2, :].mean()
                + self.primary_thres_hilus * self.filtered_lfp.data[2, :].std()
            )

            peaks, properties = find_peaks(
                self.filtered_lfp.data[2, :],
                height=PrimaryThreshold,
                width=time_widths,
            )

            # create EpochArray with bounds
            noise_epoch = nel.EpochArray(
                np.array([properties["left_ips"], properties["right_ips"]]).T
                / self.filtered_lfp.fs
            )

        # remove hilus spikes that are not overlapping with mol spikes
        # first, find mol peaks that are within hilus epoch
        idx = in_intervals(mol_epoch_peak, hilus_epoch.data)
        mol_epoch._data = mol_epoch.data[idx]

        overlap = find_intersecting_intervals(
            hilus_epoch, mol_epoch, return_indices=True
        )
        self.ds_epoch = nel.EpochArray(hilus_epoch.data[overlap])
        self.peak_val = self.peak_val[overlap]
        self.peaks = self.peaks[overlap]

        # remove dentate spikes that are overlapping with noise spikes
        if self.noise_ch is not None:
            overlap = find_intersecting_intervals(
                self.ds_epoch, noise_epoch, return_indices=True
            )
            self.ds_epoch = nel.EpochArray(self.ds_epoch.data[~overlap])
            self.peak_val = self.peak_val[~overlap]
            self.peaks = self.peaks[~overlap]

        # remove ds in high emg
        _, high_emg_epoch, _ = loading.load_emg(self.basepath, self.emg_threshold)
        if not high_emg_epoch.isempty:
            idx = find_intersecting_intervals(self.ds_epoch, high_emg_epoch)
            self.ds_epoch._data = self.ds_epoch.data[~idx]
            self.peak_val = self.peak_val[~idx]
            self.peaks = self.peaks[~idx]

    def detect_ds(self):
        """
        Detect the dentate spikes based on the method provided
        """
        if self.method == "difference":
            # deprecated
            raise NotImplementedError
            # self.detect_ds_difference()
        elif self.method == "seperately":
            self.detect_ds_seperately()
        else:
            raise ValueError(f"Method {self.method} not recognized")

    def save_ds_epoch(self):
        """
        Save the dentate spikes as a cellexplorer mat file
        """

        filename = os.path.join(
            self.basepath, os.path.basename(self.basepath) + ".DS2.events.mat"
        )
        data = {}
        data["DS2"] = {}
        data["DS2"]["detectorinfo"] = {}
        data["DS2"]["timestamps"] = self.ds_epoch.data
        data["DS2"]["peaks"] = self.peaks
        data["DS2"]["amplitudes"] = self.peak_val.T
        data["DS2"]["amplitudeUnits"] = "mV"
        data["DS2"]["eventID"] = []
        data["DS2"]["eventIDlabels"] = []
        data["DS2"]["eventIDbinary"] = []
        data["DS2"]["duration"] = self.ds_epoch.durations.T
        data["DS2"]["center"] = np.median(self.ds_epoch.data, axis=1).T
        data["DS2"]["detectorinfo"]["detectorname"] = "DetectDS"
        data["DS2"]["detectorinfo"]["detectionparms"] = []
        data["DS2"]["detectorinfo"]["detectionintervals"] = []
        data["DS2"]["detectorinfo"]["ml_channel"] = self.mol_ch
        data["DS2"]["detectorinfo"]["h_channel"] = self.hilus_ch
        if self.noise_ch is not None:
            data["DS2"]["detectorinfo"]["noise_channel"] = self.noise_ch

        savemat(filename, data, long_field_names=True)

    def get_average_trace(self, shank=None, window=[-0.15, 0.15]):
        """
        Get the average LFP trace around the dentate spikes

        Parameters
        ----------
        shank : int, optional
            Shank number of the hilus signal
        window : list, optional
            Window around the dentate spikes

        Returns
        -------
        np.ndarray
            Average LFP trace around the dentate spikes
        np.ndarray
            Time lags around the dentate spikes
        """

        lfp, _ = loading.loadLFP(
            self.basepath,
            n_channels=self.nChannels,
            frequency=self.fs,
            ext="lfp",
        )

        if shank is None:
            hilus_shank = [
                k for k, v in self.shank_to_channel.items() if self.hilus_ch in v
            ][0]

        ds_average, time_lags = event_triggered_average_fast(
            signal=lfp[:, self.shank_to_channel[hilus_shank]].T,
            events=self.ds_epoch.starts,
            sampling_rate=self.fs,
            window=window,
            return_average=True,
        )
        return ds_average, time_lags

    def plot(self, ax=None, window=[-0.15, 0.15], channel_offset=9e4):
        """
        Plot the average LFP trace around the dentate spikes

        Parameters
        ----------
        ax : matplotlib.axes._subplots.AxesSubplot, optional
            Axis to plot the average LFP trace
        window : list, optional
            Window around the dentate spikes
        channel_offset : float, optional
            Offset between the channels

        Returns
        -------
        matplotlib.axes._subplots.AxesSubplot
            Axis with the average LFP trace
        """

        import matplotlib.pyplot as plt

        ds_average, time_lags = self.get_average_trace(window=window)

        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 10))

        ax.plot(
            time_lags,
            ds_average.T - np.linspace(0, channel_offset, ds_average.shape[0]),
            alpha=0.75,
        )
        return ax

    def _detach(self):
        """Detach the data from the object to allow for pickling"""
        self.filtered_lfp = None
        self.lfp = None
        self.mol_hilus_diff = None

    def save(self, filename: str):
        """
        Save the DetectDS object as a pickle file

        Parameters
        ----------
        filename : str
            Path to the file where the DetectDS object will be saved

        Returns
        -------
        None

        """
        self._detach()
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename: str):
        """
        Load a DetectDS object from a pickle file

        Parameters
        ----------
        filename : str
            Path to the file where the DetectDS object is saved

        Returns
        -------
        DetectDS
            The loaded DetectDS object

        """
        with open(filename, "rb") as f:
            return pickle.load(f)

    def __repr__(self) -> str:
        address_str = " at " + str(hex(id(self)))

        if not hasattr(self, "ds_epoch"):
            return "<%s%s>" % (self.type_name, address_str)

        if self.ds_epoch.isempty:
            return "<%s%s: empty>" % self.type_name

        dentate_spikes = f"dentate spikes {self.ds_epoch.n_intervals}"
        dstr = f"of length {self.ds_epoch.length}"

        return "<%s%s: %s> %s" % (self.type_name, address_str, dentate_spikes, dstr)

    def __str__(self) -> str:
        return self.__repr__()

    def __len__(self) -> int:
        if not hasattr(self, "ds_epoch"):
            return 0
        return self.ds_epoch.n_intervals

    def __getitem__(self, key):
        if not hasattr(self, "ds_epoch"):
            raise IndexError("No dentate spikes detected yet")
        return self.ds_epoch[key]

    def __iter__(self):
        if not hasattr(self, "ds_epoch"):
            raise IndexError("No dentate spikes detected yet")
        return iter(self.ds_epoch)

    def __contains__(self, item):
        if not hasattr(self, "ds_epoch"):
            raise IndexError("No dentate spikes detected yet")
        return item in self.ds_epoch


if __name__ == "__main__":
    basepath = sys.argv[1]
    channel_tags = loading.load_channel_tags(basepath)

    dds = DetectDS(
        basepath,
        channel_tags["hilus"]["channels"] - 1,
        channel_tags["mol"]["channels"] - 1,
    )

    dds.detect_ds()
    dds.save_ds_epoch()
