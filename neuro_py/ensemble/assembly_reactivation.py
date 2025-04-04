import copy
import logging
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import nelpy as nel
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from neuro_py.ensemble import assembly
from neuro_py.io import loading
from neuro_py.session.locate_epochs import compress_repeated_epochs, find_pre_task_post


logging.getLogger().setLevel(logging.ERROR)


class AssemblyReact:
    """
    Class for running assembly reactivation analysis

    Core assembly methods come from assembly.py by Vítor Lopes dos Santos
        https://doi.org/10.1016/j.jneumeth.2013.04.010

    Parameters
    ----------
    basepath : str
        Path to the session folder
    brainRegion : str
        Brain region to restrict to. Can be multi ex. "CA1|CA2"
    putativeCellType : str
        Cell type to restrict to
    weight_dt : float
        Time resolution of the weight matrix
    z_mat_dt : float
        Time resolution of the z matrix
    method : str
        Defines how to extract assembly patterns (ica,pca).
    nullhyp : str
        Defines how to generate statistical threshold for assembly detection (bin,circ,mp).
    nshu : int
        Number of shuffles for bin and circ null hypothesis.
    percentile : int
        Percentile for mp null hypothesis.
    tracywidom : bool
        If true, uses Tracy-Widom distribution for mp null hypothesis.

    Attributes
    ----------
    st : nelpy.SpikeTrainArray
        Spike train
    cell_metrics : pd.DataFrame
        Cell metrics
    ripples : nelpy.EpochArray
        Ripples
    patterns : np.ndarray
        Assembly patterns
    assembly_act : nelpy.AnalogSignalArray
        Assembly activity

    Methods
    -------
    load_data()
        Load data (st, ripples, epochs)
    restrict_to_epoch(epoch)
        Restrict to a specific epoch
    get_z_mat(st)
        Get z matrix
    get_weights(epoch=None)
        Get assembly weights
    get_assembly_act(epoch=None)
        Get assembly activity
    n_assemblies()
        Number of detected assemblies
    isempty()
        Check if empty
    copy()
        Returns copy of class
    plot()
        Stem plot of assembly weights
    find_members()
        Find members of an assembly


    Examples
    --------
    >>> # create the object assembly_react
    >>> assembly_react = assembly_reactivation.AssemblyReact(
    ...    basepath=basepath,
    ...    )

    >>> # load need data (spikes, ripples, epochs)
    >>> assembly_react.load_data()

    >>> # detect assemblies
    >>> assembly_react.get_weights()

    >>> # visually inspect weights for each assembly
    >>> assembly_react.plot()

    >>> # compute time resolved signal for each assembly
    >>> assembly_act = assembly_react.get_assembly_act()

    >>> # locate members of assemblies
    >>> assembly_members = assembly_react.find_members()

    """

    def __init__(
        self,
        basepath: Union[str, None] = None,
        brainRegion: str = "CA1",
        putativeCellType: str = "Pyramidal Cell",
        weight_dt: float = 0.025,
        z_mat_dt: float = 0.002,
        method: str = "ica",
        nullhyp: str = "mp",
        nshu: int = 1000,
        percentile: int = 99,
        tracywidom: bool = False,
        whiten: str = "unit-variance",
    ):
        self.basepath = basepath
        self.brainRegion = brainRegion
        self.putativeCellType = putativeCellType
        self.weight_dt = weight_dt
        self.z_mat_dt = z_mat_dt
        self.method = method
        self.nullhyp = nullhyp
        self.nshu = nshu
        self.percentile = percentile
        self.tracywidom = tracywidom
        self.whiten = whiten
        self.type_name = self.__class__.__name__

    def add_st(self, st: nel.SpikeTrainArray) -> None:
        self.st = st

    def add_ripples(self, ripples: nel.EpochArray) -> None:
        self.ripples = ripples

    def add_epoch_df(self, epoch_df: pd.DataFrame) -> None:
        self.epoch_df = epoch_df

    def load_spikes(self) -> None:
        """
        loads spikes from the session folder
        """
        self.st, self.cell_metrics = loading.load_spikes(
            self.basepath,
            brainRegion=self.brainRegion,
            putativeCellType=self.putativeCellType,
            support=self.time_support,
        )

    def load_ripples(self) -> None:
        """
        loads ripples from the session folder
        """
        ripples = loading.load_ripples_events(self.basepath)
        self.ripples = nel.EpochArray(
            [np.array([ripples.start, ripples.stop]).T], domain=self.time_support
        )

    def load_epoch(self) -> None:
        """
        loads epochs from the session folder
        """
        epoch_df = loading.load_epoch(self.basepath)
        epoch_df = compress_repeated_epochs(epoch_df)
        self.time_support = nel.EpochArray(
            [epoch_df.iloc[0].startTime, epoch_df.iloc[-1].stopTime]
        )
        self.epochs = nel.EpochArray(
            [np.array([epoch_df.startTime, epoch_df.stopTime]).T],
            domain=self.time_support,
        )
        self.epoch_df = epoch_df

    def load_data(self) -> None:
        """
        loads data (spikes,ripples,epochs) from the session folder
        """
        self.load_epoch()
        self.load_spikes()
        self.load_ripples()

    def restrict_epochs_to_pre_task_post(self) -> None:
        """
        Restricts the epochs to the specified epochs
        """
        # fetch data
        epoch_df = loading.load_epoch(self.basepath)
        # compress back to back sleep epochs (an issue further up the pipeline)
        epoch_df = compress_repeated_epochs(epoch_df)
        # restrict to pre task post epochs
        idx = find_pre_task_post(epoch_df.environment)
        self.epoch_df = epoch_df[idx[0]]
        # convert to epoch array and add to object
        self.epochs = nel.EpochArray(
            [np.array([self.epoch_df.startTime, self.epoch_df.stopTime]).T],
            label="session_epochs",
            domain=self.time_support,
        )

    def restrict_to_epoch(self, epoch) -> None:
        """
        Restricts the spike data to a specific epoch.

        Parameters
        ----------
        epoch : nel.EpochArray
            The epoch to restrict to.
        """
        self.st_resticted = self.st[epoch]

    def get_z_mat(self, st: nel.SpikeTrainArray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get z matrix.

        Parameters
        ----------
        st : nel.SpikeTrainArray
            Spike train array.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Z-scored binned spike train and bin centers.
        """
        # binning the spike train
        z_t = st.bin(ds=self.z_mat_dt)
        # gaussian kernel to match the bin-size used to identify the assembly patterns
        sigma = self.weight_dt / np.sqrt(int(1000 * self.weight_dt / 2))
        z_t.smooth(sigma=sigma, inplace=True)
        # zscore the z matrix
        z_scored_bst = stats.zscore(z_t.data, axis=1)
        # make sure there are no nans, important as strengths will all be nan otherwise
        z_scored_bst[np.isnan(z_scored_bst).any(axis=1)] = 0

        return z_scored_bst, z_t.bin_centers

    def get_weights(self, epoch: Optional[nel.EpochArray] = None) -> None:
        """
        Gets the assembly weights.

        Parameters
        ----------
        epoch : nel.EpochArray, optional
            The epoch to restrict to, by default None.
        """

        # check if st has any neurons
        if self.st.isempty:
            self.patterns = None
            return

        if epoch is not None:
            bst = self.st[epoch].bin(ds=self.weight_dt).data
        else:
            bst = self.st.bin(ds=self.weight_dt).data

        if (bst == 0).all():
            self.patterns = None
            return
        else:
            patterns, _, _ = assembly.runPatterns(
                bst,
                method=self.method,
                nullhyp=self.nullhyp,
                nshu=self.nshu,
                percentile=self.percentile,
                tracywidom=self.tracywidom,
                whiten=self.whiten,
            )

            if patterns is None: 
                self.patterns = None
                return 
            
            # flip patterns to have positive max
            self.patterns = np.array(
                [
                    (
                        patterns[i, :]
                        if patterns[i, np.argmax(np.abs(patterns[i, :]))] > 0
                        else -patterns[i, :]
                    )
                    for i in range(patterns.shape[0])
                ]
            )

    def get_assembly_act(
        self, epoch: Optional[nel.EpochArray] = None
    ) -> nel.AnalogSignalArray:
        """
        Get assembly activity.

        Parameters
        ----------
        epoch : nel.EpochArray, optional
            The epoch to restrict to, by default None.

        Returns
        -------
        nel.AnalogSignalArray
            Assembly activity.
        """
        # check for num of assemblies first
        if self.n_assemblies() == 0:
            return nel.AnalogSignalArray(empty=True)

        if epoch is not None:
            zactmat, ts = self.get_z_mat(self.st[epoch])
        else:
            zactmat, ts = self.get_z_mat(self.st)

        assembly_act = nel.AnalogSignalArray(
            data=assembly.computeAssemblyActivity(self.patterns, zactmat),
            timestamps=ts,
            fs=1 / self.z_mat_dt,
        )
        return assembly_act

    def plot(
        self,
        plot_members: bool = True,
        central_line_color: str = "grey",
        marker_color: str = "k",
        member_color: Union[str, list] = "#6768ab",
        line_width: float = 1.25,
        markersize: float = 4,
        x_padding: float = 0.2,
        figsize: Union[tuple, None] = None,
    ) -> Union[Tuple[plt.Figure, np.ndarray], str, None]:
        """
        Plots basic stem plot to display assembly weights.

        Parameters
        ----------
        plot_members : bool, optional
            Whether to plot assembly members, by default True.
        central_line_color : str, optional
            Color of the central line, by default "grey".
        marker_color : str, optional
            Color of the markers, by default "k".
        member_color : Union[str, List[str]], optional
            Color of the members, by default "#6768ab".
        line_width : float, optional
            Width of the lines, by default 1.25.
        markersize : float, optional
            Size of the markers, by default 4.
        x_padding : float, optional
            Padding on the x-axis, by default 0.2.
        figsize : Optional[Tuple[float, float]], optional
            Size of the figure, by default None.

        Returns
        -------
        Union[Tuple[plt.Figure, np.ndarray], str, None]
            The figure and axes if successful, otherwise a message or None.
        """
        if not hasattr(self, "patterns"):
            return "run get_weights first"
        else:
            if self.patterns is None:
                return None, None
            if plot_members:
                self.find_members()
            if figsize is None:
                figsize = (self.n_assemblies() + 1, np.round(self.n_assemblies() / 2))
            # set up figure with size relative to assembly matrix
            fig, axes = plt.subplots(
                1,
                self.n_assemblies(),
                figsize=figsize,
                sharey=True,
                sharex=True,
            )
            # iter over each assembly and plot the weight per cell
            for i in range(self.n_assemblies()):
                markerline, stemlines, baseline = axes[i].stem(
                    self.patterns[i, :], orientation="horizontal"
                )
                markerline._color = marker_color
                baseline._color = central_line_color
                baseline.zorder = -1000
                plt.setp(stemlines, "color", plt.getp(markerline, "color"))
                plt.setp(stemlines, linewidth=line_width)
                plt.setp(markerline, markersize=markersize)

                if plot_members:
                    current_pattern = self.patterns[i, :].copy()
                    current_pattern[~self.assembly_members[i, :]] = np.nan
                    markerline, stemlines, baseline = axes[i].stem(
                        current_pattern, orientation="horizontal"
                    )
                    if isinstance(
                        member_color, sns.palettes._ColorPalette
                    ) or isinstance(member_color, list):
                        markerline._color = member_color[i]
                    else:
                        markerline._color = member_color
                    baseline._color = "#00000000"
                    baseline.zorder = -1000
                    plt.setp(stemlines, "color", plt.getp(markerline, "color"))
                    plt.setp(stemlines, linewidth=line_width)
                    plt.setp(markerline, markersize=markersize)

                axes[i].spines["top"].set_visible(False)
                axes[i].spines["right"].set_visible(False)

            # give room for marker
            axes[0].set_xlim(
                -self.patterns.max() - x_padding, self.patterns.max() + x_padding
            )

            axes[0].set_ylabel("Neurons #")
            axes[0].set_xlabel("Weights (a.u.)")

            return fig, axes

    def n_assemblies(self) -> int:
        """
        Get the number of detected assemblies.

        Returns
        -------
        int
            Number of detected assemblies.
        """
        if hasattr(self, "patterns"):
            if self.patterns is None:
                return 0
            return self.patterns.shape[0]

    @property
    def isempty(self) -> bool:
        """
        Check if the object is empty.

        Returns
        -------
        bool
            True if empty, False otherwise.
        """
        if hasattr(self, "st"):
            return False
        elif not hasattr(self, "st"):
            return True

    def copy(self) -> "AssemblyReact":
        """
        Returns a copy of the current class.

        Returns
        -------
        AssemblyReact
            A copy of the current class.
        """
        newcopy = copy.deepcopy(self)
        return newcopy

    def __repr__(self) -> str:
        if self.isempty:
            return f"<{self.type_name}: empty>"

        # if st data as been loaded and patterns have been computed
        if hasattr(self, "patterns"):
            n_units = f"{self.st.n_active} units"
            n_patterns = f"{self.n_assemblies()} assemblies"
            dstr = f"of length {self.st.support.length}"
            return "<%s: %s, %s> %s" % (self.type_name, n_units, n_patterns, dstr)

        # if st data as been loaded
        if hasattr(self, "st"):
            n_units = f"{self.st.n_active} units"
            dstr = f"of length {self.st.support.length}"
            return "<%s: %s> %s" % (self.type_name, n_units, dstr)

    def find_members(self) -> np.ndarray:
        """
        Finds significant assembly patterns and significant assembly members.

        Returns
        -------
        np.ndarray
            A ndarray of booleans indicating whether each unit is a significant member of an assembly.

        Notes
        -----
        also, sets self.assembly_members and self.valid_assembly

        self.valid_assembly: a ndarray of booleans indicating an assembly has members with the same sign (Boucly et al. 2022)
        """

        def Otsu(vector: np.ndarray) -> Tuple[np.ndarray, float, float]:
            """
            The Otsu method for splitting data into two groups.

            Parameters
            ----------
            vector : np.ndarray
                Arbitrary vector.

            Returns
            -------
            Tuple[np.ndarray, float, float]
                Group, threshold used for classification, and effectiveness metric.
            """
            sorted = np.sort(vector)
            n = len(vector)
            intraClassVariance = [np.nan] * n
            for i in np.arange(n):
                p = (i + 1) / n
                p0 = 1 - p
                if i + 1 == n:
                    intraClassVariance[i] = np.nan
                else:
                    intraClassVariance[i] = p * np.var(sorted[0 : i + 1]) + p0 * np.var(
                        sorted[i + 1 :]
                    )

            minIntraVariance = np.nanmin(intraClassVariance)
            idx = np.nanargmin(intraClassVariance)
            threshold = sorted[idx]
            group = vector > threshold

            em = 1 - (minIntraVariance / np.var(vector))

            return group, threshold, em

        is_member = []
        keep_assembly = []
        for pat in self.patterns:
            isMember, _, _ = Otsu(np.abs(pat))
            is_member.append(isMember)

            if np.any(pat[isMember] < 0) & np.any(pat[isMember] > 0):
                keep_assembly.append(False)
            elif sum(isMember) == 0:
                keep_assembly.append(False)
            else:
                keep_assembly.append(True)

        self.assembly_members = np.array(is_member)
        self.valid_assembly = np.array(keep_assembly)

        return self.assembly_members
