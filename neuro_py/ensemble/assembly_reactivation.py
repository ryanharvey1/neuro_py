from typing import Tuple, Union
import numpy as np
import nelpy as nel
from scipy import stats
from neuro_py.io import loading
from neuro_py.session.locate_epochs import compress_repeated_epochs
from neuro_py.ensemble import assembly
from neuro_py.session.locate_epochs import find_pre_task_post
import logging
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union

logging.getLogger().setLevel(logging.ERROR)


class AssemblyReact(object):
    """
    Class for running assembly reactivation analysis

    Core assembly methods come from assembly.py by Vítor Lopes dos Santos
        https://doi.org/10.1016/j.jneumeth.2013.04.010

    Parameters:
    -----------
    basepath: str
        Path to the session folder
    brainRegion: str
        Brain region to restrict to. Can be multi ex. "CA1|CA2"
    putativeCellType: str
        Cell type to restrict to
    weight_dt: float
        Time resolution of the weight matrix
    z_mat_dt: float
        Time resolution of the z matrix
    method: str
        Defines how to extract assembly patterns (ica,pca).
    nullhyp: str
        Defines how to generate statistical threshold for assembly detection (bin,circ,mp).
    nshu: int
        Number of shuffles for bin and circ null hypothesis.
    percentile: int
        Percentile for mp null hypothesis.
    tracywidom: bool
        If true, uses Tracy-Widom distribution for mp null hypothesis.

    attributes:
    -----------
    st: spike train (nelpy:SpikeTrainArray)
    cell_metrics: cell metrics (pandas:DataFrame)
    ripples: ripples (nelpy:EpochArray)
    patterns: assembly patterns (numpy:array)
    assembly_act: assembly activity (nelpy:AnalogSignalArray)

    methods:
    --------
    load_data: load data (st, ripples, epochs)
    restrict_to_epoch: restrict to a epoch
    get_z_mat: get z matrix
    get_weights: get assembly weights
    get_assembly_act: get assembly activity
    n_assemblies: number of detected assemblies
    isempty: isempty (bool)
    copy: returns copy of class
    plot: stem plot of assembly weights
    find_members: find members of an assembly

    *Usage*::

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

    def add_st(self, st):
        self.st = st

    def add_ripples(self, ripples):
        self.ripples = ripples

    def add_epoch_df(self, epoch_df):
        self.epoch_df = epoch_df

    def load_spikes(self):
        """
        loads spikes from the session folder
        """
        self.st, self.cell_metrics = loading.load_spikes(
            self.basepath,
            brainRegion=self.brainRegion,
            putativeCellType=self.putativeCellType,
            support=self.time_support,
        )

    def load_ripples(self):
        """
        loads ripples from the session folder
        """
        ripples = loading.load_ripples_events(self.basepath)
        self.ripples = nel.EpochArray(
            [np.array([ripples.start, ripples.stop]).T], domain=self.time_support
        )

    def load_epoch(self):
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

    def load_data(self):
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

    def restrict_to_epoch(self, epoch):
        """
        Restricts the spike data to a specific epoch
        """
        self.st_resticted = self.st[epoch]

    def get_z_mat(self, st):
        """
        To increase the temporal resolution beyond the
        bin-size used to identify the assembly patterns,
        z(t) was obtained by convolving the spike-train
        of each neuron with a kernel-function
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

    def get_weights(self, epoch=None):
        """
        Gets the assembly weights
        """

        # check if st has any neurons
        if self.st.isempty:
            self.patterns = []
            return

        if epoch is not None:
            bst = self.st[epoch].bin(ds=self.weight_dt).data
        else:
            bst = self.st.bin(ds=self.weight_dt).data

        if (bst == 0).all():
            self.patterns = []
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

    def get_assembly_act(self, epoch=None):
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
    ):
        """
        plots basic stem plot to display assembly weights
        """

        if not hasattr(self, "patterns"):
            return f"run get_weights first"
        else:
            if self.patterns == []:
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

    def n_assemblies(self):
        if hasattr(self, "patterns"):
            if self.patterns == []:
                return 0
            elif self.patterns is None:
                return 0
            return self.patterns.shape[0]

    @property
    def isempty(self):
        if hasattr(self, "st"):
            return False
        elif not hasattr(self, "st"):
            return True

    def copy(self):
        """Returns a copy of the current class."""
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
        Finds significant assembly patterns and signficant assembly members

        Output:
            assembly_members: a ndarray of booleans indicating whether each unit is a significant member of an assembly

        also, sets self.assembly_members and self.valid_assembly

        self.valid_assembly: a ndarray of booleans indicating an assembly has members with the same sign (Boucly et al. 2022)

        """

        def Otsu(vector: np.ndarray) -> Tuple[np.ndarray, float, float]:
            """
            The Otsu method for splitting data into two groups.
            This is somewhat equivalent to kmeans(vector,2), but while the kmeans implementation
            finds a local minimum and may therefore produce different results each time,
            the Otsu implementation is guaranteed to find the best division every time.

            input:
                vector: arbitrary vector
            output:
                group: binary class
                threshold: threshold used for classification
                em: effectiveness metric

            From Raly
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


# def get_peak_activity(assembly_act, epochs):
#     """
#     Gets the peak activity of the assembly activity
#     """
#     strengths = []
#     assembly_id = []
#     centers = []
#     for assembly_act, ep in zip(assembly_act[epochs], epochs):
#         strengths.append(assembly_act.max())
#         assembly_id.append(np.arange(assembly_act.n_signals))
#         centers.append(np.tile(ep.centers, assembly_act.n_signals))

#     return np.hstack(assembly_id), np.hstack(strengths), np.hstack(centers)


# def get_pre_post_assembly_strengths(basepath):
#     """
#     Gets the pre and post assembly strengths
#     """
#     # initialize session
#     m1 = AssemblyReact(basepath, weight_dt=0.025)
#     # load data
#     m1.load_data()
#     # check if no cells were found
#     if m1.cell_metrics.shape[0] == 0:
#         return None
#     # restrict to pre/task/post epochs
#     m1.restrict_epochs_to_pre_task_post()
#     # get weights for task outside ripples
#     # % (TODO: use more robust method to locate epochs than index)
#     m1.get_weights(m1.epochs[1][~m1.ripples])

#     # get assembly activity
#     assembly_act_pre = m1.get_assembly_act(epoch=m1.ripples[m1.epochs[0]])
#     assembly_act_task = m1.get_assembly_act(epoch=m1.ripples[m1.epochs[1]])
#     assembly_act_post = m1.get_assembly_act(epoch=m1.ripples[m1.epochs[2]])
#     results = {
#         "assembly_act_pre": assembly_act_pre,
#         "assembly_act_task": assembly_act_task,
#         "assembly_act_post": assembly_act_post,
#         "react": m1,
#     }

#     return results


# def session_loop(basepath, save_path):
#     save_file = os.path.join(
#         save_path, basepath.replace(os.sep, "_").replace(":", "_") + ".pkl"
#     )
#     if os.path.exists(save_file):
#         return
#     results = get_pre_post_assembly_strengths(basepath)
#     # save file
#     with open(save_file, "wb") as f:
#         pickle.dump(results, f)


# def run(df, save_path, parallel=True):
#     # find sessions to run
#     basepaths = pd.unique(df.basepath)

#     if not os.path.exists(save_path):
#         os.mkdir(save_path)

#     if parallel:
#         num_cores = multiprocessing.cpu_count()
#         processed_list = Parallel(n_jobs=num_cores)(
#             delayed(session_loop)(basepath, save_path) for basepath in basepaths
#         )
#     else:
#         for basepath in basepaths:
#             print(basepath)
#             session_loop(basepath, save_path)


# def load_results(save_path):
#     sessions = glob.glob(save_path + os.sep + "*.pkl")
#     all_results = {}
#     for session in sessions:
#         with open(session, "rb") as f:
#             results = pickle.load(f)
#             if results is None:
#                 continue
#         all_results[results["react"].basepath] = results
#     return all_results
