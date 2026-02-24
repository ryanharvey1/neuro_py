import numpy as np
from nelpy.core._analogsignalarray import AnalogSignalArray
from nelpy.core._eventarray import SpikeTrainArray
from nelpy.core._intervalarray import EpochArray
from numba import jit


class ExplainedVariance(object):
    """Explained variance measure for assessing reactivation of neuronal activity using pairwise correlations.

    References
    -------
    1) Kudrimoti, H. S., Barnes, C. A., & McNaughton, B. L. (1999).
        Reactivation of Hippocampal Cell Assemblies: Effects of Behavioral State, Experience, and EEG Dynamics.
        Journal of Neuroscience, 19(10), 4090-4101. https://doi.org/10/4090
    2) Tatsuno, M., Lipa, P., & McNaughton, B. L. (2006).
        Methodological Considerations on the Use of Template Matching to Study Long-Lasting Memory Trace Replay.
        Journal of Neuroscience, 26(42), 10727-10742. https://doi.org/10.1523/JNEUROSCI.3317-06.2006

    Adapted from https://github.com/diba-lab/NeuroPy/blob/main/neuropy/analyses/reactivation.py

    Attributes
    ----------
    st : SpikeTrainArray
        obj that holds spiketrains
    template : EpochArray
        time in seconds, pairwise correlation calculated from this period will be compared to matching period (task-period)
    matching : EpochArray
        time in seconds, template-correlations will be correlated with pariwise correlations of this period (post-task period)
    control : EpochArray
        time in seconds, control for pairwise correlations within this period (pre-task period)
    bin_size : float
        in seconds, binning size for spike counts
    window : int
        window over which pairwise correlations will be calculated in matching and control time periods,
            if window is None entire time period is considered, in seconds
    slideby : int
        slide window by this much, in seconds
    matching_windows : array
        windows for matching period
    control_windows : array
        windows for control period
    template_corr : array
        pairwise correlations for template period
    matching_paircorr : array
        pairwise correlations for matching period
    control_paircorr : array
        pairwise correlations for control period
    ev : array
        explained variance for each time point
    rev : array
        reverse explained variance for each time point
    ev_std : array
        explained variance standard deviation for each time point
    rev_std : array
        reverse explained variance standard deviation for each time point
    partial_corr : array
        partial correlations for each time point
    rev_partial_corr : array
        reverse partial correlations for each time point
    n_pairs : int
        number of pairs
    matching_time : array
        time points for matching period
    control_time : array
        time points for control period
    ev_signal : AnalogSignalArray
        explained variance signal
    rev_signal : AnalogSignalArray
        reverse explained variance signal
    plot : function
        plot explained variance
    pvalue : function
        calculate p-value for explained variance by shuffling the template correlations

    Examples
    --------
    # Load data
    >>> basepath = "U:/data/HMC/HMC1/day8"
    >>> st,cm = loading.load_spikes(basepath,brainRegion="CA1",putativeCellType="Pyr")

    >>> epoch_df = loading.load_epoch(basepath)
    >>> beh_epochs = nel.EpochArray(epoch_df[["startTime", "stopTime"]].values)


    # Most simple case, returns single explained variance value
    >>> expvar = explained_variance.ExplainedVariance(
    >>>        st=st,
    >>>        template=beh_epochs[1],
    >>>        matching=beh_epochs[2],
    >>>        control=beh_epochs[0],
    >>>        window=None,
    >>>    )

    # Get time resolved explained variance across entire session in 200sec bins
    >>> expvar = explained_variance.ExplainedVariance(
    >>>        st=st,
    >>>        template=beh_epochs[1],
    >>>        matching=nel.EpochArray([beh_epochs.start, beh_epochs.stop]),
    >>>        control=beh_epochs[0],
    >>>        window=200
    >>>    )

    # Get time resolved explained variance across entire session in 200sec bins sliding by 100sec
    >>> expvar = explained_variance.ExplainedVariance(
    >>>        st=st,
    >>>        template=beh_epochs[1],
    >>>        matching=nel.EpochArray([beh_epochs.start, beh_epochs.stop]),
    >>>        control=beh_epochs[0],
    >>>        window=200,
    >>>        slideby=100
    >>>    )
    """

    def __init__(
        self,
        st: SpikeTrainArray,
        template: EpochArray,
        matching: EpochArray,
        control: EpochArray,
        bin_size: float = 0.2,
        window: int = 900,
        slideby: int = None,
    ):
        """Explained variance measure for assessing reactivation of neuronal activity using pairwise correlations.

        Parameters
        ----------
        st : SpikeTrainArray
            obj that holds spiketrains
        template : EpochArray
            time in seconds, pairwise correlation calculated from this period will be compared to matching period (task-period)
        matching : EpochArray
            time in seconds, template-correlations will be correlated with pariwise correlations of this period (post-task period)
        control : EpochArray
            time in seconds, control for pairwise correlations within this period (pre-task period)
        bin_size : float, optional
            in seconds, binning size for spike counts, by default 0.2
        window : int, optional
            window over which pairwise correlations will be calculated in matching and control time periods,
                if window is None entire time period is considered, in seconds, by default 900
        slideby : int, optional
            slide window by this much, in seconds, by default None
        """
        self.__dict__.update(locals())
        del self.__dict__["self"]

        self.__validate_input()
        self.__calculate()

    def __validate_input(self):
        """Validate input parameters."""
        assert isinstance(self.st, SpikeTrainArray)
        assert isinstance(self.template, EpochArray)
        assert isinstance(self.matching, EpochArray)
        assert isinstance(self.control, EpochArray)
        assert isinstance(self.bin_size, (float, int))
        assert isinstance(self.window, (int, type(None)))
        assert isinstance(self.slideby, (int, type(None)))

    def __calculate(self):
        """processing steps for explained variance calculation."""
        control_window_size, matching_window_size, slideby = self.__get_window_sizes()

        self.matching_windows = self.__get_windows_array(
            self.matching, matching_window_size, slideby
        )
        self.control_windows = self.__get_windows_array(
            self.control, control_window_size, slideby
        )
        self.__validate_window_sizes(control_window_size, matching_window_size)
        self.template_corr = self.__get_template_corr()
        self.__calculate_pairwise_correlations()
        self.__calculate_partial_correlations()

    def __get_window_sizes(self):
        """Get window sizes for control and matching periods."""
        if self.window is None:
            control_window_size = np.array(self.control.duration).astype(int)
            matching_window_size = np.array(self.matching.duration).astype(int)
            slideby = None
        elif self.slideby is None:
            control_window_size = self.window
            matching_window_size = self.window
            slideby = None
        else:
            control_window_size = self.window
            matching_window_size = self.window
            slideby = self.slideby
        return control_window_size, matching_window_size, slideby

    def __get_windows_array(self, epoch_array, window_size, slideby):
        """Get windows array for control and matching periods."""
        if slideby is not None:
            array = np.arange(epoch_array.start, epoch_array.stop)
            windows = np.lib.stride_tricks.sliding_window_view(array, window_size)
            windows = windows[::slideby, [0, -1]]
        elif np.array(epoch_array.duration) == window_size:
            windows = np.array([[epoch_array.start, epoch_array.stop]])
        else:
            array = np.arange(epoch_array.start, epoch_array.stop, window_size)
            windows = np.array([array[:-1], array[1:]]).T
        return windows

    def __validate_window_sizes(self, control_window_size, matching_window_size):
        """Validate window sizes."""
        assert (
            control_window_size <= self.control.duration
        ), "window is bigger than control"
        assert (
            matching_window_size <= self.matching.duration
        ), "window is bigger than matching"

    def __get_template_corr(self):
        """Get pairwise correlations for template period."""
        self.bst = self.st.bin(ds=self.bin_size)
        return self.__get_pairwise_corr(self.bst[self.template].data)

    def __calculate_pairwise_correlations(self):
        """Calculate pairwise correlations for matching and control periods."""
        self.matching_paircorr = self.__time_resolved_correlation(self.matching_windows)
        self.control_paircorr = self.__time_resolved_correlation(self.control_windows)

    @staticmethod
    def __get_pairwise_corr(bst_data):
        """Calculate pairwise correlations."""
        corr = np.corrcoef(bst_data)
        return corr[np.tril_indices(corr.shape[0], k=-1)]

    def __time_resolved_correlation(self, windows):
        """Calculate pairwise correlations for given windows."""
        paircorr = []
        bst_data = self.bst.data
        bin_centers = self.bst.bin_centers

        for w in windows:
            start, stop = w
            idx = (bin_centers > start) & (bin_centers < stop)
            corr = np.corrcoef(bst_data[:, idx])
            paircorr.append(corr[np.tril_indices(corr.shape[0], k=-1)])

        return np.array(paircorr)

    def __calculate_partial_correlations(self):
        """Calculate partial correlations."""
        partial_corr, rev_partial_corr = self.__calculate_partial_correlations_(
            self.matching_paircorr, self.control_paircorr, self.template_corr
        )
        self.__calculate_statistics(partial_corr, rev_partial_corr)

    @staticmethod
    @jit(nopython=True)
    def __calculate_partial_correlations_(
        matching_paircorr, control_paircorr, template_corr
    ):
        """Calculate partial correlations."""

        def __explained_variance(x, y, covar):
            """Calculate explained variance and reverse explained variance."""

            # Calculate covariance matrix
            n = len(covar)
            valid = np.zeros(n, dtype=np.bool_)
            for i in range(n):
                valid[i] = not (np.isnan(covar[i]) or np.isnan(x[i]) or np.isnan(y[i]))
            mat = np.empty((3, len(x)))
            mat[0] = covar
            mat[1] = x
            mat[2] = y
            cov = np.corrcoef(mat[:, valid])

            # Calculate explained variance
            EV = (cov[1, 2] - cov[0, 1] * cov[0, 2]) / (
                np.sqrt((1 - cov[0, 1] ** 2) * (1 - cov[0, 2] ** 2)) + 1e-10
            )

            # Calculate reverse explained variance
            rEV = (cov[0, 1] - cov[1, 2] * cov[0, 2]) / (
                np.sqrt((1 - cov[1, 2] ** 2) * (1 - cov[0, 2] ** 2)) + 1e-10
            )

            return EV, rEV

        n_matching = len(matching_paircorr)
        n_control = len(control_paircorr)
        partial_corr = np.zeros((n_control, n_matching))
        rev_partial_corr = np.zeros((n_control, n_matching))

        for m_i, m_pairs in enumerate(matching_paircorr):
            for c_i, c_pairs in enumerate(control_paircorr):
                partial_corr[c_i, m_i], rev_partial_corr[c_i, m_i] = (
                    __explained_variance(template_corr, m_pairs, c_pairs)
                )
        return partial_corr, rev_partial_corr

    def __calculate_statistics(self, partial_corr, rev_partial_corr):
        """Calculate explained variance statistics."""
        self.ev = np.nanmean(partial_corr**2, axis=0)
        self.rev = np.nanmean(rev_partial_corr**2, axis=0)
        self.ev_std = np.nanstd(partial_corr**2, axis=0)
        self.rev_std = np.nanstd(rev_partial_corr**2, axis=0)
        self.partial_corr = partial_corr**2
        self.rev_partial_corr = rev_partial_corr**2
        self.n_pairs = len(self.template_corr)
        self.matching_time = np.mean(self.matching_windows, axis=1)
        self.control_time = np.mean(self.control_windows, axis=1)

    @property
    def ev_signal(self):
        """Return explained variance signal."""
        return AnalogSignalArray(
            data=self.ev,
            timestamps=self.matching_time,
            fs=1 / np.diff(self.matching_time)[0],
            support=EpochArray(data=[self.matching.start, self.matching.stop]),
        )

    @property
    def rev_signal(self):
        """Return reverse explained variance signal."""
        return AnalogSignalArray(
            data=self.rev,
            timestamps=self.matching_time,
            fs=1 / np.diff(self.matching_time)[0],
            support=EpochArray(data=[self.matching.start, self.matching.stop]),
        )

    def pvalue(self, n_shuffles=1000):
        """
        Calculate p-value for explained variance by shuffling the template correlations.
        """
        from copy import deepcopy

        def shuffle_template(self):
            template_corr = deepcopy(self.template_corr)
            np.random.shuffle(template_corr)

            partial_corr, _ = self.__calculate_partial_correlations_(
                self.matching_paircorr, self.control_paircorr, template_corr
            )
            ev = np.nanmean(partial_corr**2, axis=0)
            return ev.flatten()

        if len(self.ev) > 1:
            print("Multiple time points, p-values are not supported")
            return

        ev_shuffle = [shuffle_template(self) for _ in range(n_shuffles)]

        ev_shuffle = np.array(ev_shuffle)

        n = len(ev_shuffle)
        r = np.sum(ev_shuffle > self.ev)
        pvalues = (r + 1) / (n + 1)
        return pvalues

    def plot(self):
        """Plot explained variance."""
        if self.matching_time.size == 1:
            print("Only single time point, cannot plot")
            return
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(8, 3))
        ax.plot(self.matching_time, self.ev, label="EV")
        ax.fill_between(
            self.matching_time,
            self.ev - self.ev_std,
            self.ev + self.ev_std,
            alpha=0.5,
        )
        ax.plot(self.matching_time, self.rev, label="rEV", color="grey")
        ax.fill_between(
            self.matching_time,
            self.rev - self.rev_std,
            self.rev + self.rev_std,
            alpha=0.5,
            color="grey",
        )
        # check if matching time overlaps with control time and plot control time
        if np.any(
            (self.control_time >= self.matching_time[0])
            & (self.control_time <= self.matching_time[-1])
        ):
            ax.axvspan(
                self.control.start,
                self.control.stop,
                color="green",
                alpha=0.3,
                label="Control",
                zorder=-10,
            )
        # check if matching time overlaps with template time and plot template time
        if np.any(
            (self.template.start >= self.matching_time[0])
            & (self.template.stop <= self.matching_time[-1])
        ):
            ax.axvspan(
                self.template.start,
                self.template.stop,
                color="purple",
                alpha=0.4,
                label="Template",
                zorder=-10,
            )
        # remove axis spines
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        ax.legend(frameon=False)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Explained Variance")
        ax.set_title("Explained Variance")
        plt.show()


def explained_variance(
    task: np.ndarray, post_task: np.ndarray, pre_task: np.ndarray
) -> tuple:
    """
    Simplified version of explained variance and reverse explained variance

    Parameters
    ----------
    task : np.ndarray
        2D array, spike counts matrix with shape(n_features, n_timepoints)
    post_task : np.ndarray
        2D array, spike counts matrix with shape(n_features, n_timepoints)
    pre_task : np.ndarray
        2D array, spike counts matrix with shape(n_features, n_timepoints)

    Returns
    -------
    EV : float
        explained variance
    rEV : float
        reverse explained variance

    Examples
    --------
    >>> import numpy as np
    >>> from neuro_py.ensemble import explained_variance
    >>> # build correlated task/post epochs and a weaker pre epoch
    >>> rng = np.random.default_rng(0)
    >>> n_features, n_time = 10, 300
    >>> rho_task, rho_pre = 0.5, 0.1
    >>> cov_task = np.full((n_features, n_features), rho_task); np.fill_diagonal(cov_task, 1.0)
    >>> cov_pre = np.full((n_features, n_features), rho_pre); np.fill_diagonal(cov_pre, 1.0)
    >>> task = rng.multivariate_normal(np.zeros(n_features), cov_task, size=n_time).T
    >>> post = rng.multivariate_normal(np.zeros(n_features), cov_task, size=n_time).T
    >>> pre = rng.multivariate_normal(np.zeros(n_features), cov_pre, size=n_time).T
    >>> EV, rEV = explained_variance.explained_variance(task, post, pre)
    >>> EV > rEV
    True


    >>> import neuro_py as npy
    >>> import nelpy as nel
    >>> from neuro_py.ensemble import explained_variance
    >>> basepath = "S:/data/HMC/HMC1/day8"
    >>> st, cm = npy.io.load_spikes(basepath, brainRegion="CA1")
    >>> epoch_df = npy.io.load_epoch(basepath)
    >>> beh_epochs = nel.EpochArray(epoch_df[["startTime", "stopTime"]].values)
    >>> state_dict = npy.io.load_SleepState_states(basepath)
    >>> nrem_epochs = nel.EpochArray(
    ...    state_dict["NREMstate"],
    ... )
    >>> theta_cycles = npy.io.load_theta_cycles(basepath, return_epoch_array=True)
    >>> theta_cycles = theta_cycles[beh_epochs[1]]  # only during behavior
    >>> # bin spike trains into each theta cycle
    >>> bst_task = npy.process.count_in_interval(
    ...     st.data, theta_cycles.starts, theta_cycles.stops
    ... )
    >>> # bin spike trains into 50ms bins during pre sleep
    >>> bst_pre = st[beh_epochs[0] & nrem_epochs].bin(ds=0.05).data
    >>> # bin spike trains into 50ms bins during post sleep
    >>> bst_post = st[beh_epochs[2] & nrem_epochs].bin(ds=0.05).data

    >>> ev, rev = explained_variance.explained_variance(bst_task, bst_post, bst_pre)
    >>> print(f"Explained Variance: {ev}, Reverse Explained Variance: {rev}")
    Explained Variance: 0.21654828336188703, Reverse Explained Variance: 0.00413191971965775

    Notes
    -----
    n_timepoints can differ between task, post_task, pre_task
    """

    # Coerce inputs to NumPy arrays and validate dimensionality
    task = np.asarray(task)
    pre_task = np.asarray(pre_task)
    post_task = np.asarray(post_task)

    for name, arr in (("task", task), ("post_task", post_task), ("pre_task", pre_task)):
        if arr.ndim != 2:
            raise ValueError(
                f"{name} must be a 2D array of shape (n_units, n_bins); "
                f"got array with shape {arr.shape} and ndim={arr.ndim}"
            )

    # Validate feature dimensions match
    if task.shape[0] != post_task.shape[0] or task.shape[0] != pre_task.shape[0]:
        raise ValueError("All inputs must have the same number of features (rows)")

    # Pairwise correlation matrices for each epoch
    corr_beh = np.corrcoef(task)
    corr_pre = np.corrcoef(pre_task)
    corr_post = np.corrcoef(post_task)

    # Use strictly lower triangle (no diagonal) to form pair vectors
    n = corr_beh.shape[0]
    li = np.tril_indices(n, k=-1)
    r_beh = corr_beh[li]
    r_pre = corr_pre[li]
    r_post = corr_post[li]

    # Helper: correlation between 1D vectors (guard against degenerate variance and NaNs)
    def _corr(a, b):
        # Remove entries where either vector has NaN
        mask = ~np.isnan(a) & ~np.isnan(b)
        a_clean = a[mask]
        b_clean = b[mask]
        if a_clean.size == 0 or b_clean.size == 0:
            return np.nan
        if np.nanstd(a_clean) == 0 or np.nanstd(b_clean) == 0:
            return 0.0
        return float(np.corrcoef(a_clean, b_clean)[0, 1])

    # Between-epoch correlations of pairwise templates
    beh_pos = _corr(r_beh, r_post)
    beh_pre = _corr(r_beh, r_pre)
    pre_pos = _corr(r_pre, r_post)

    # Explained variance and reverse explained variance (squared partial correlations)
    eps = 1e-10
    denom_ev = np.sqrt((1 - beh_pre**2) * (1 - pre_pos**2)) + eps
    denom_rev = np.sqrt((1 - beh_pos**2) * (1 - pre_pos**2)) + eps
    EV = ((beh_pos - beh_pre * pre_pos) / denom_ev) ** 2
    rEV = ((beh_pre - beh_pos * pre_pos) / denom_rev) ** 2

    return EV, rEV
