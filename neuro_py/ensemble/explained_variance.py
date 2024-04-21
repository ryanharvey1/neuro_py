import numpy as np
import pandas as pd
import nelpy as nel
import warnings
from numba import jit

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)


class ExplainedVariance(object):
    """Explained variance measure for assessing reactivation of neuronal activity using pairwise correlations.

    References
    -------
    1) Kudrimoti, H. S., Barnes, C. A., & McNaughton, B. L. (1999). Reactivation of Hippocampal Cell Assemblies: Effects of Behavioral State, Experience, and EEG Dynamics. Journal of Neuroscience, 19(10), 4090–4101. https://doi.org/10/4090
    2) Tatsuno, M., Lipa, P., & McNaughton, B. L. (2006). Methodological Considerations on the Use of Template Matching to Study Long-Lasting Memory Trace Replay. Journal of Neuroscience, 26(42), 10727–10742. https://doi.org/10.1523/JNEUROSCI.3317-06.2006
    """

    def __init__(
        self,
        st: nel.core._eventarray.SpikeTrainArray,
        template: nel.core._intervalarray.EpochArray,
        matching: nel.core._intervalarray.EpochArray,
        control: nel.core._intervalarray.EpochArray,
        bin_size: float = 0.250,
        window: int = 900,
        slideby: int = None,
        ignore_epochs: nel.core._intervalarray.EpochArray = None,
    ):
        """Explained variance measure for assessing reactivation of neuronal activity using pairwise correlations.

        Parameters
        ----------
        st : core.st
            obj that holds spiketrains for multiple st
        template : list/array of length 2
            time in seconds, pairwise correlation calculated from this period will be compared to matching period
        matching : list/array of length 2
            time in seconds, template-correlations will be correlated with pariwise correlations of this period
        control : list/array of length 2
            time in seconds, control for pairwise correlations within this period
        bin_size : float, optional
            in seconds, binning size for spike counts, by default 0.250
        window : int or typle, optional
            window over which pairwise correlations will be calculated in matching and control time periods,
                if window is None entire time period is considered, in seconds, by default 900
        slideby : int, optional
            slide window by this much, in seconds, by default None
        ignore_epochs : core.Epoch, optional
            ignore calculation for these epochs, helps with noisy epochs, by default None
        """
        self.__dict__.update(locals())
        del self.__dict__["self"]

        self.__calculate()

    def __calculate(self):
        self.__truncate_ignore_epochs()
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

    def __truncate_ignore_epochs(self):
        if self.ignore_epochs is not None:
            self.st = self.st[~self.ignore_epochs]

    def __get_window_sizes(self):
        if self.window is None:
            control_window_size = (
                len(np.arange(self.control.start, self.control.stop)) - 1
            )
            matching_window_size = (
                len(np.arange(self.matching.start, self.matching.stop)) - 1
            )
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
        if slideby is not None:
            array = np.arange(epoch_array.start, epoch_array.stop)
            windows = np.lib.stride_tricks.sliding_window_view(array, window_size)
            windows = windows[::slideby, [0, -1]]
        else:
            array = np.arange(epoch_array.start, epoch_array.stop, window_size)
            windows = np.array([array[:-1], array[1:]]).T
        return windows

    def __validate_window_sizes(self, control_window_size, matching_window_size):
        assert (
            control_window_size <= self.control.duration
        ), "window is bigger than matching"
        assert (
            matching_window_size <= self.matching.duration
        ), "window is bigger than matching"

    def __get_template_corr(self):
        self.bst = self.st.bin(ds=self.bin_size)
        return self.__get_pairwise_corr(self.bst[self.template].data)

    def __calculate_pairwise_correlations(self):
        self.matching_paircorr = self.__time_resolved_correlation(self.matching_windows)
        self.control_paircorr = self.__time_resolved_correlation(self.control_windows)

    def __time_resolved_correlation(self, windows):
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
        partial_corr, rev_partial_corr = self.__calculate_partial_correlations_(
            self.matching_paircorr, self.control_paircorr, self.template_corr
        )
        self.__calculate_statistics(partial_corr, rev_partial_corr)

    @staticmethod
    @jit(nopython=True)
    def __calculate_partial_correlations_(
        matching_paircorr, control_paircorr, template_corr
    ):
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
        self.ev = np.nanmean(partial_corr**2, axis=0)
        self.rev = np.nanmean(rev_partial_corr**2, axis=0)
        self.ev_std = np.nanstd(partial_corr**2, axis=0)
        self.rev_std = np.nanstd(rev_partial_corr**2, axis=0)
        self.partial_corr = partial_corr**2
        self.rev_partial_corr = rev_partial_corr**2
        self.n_pairs = len(self.template_corr)
        self.matching_time = np.mean(self.matching_windows, axis=1)
        self.control_time = np.mean(self.control_windows, axis=1)

    @staticmethod
    def __get_pairwise_corr(bst_data):
        """Calculate pairwise correlations."""
        corr = np.corrcoef(bst_data)
        return corr[np.tril_indices(corr.shape[0], k=-1)]

