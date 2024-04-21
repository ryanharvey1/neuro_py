import numpy as np
import pandas as pd
import nelpy as nel
import warnings
from numba import jit

warnings.simplefilter(action="ignore", category=FutureWarning)


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

    # def __calculate_partial_correlations(self):
    #     n_matching = len(self.matching_paircorr)
    #     n_control = len(self.control_paircorr)
    #     partial_corr = np.zeros((n_control, n_matching))
    #     rev_partial_corr = np.zeros((n_control, n_matching))

    #     for m_i, m_pairs in enumerate(self.matching_paircorr):
    #         for c_i, c_pairs in enumerate(self.control_paircorr):
    #             partial_corr[c_i, m_i], rev_partial_corr[c_i, m_i] = (
    #                 self.__explained_variance(self.template_corr, m_pairs, c_pairs)
    #             )

    #     self.__calculate_statistics(partial_corr, rev_partial_corr)

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
            # mat = np.array([covar, x, y])
            mat = np.empty((3, len(x)))
            mat[0] = covar
            mat[1] = x
            mat[2] = y
            # valid = ~np.isnan(mat).any(axis=0)
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

    # @staticmethod
    # def __explained_variance(x, y, covar):
    #     """Calculate explained variance and reverse explained variance."""

    #     corr_df = pd.DataFrame(
    #         {
    #             "r_pre": covar.flatten(),
    #             "r_beh": x.flatten(),
    #             "r_post": y.flatten(),
    #         }
    #     ).corr()
    #     # pull out specific between epoch correlations
    #     beh_pos = corr_df.loc["r_beh", "r_post"]
    #     beh_pre = corr_df.loc["r_beh", "r_pre"]
    #     pre_pos = corr_df.loc["r_pre", "r_post"]

    #     EV = (beh_pos - beh_pre * pre_pos) / (
    #         np.sqrt((1 - beh_pre**2) * (1 - pre_pos**2)) + 1e-10
    #     )
    #     rEV = (beh_pre - beh_pos * pre_pos) / (
    #         np.sqrt((1 - beh_pos**2) * (1 - pre_pos**2)) + 1e-10
    #     )

    #     return EV, rEV

    # valid = ~np.isnan(np.array([covar, x, y])).any(axis=0)
    # numpy_corr = np.corrcoef(np.array([covar, x, y])[:, valid])

    # masked_A = np.ma.MaskedArray(np.array([covar, x, y]).T, np.isnan(np.array([covar, x, y]).T))
    # cov_np = np.ma.cov(masked_A, rowvar=0)

    # rewrite __explained_variance function to use numpy instead of pandas
    # @jit(nopython=True)

    @staticmethod
    def __explained_variance(x, y, covar):
        """Calculate explained variance and reverse explained variance."""

        # Calculate covariance matrix
        mat = np.array([covar, x, y])
        valid = ~np.isnan(mat).any(axis=0)
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


# @jit(nopython=True)
# def calculate_partial_correlations(
#     partial_corr, rev_partial_corr, matching_paircorr, control_paircorr, template_corr
# ):
#     for m_i, m_pairs in enumerate(matching_paircorr):
#         for c_i, c_pairs in enumerate(control_paircorr):
#             partial_corr[c_i, m_i], rev_partial_corr[c_i, m_i] = __explained_variance(
#                 template_corr, m_pairs, c_pairs
#             )
#     return partial_corr, rev_partial_corr


# class ExplainedVariance(object):
#     """Explained variance measure for assessing reactivation of neuronal activity using pairwise correlations.


#     References
#     -------
#     1) Kudrimoti, H. S., Barnes, C. A., & McNaughton, B. L. (1999). Reactivation of Hippocampal Cell Assemblies: Effects of Behavioral State, Experience, and EEG Dynamics. Journal of Neuroscience, 19(10), 4090–4101. https://doi.org/10/4090
#     2) Tatsuno, M., Lipa, P., & McNaughton, B. L. (2006). Methodological Considerations on the Use of Template Matching to Study Long-Lasting Memory Trace Replay. Journal of Neuroscience, 26(42), 10727–10742. https://doi.org/10.1523/JNEUROSCI.3317-06.2006
#     """

#     def __init__(
#         self,
#         st: nel.core._eventarray.SpikeTrainArray,
#         template: nel.core._intervalarray.EpochArray,
#         matching: nel.core._intervalarray.EpochArray,
#         control: nel.core._intervalarray.EpochArray,
#         bin_size: float = 0.250,
#         window: int = 900,
#         slideby: int = None,
#         ignore_epochs: nel.core._intervalarray.EpochArray = None,
#     ):
#         """Explained variance measure for assessing reactivation of neuronal activity using pairwise correlations.

#         Parameters
#         ----------
#         st : core.st
#             obj that holds spiketrains for multiple st
#         template : list/array of length 2
#             time in seconds, pairwise correlation calculated from this period will be compared to matching period
#         matching : list/array of length 2
#             time in seconds, template-correlations will be correlated with pariwise correlations of this period
#         control : list/array of length 2
#             time in seconds, control for pairwise correlations within this period
#         bin_size : float, optional
#             in seconds, binning size for spike counts, by default 0.250
#         window : int or typle, optional
#             window over which pairwise correlations will be calculated in matching and control time periods,
#                 if window is None entire time period is considered, in seconds, by default 900
#         slideby : int, optional
#             slide window by this much, in seconds, by default None
#         ignore_epochs : core.Epoch, optional
#             ignore calculation for these epochs, helps with noisy epochs, by default None
#         """
#         self.__dict__.update(locals())
#         del self.__dict__["self"]

#         self.__calculate()

#     def __time_resolved_correlation(self, windows):
#         paircorr = []
#         for w in windows:
#             idx = (self.bst.bin_centers > w[0]) & (self.bst.bin_centers < w[1])
#             paircorr.append(self.__get_pairwise_corr(self.bst.data[:, idx]))
#         return paircorr

#     def __calculate(self):
#         # truncate/delete windows if they fall within ignore_epochs
#         if self.ignore_epochs is not None:
#             st = st[~self.ignore_epochs]

#         if self.window is None:
#             control_window_size = self.control.duration
#             matching_window_size = self.matching.duration
#             slideby = None
#         elif self.window is not None and self.slideby is None:
#             control_window_size = self.window
#             matching_window_size = self.window
#             slideby = None
#         else:
#             control_window_size = self.window
#             matching_window_size = self.window
#             slideby = self.slideby

#         matching_array = np.arange(self.matching.start, self.matching.stop)
#         matching_windows = np.lib.stride_tricks.sliding_window_view(
#             matching_array, matching_window_size
#         )[::slideby, [0, -1]]

#         control_array = np.arange(self.control.start, self.control.stop)
#         control_windows = np.lib.stride_tricks.sliding_window_view(
#             control_array, control_window_size
#         )[::slideby, [0, -1]]

#         assert (
#             control_window_size <= self.control.duration
#         ), "window is bigger than matching"
#         assert (
#             matching_window_size <= self.matching.duration
#         ), "window is bigger than matching"

#         # bin spike train
#         self.bst = self.st.bin(ds=self.bin_size)

#         template_corr = self.__get_pairwise_corr(self.bst[self.template].data)

#         matching_paircorr = self.__time_resolved_correlation(matching_windows)
#         control_paircorr = self.__time_resolved_correlation(control_windows)

#         partial_corr = np.zeros((control_windows.shape[0], matching_windows.shape[0]))
#         rev_partial_corr = np.zeros(
#             (control_windows.shape[0], matching_windows.shape[0])
#         )
#         for m_i, m_pairs in enumerate(matching_paircorr):
#             for c_i, c_pairs in enumerate(control_paircorr):
#                 df = pd.DataFrame({"t": template_corr, "m": m_pairs, "c": c_pairs})

#                 partial_corr[c_i, m_i] = pg.partial_corr(df, x="t", y="m", covar="c").r

#                 rev_partial_corr[c_i, m_i] = pg.partial_corr(
#                     df, x="t", y="c", covar="m"
#                 ).r

#         self.ev = np.nanmean(partial_corr**2, axis=0)
#         self.rev = np.nanmean(rev_partial_corr**2, axis=0)
#         self.ev_std = np.nanstd(partial_corr**2, axis=0)
#         self.rev_std = np.nanstd(rev_partial_corr**2, axis=0)
#         self.partial_corr = partial_corr
#         self.rev_partial_corr = rev_partial_corr
#         self.n_pairs = len(template_corr)
#         self.matching_time = np.mean(matching_windows, axis=1)
#         self.control_time = np.mean(control_windows, axis=1)

#     @staticmethod
#     def __get_pairwise_corr(bst_data):
#         corr = np.corrcoef(bst_data)
#         return corr[np.tril(np.ones(corr.shape).astype("bool"), k=-1)]
