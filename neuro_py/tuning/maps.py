import numpy as np
import nelpy as nel
import copy
import scipy
from neuro_py.tuning import fields
from neuro_py.stats.stats import get_significant_events
from scipy.spatial.distance import pdist
import logging
import multiprocessing
from joblib import Parallel, delayed
from scipy.io import savemat
import os
from typing import Union, List

# logging.getLogger().setLevel(logging.ERROR)
np.seterr(divide="ignore", invalid="ignore")


class SpatialMap(object):
    """
    SpatialMap: make a spatial map tuning curve
        maps timestamps or continuous signals onto positions

    args:
        pos: position data (nelpy.AnalogSignal or nel.PositionArray)
        st: spike train data (nelpy.SpikeTrain or nelpy.AnalogSignal)
        
        optional:
            speed: speed data (nelpy.AnalogSignal), recommended input: from non-epoched data
            dim: dimension of the map (1 or 2) *deprecated*
            dir_epoch: epochs of the running direction, for linear data (nelpy.Epoch) *deprecated*
            speed_thres: speed threshold for running (float)
            s_binsize: bin size for the spatial map (float)
            x_minmax: min and max x values for the spatial map (list)
            y_minmax: min and max y values for the spatial map (list)
            tuning_curve_sigma: sigma for the tuning curve (float)
            smooth_mode: mode for smoothing curve (str) reflect,constant,nearest,mirror,wrap
            min_duration: minimum duration for a tuning curve (float)
            minbgrate: min firing rate for tuning curve, will set to this if lower (float)
            n_shuff: number of positon shuffles for spatial information (int)
            parallel_shuff: parallelize shuffling (bool)

            Place field detector parameters:
                place_field_thres: percent of continuous region of peak firing rate (float)
                place_field_min_size: min size of place field (cm) (float)
                place_field_min_peak: min peak rate of place field (float)
                place_field_sigma: extra smoothing sigma to apply before field detection (float)

    attributes:
        tc: tuning curves (nelpy.TuningCurve)
        st_run: spike train restricted to running epochs (nelpy.SpikeTrain)
        bst_run: binned spike train restricted to running epochs (nelpy.binnedSpikeTrain)
        speed: speed data (nelpy.AnalogSignal)
        run_epochs: running epochs (nelpy.EpochArray)
    Note:
        Place field detector (.find_fields()) is sensitive to many parameters.
        For 2D, it is highly recommended to have good environmental sampling.
        In brief testing with 300cm linear track, optimal 1D parameters were:
            place_field_min_size=15
            place_field_max_size=None
            place_field_min_peak=3
            place_field_sigma=None
            place_field_thres=.33

    TODO: place field detector currently collects field width and peak rate for peak place field
            In the future, these should be stored for all sub fields
    """

    def __init__(
        self,
        pos: object,
        st: object,
        speed: object = None,
        dim: int = None,  # deprecated
        dir_epoch: object = None,  # deprecated
        speed_thres: Union[int, float] = 4,
        s_binsize: Union[int, float] = 3,
        tuning_curve_sigma: Union[int, float] = 3,
        x_minmax: List[Union[int, float]] = None,
        y_minmax: List[Union[int, float]] = None,
        smooth_mode: str = "reflect",
        min_duration: float = 0.1,
        minbgrate: Union[int, float] = 0,
        n_shuff: int = 500,
        parallel_shuff: bool = True,
        place_field_thres: Union[int, float] = 0.2,
        place_field_min_size: Union[int, float] = None,
        place_field_max_size: Union[int, float] = None,
        place_field_min_peak: Union[int, float] = 3,
        place_field_sigma: Union[int, float] = 2,
    ) -> None:
        # add all the inputs to self
        self.__dict__.update(locals())
        del self.__dict__["self"]

        # Verify inputs: make sure pos and st are nelpy objects
        if not isinstance(
            pos, (nel.core._analogsignalarray.AnalogSignalArray, nel.core.PositionArray)
        ):
            raise TypeError("pos must be nelpy.AnalogSignal or nelpy.PositionArray")
        if not isinstance(
            st,
            (
                nel.core._eventarray.SpikeTrainArray,
                nel.core._analogsignalarray.AnalogSignalArray,
            ),
        ):
            raise TypeError(
                "st must be nelpy.SpikeTrain or nelpy.BinnedSpikeTrainArray"
            )

        # get speed and running epochs (highly recommended you calculate 
        #   speed before hand on non epoched data)
        if self.speed is None:
            self.speed = nel.utils.ddt_asa(self.pos, smooth=True, sigma=0.1, norm=True)

        self.run_epochs = nel.utils.get_run_epochs(
            self.speed, v1=self.speed_thres, v2=self.speed_thres
        ).merge()

        # calculate maps, 1d or 2d
        self.dim = pos.n_signals
        if pos.n_signals == 2:
            self.tc, self.st_run = self.map_2d()
        elif pos.n_signals == 1:
            self.tc, self.st_run = self.map_1d()
        else:
            raise ValueError("pos dims must be 1 or 2")
        

        # find place fields. Currently only collects metrics from peak field
        # self.find_fields()

    def map_1d(self, pos: object = None):
        # dir_epoch is deprecated input
        if self.dir_epoch is not None:
            # warn user
            logging.warning(
                "dir_epoch is deprecated and will be removed. Epoch data by direction prior to calling SpatialMap"
            )
            self.st = self.st[self.dir_epoch]
            self.pos = self.pos[self.dir_epoch]

        # restrict spike trains to those epochs during which the animal was running
        st_run = self.st[self.run_epochs]

        # take pos as input for case of shuffling
        if pos is not None:
            pos_run = pos[self.run_epochs]
        else:
            pos_run = self.pos[self.run_epochs]

        if self.x_minmax is None:
            x_max = np.ceil(np.nanmax(self.pos.data))
            x_min = np.floor(np.nanmin(self.pos.data))
        else:
            x_min, x_max = self.x_minmax

        self.x_edges = np.arange(x_min, x_max + self.s_binsize, self.s_binsize)

        # compute occupancy
        occupancy = self.compute_occupancy_1d(pos_run)

        # compute ratemap (in Hz)
        ratemap = self.compute_ratemap_1d(st_run, pos_run, occupancy)

        # enforce minimum background firing rate
        # background firing rate of xx Hz
        ratemap[ratemap < self.minbgrate] = self.minbgrate

        # enforce minimum background occupancy
        for uu in range(st_run.data.shape[0]):
            ratemap[uu][occupancy < self.min_duration] = 0

        # add to nelpy tuning curve class
        tc = nel.TuningCurve1D(
            ratemap=ratemap,
            extmin=x_min,
            extmax=x_max,
        )

        tc._occupancy = occupancy

        if self.tuning_curve_sigma is not None:
            if self.tuning_curve_sigma > 0:
                tc.smooth(
                    sigma=self.tuning_curve_sigma, inplace=True, mode=self.smooth_mode
                )

        return tc, st_run

    def compute_occupancy_1d(self, pos_run: object):
        occupancy, _ = np.histogram(pos_run.data[0, :], bins=self.x_edges)
        return occupancy / pos_run.fs

    def compute_ratemap_1d(
        self, st_run: object, pos_run: object, occupancy: np.ndarray
    ):
        # initialize ratemap
        ratemap = np.zeros((st_run.data.shape[0], occupancy.shape[0]))

        # if data to map is spike train (point process)
        if isinstance(st_run, nel.core._eventarray.SpikeTrainArray):
            for i in range(st_run.data.shape[0]):
                # get spike counts in each bin
                (
                    ratemap[i, : len(self.x_edges)],
                    _,
                ) = np.histogram(
                    np.interp(
                        st_run.data[i], pos_run.abscissa_vals, pos_run.data[0, :]
                    ),
                    bins=self.x_edges,
                )
            # divide by occupancy
            ratemap = ratemap / occupancy

        # if data to map is analog signal (continous)
        elif isinstance(st_run, nel.core._analogsignalarray.AnalogSignalArray):
            # get x location for every bin center
            x = np.interp(
                st_run.abscissa_vals, pos_run.abscissa_vals, pos_run.data[0, :]
            )
            # get indices location within bin edges
            ext_bin_idx = np.squeeze(np.digitize(x, self.x_edges, right=True))
            # iterate over each time step and add data values to ratemap
            for tt, bidx in enumerate(ext_bin_idx):
                ratemap[:, bidx - 1] += st_run.data[:, tt]
            # divide by sampling rate
            ratemap = ratemap * st_run.fs
            # divide by occupancy
            ratemap = ratemap / occupancy

        # remove nans and infs
        bad_idx = np.isnan(ratemap) | np.isinf(ratemap)
        ratemap[bad_idx] = 0

        return ratemap

    def map_2d(self, pos: object = None):
        # restrict spike trains to those epochs during which the animal was running
        st_run = self.st[self.run_epochs]

        # take pos as input for case of shuffling
        if pos is not None:
            pos_run = pos[self.run_epochs]
        else:
            pos_run = self.pos[self.run_epochs]

        # get xy max min
        if self.x_minmax is None:
            ext_xmin, ext_xmax = (
                np.floor(np.nanmin(self.pos.data[0, :])),
                np.ceil(np.nanmax(self.pos.data[0, :])),
            )
        else:
            ext_xmin, ext_xmax = self.x_minmax

        if self.y_minmax is None:
            ext_ymin, ext_ymax = (
                np.floor(np.nanmin(self.pos.data[1, :])),
                np.ceil(np.nanmax(self.pos.data[1, :])),
            )
        else:
            ext_ymin, ext_ymax = self.y_minmax

        # create bin edges
        self.x_edges = np.arange(ext_xmin, ext_xmax + self.s_binsize, self.s_binsize)
        self.y_edges = np.arange(ext_ymin, ext_ymax + self.s_binsize, self.s_binsize)

        # number of bins in each dimension
        ext_nx, ext_ny = len(self.x_edges), len(self.y_edges)

        # compute occupancy
        occupancy = self.compute_occupancy_2d(pos_run)

        # compute ratemap (in Hz)
        ratemap = self.compute_ratemap_2d(st_run, pos_run, occupancy)

        # enforce minimum background occupancy
        for uu in range(st_run.data.shape[0]):
            ratemap[uu][occupancy < self.min_duration] = 0

        # enforce minimum background firing rate
        # background firing rate of xx Hz
        ratemap[ratemap < self.minbgrate] = self.minbgrate

        tc = nel.TuningCurve2D(
            ratemap=ratemap,
            ext_xmin=ext_xmin,
            ext_ymin=ext_ymin,
            ext_xmax=ext_xmax,
            ext_ymax=ext_ymax,
            ext_ny=ext_ny,
            ext_nx=ext_nx,
        )
        tc._occupancy = occupancy

        if self.tuning_curve_sigma is not None:
            if self.tuning_curve_sigma > 0:
                tc.smooth(
                    sigma=self.tuning_curve_sigma, inplace=True, mode=self.smooth_mode
                )

        return tc, st_run

    def compute_occupancy_2d(self, pos_run: object) -> np.ndarray:
        occupancy, _, _ = np.histogram2d(
            pos_run.data[0, :], pos_run.data[1, :], bins=(self.x_edges, self.y_edges)
        )
        return occupancy / pos_run.fs

    def compute_ratemap_2d(
        self, st_run: object, pos_run: object, occupancy: np.ndarray
    ) -> np.ndarray:
        ratemap = np.zeros(
            (st_run.data.shape[0], occupancy.shape[0], occupancy.shape[1])
        )
        if isinstance(st_run, nel.core._eventarray.SpikeTrainArray):
            for i in range(st_run.data.shape[0]):
                ratemap[i, : len(self.x_edges), : len(self.y_edges)], _, _ = np.histogram2d(
                    np.interp(st_run.data[i], pos_run.abscissa_vals, pos_run.data[0, :]),
                    np.interp(st_run.data[i], pos_run.abscissa_vals, pos_run.data[1, :]),
                    bins=(self.x_edges, self.y_edges),
                )
            ratemap = ratemap / occupancy

        elif isinstance(st_run, nel.core._analogsignalarray.AnalogSignalArray):
            x = np.interp(
                st_run.abscissa_vals, pos_run.abscissa_vals, pos_run.data[0, :]
            )
            y = np.interp(
                st_run.abscissa_vals, pos_run.abscissa_vals, pos_run.data[1, :]
            )
            ext_bin_idx_x = np.squeeze(np.digitize(x, self.x_edges, right=True))
            ext_bin_idx_y = np.squeeze(np.digitize(y, self.y_edges, right=True))
            for tt, (bidxx, bidxy) in enumerate(zip(ext_bin_idx_x, ext_bin_idx_y)):
                ratemap[:, bidxx - 1, bidxy - 1] += st_run.data[:, tt]
            ratemap = ratemap * st_run.fs
            ratemap = ratemap / occupancy

        bad_idx = np.isnan(ratemap) | np.isinf(ratemap)
        ratemap[bad_idx] = 0

        return ratemap

    def shuffle_spatial_information(self) -> np.ndarray:
        def create_shuffled_coordinates(X, n_shuff=500):
            range_ = X.shape[1]

            # if fewer coordinates then shuffles, reduce number of shuffles to n coords
            n_shuff = np.min([range_, n_shuff])

            surrogate = np.random.choice(
                np.arange(-range_, range_), size=n_shuff, replace=False
            )
            x_temp = []
            for n in surrogate:
                x_temp.append(np.roll(X, n, axis=1))

            return x_temp

        def get_spatial_infos(pos_shuff, ts, dim):
            pos_shuff = nel.AnalogSignalArray(
                data=pos_shuff,
                timestamps=ts,
            )
            if dim == 1:
                tc, _ = self.map_1d(pos_shuff)
                return tc.spatial_information()
            elif dim == 2:
                tc, _ = self.map_2d(pos_shuff)
                return tc.spatial_information()

        pos_data_shuff = create_shuffled_coordinates(
            self.pos.data, n_shuff=self.n_shuff
        )

        # construct tuning curves for each position shuffle
        if self.parallel_shuff:
            num_cores = multiprocessing.cpu_count()
            shuffle_spatial_info = Parallel(n_jobs=num_cores)(
                delayed(get_spatial_infos)(
                    pos_data_shuff[i], self.pos.abscissa_vals, self.dim
                )
                for i in range(self.n_shuff)
            )
        else:
            shuffle_spatial_info = [
                get_spatial_infos(pos_data_shuff[i], self.pos.abscissa_vals, self.dim)
                for i in range(self.n_shuff)
            ]

        # calculate p values for the obs vs null
        _, self.spatial_information_pvalues, _ = get_significant_events(
            self.tc.spatial_information(), np.array(shuffle_spatial_info)
        )

        return self.spatial_information_pvalues

    def find_fields(self) -> None:
        """
        Find place fields in the spatial maps.

        args: inherited from Class

        Returns:
            None.

        Attributes:
            field_mask: mask of the place fields (list of numpy arrays).
            n_fields: number of place fields detected (int).
            field_width: width of the place fields (list of floats).
            field_peak_rate: peak firing rate of the place fields (list of floats).
        """
        from skimage import measure

        field_width = []
        peak_rate = []
        mask = []

        if self.place_field_max_size is None and self.dim == 1:
            self.place_field_max_size = self.tc.n_bins * self.s_binsize
        elif self.place_field_max_size is None and self.dim == 2:
            self.place_field_max_size = self.tc.n_bins * self.s_binsize

        if self.dim == 1:
            for ratemap_ in self.tc.ratemap:
                map_fields = fields.map_stats2(
                    ratemap_,
                    threshold=self.place_field_thres,
                    min_size=self.place_field_min_size / self.s_binsize,
                    max_size=self.place_field_max_size / self.s_binsize,
                    min_peak=self.place_field_min_peak,
                    sigma=self.place_field_sigma,
                )
                if len(map_fields["sizes"]) == 0:
                    field_width.append(np.nan)
                    peak_rate.append(np.nan)
                    mask.append(map_fields["fields"])
                else:
                    field_width.append(
                        np.array(map_fields["sizes"]).max()
                        * len(ratemap_)
                        * self.s_binsize
                    )
                    peak_rate.append(np.array(map_fields["peaks"]).max())
                    mask.append(map_fields["fields"])

        if self.dim == 2:
            for ratemap_ in self.tc.ratemap:
                peaks = fields.compute_2d_place_fields(
                    ratemap_,
                    min_firing_rate=self.place_field_min_peak,
                    thresh=self.place_field_thres,
                    min_size=(self.place_field_min_size / self.s_binsize),
                    max_size=(self.place_field_max_size / self.s_binsize),
                    sigma=self.place_field_sigma,
                )
                # field coords of fields using contours
                bc = measure.find_contours(
                    peaks, 0, fully_connected="low", positive_orientation="low"
                )
                if len(bc) == 0:
                    field_width.append(np.nan)
                    peak_rate.append(np.nan)
                    mask.append(peaks)
                elif np.vstack(bc).shape[0] < 3:
                    field_width.append(np.nan)
                    peak_rate.append(np.nan)
                    mask.append(peaks)
                else:
                    field_width.append(
                        np.max(pdist(bc[0], "euclidean")) * self.s_binsize
                    )
                    # field_ids = np.unique(peaks)
                    peak_rate.append(ratemap_[peaks == 1].max())
                    mask.append(peaks)

        self.tc.field_width = np.array(field_width)
        self.tc.field_peak_rate = np.array(peak_rate)
        self.tc.field_mask = np.array(mask)
        self.tc.n_fields = np.array(
            [len(np.unique(mask_)) - 1 for mask_ in self.tc.field_mask]
        )

    def save_mat_file(self, basepath: str, UID=None):
        """
        Save firing rate map data to a .mat file in MATLAB format.

        The saved file will contain the following variables:
        - map: a 1xN cell array containing the ratemaps, where N is the number of ratemaps.
        - field: a 1xN cell array containing the field masks, if they exist.
        - n_fields: the number of fields detected.
        - size: the width of the detected fields.
        - peak: the peak firing rate of the detected fields.
        - occupancy: the occupancy map.
        - spatial_information: the spatial information of the ratemaps.
        - spatial_sparsity: the spatial sparsity of the ratemaps.
        - x_bins: the bin edges for the x-axis of the ratemaps.
        - y_bins: the bin edges for the y-axis of the ratemaps.
        - run_epochs: the time points at which the animal was running.
        - speed: the speed data.
        - timestamps: the timestamps for the speed data.
        - pos: the position data.

        The file will be saved to a .mat file with the name `basepath.ratemap.firingRateMap.mat`, where
        `basepath` is the base path of the data.
        """

        if self.dim == 1:
            raise ValueError("1d storeage not implemented")

        # set up dict
        firingRateMap = {}

        # store UID if exist
        if UID is not None:
            firingRateMap["UID"] = UID.tolist()

        # set up empty fields for conversion to matlab cell array
        firingRateMap["map"] = np.empty(self.tc.ratemap.shape[0], dtype=object)
        firingRateMap["field"] = np.empty(self.tc.ratemap.shape[0], dtype=object)

        # Iterate over the ratemaps and store each one in a cell of the cell array
        for i, ratemap in enumerate(self.tc.ratemap):
            firingRateMap["map"][i] = ratemap

        # store occupancy
        firingRateMap["occupancy"] = self.tc.occupancy

        # store bin edges
        firingRateMap["x_bins"] = self.tc.xbins.tolist()
        firingRateMap["y_bins"] = self.tc.ybins.tolist()

        # store field mask if exist
        if hasattr(self.tc, "field_mask"):
            for i, field_mask in enumerate(self.tc.field_mask):
                firingRateMap["field"][i] = field_mask

            # store field finding info
            firingRateMap["n_fields"] = self.tc.n_fields.tolist()
            firingRateMap["size"] = self.tc.field_width.tolist()
            firingRateMap["peak"] = self.tc.field_peak_rate.tolist()

        # store spatial metrics
        firingRateMap["spatial_information"] = self.tc.spatial_information().tolist()
        if hasattr(self, "spatial_information_pvalues"):
            firingRateMap[
                "spatial_information_pvalues"
            ] = self.spatial_information_pvalues.tolist()
        firingRateMap["spatial_sparsity"] = self.tc.spatial_sparsity().tolist()

        # store position speed and timestamps
        firingRateMap["timestamps"] = self.speed.abscissa_vals.tolist()
        firingRateMap["pos"] = self.pos.data
        firingRateMap["speed"] = self.speed.data.tolist()
        firingRateMap["run_epochs"] = self.run_epochs.time.tolist()

        # store epoch interval
        firingRateMap["epoch_interval"] = [
            self.pos.support.start,
            self.pos.support.stop,
        ]

        # save matlab file
        savemat(
            os.path.join(
                basepath, os.path.basename(basepath) + ".ratemap.firingRateMap.mat"
            ),
            {"firingRateMap": firingRateMap},
        )

    def _unit_subset(self, unit_list):
        newtuningcurve = copy.copy(self)
        newtuningcurve.st = newtuningcurve.st._unit_subset(unit_list)
        newtuningcurve.st_run = newtuningcurve.st_run._unit_subset(unit_list)
        newtuningcurve.tc = self.tc._unit_subset(unit_list)
        return newtuningcurve

    @property
    def is2d(self):
        return self.tc.is2d

    @property
    def occupancy(self):
        return self.tc._occupancy

    @property
    def n_units(self):
        return self.tc.n_units

    @property
    def shape(self):
        return self.tc.shape

    def __repr__(self):
        return self.tc.__repr__()

    @property
    def isempty(self):
        return self.tc.isempty

    @property
    def ratemap(self):
        return self.tc.ratemap

    def __len__(self):
        return self.tc.__len__()

    def smooth(self, **kwargs):
        return self.tc.smooth(**kwargs)

    @property
    def mean(self):
        return self.tc.mean

    @property
    def std(self):
        return self.tc.std

    @property
    def max(self):
        return self.tc.max

    @property
    def min(self):
        return self.tc.min

    @property
    def mask(self):
        return self.tc.mask

    @property
    def n_bins(self):
        return self.tc.n_bins

    @property
    def n_xbins(self):
        return self.tc.n_xbins

    @property
    def n_ybins(self):
        return self.tc.n_ybins

    @property
    def xbins(self):
        return self.tc.xbins

    @property
    def ybins(self):
        return self.tc.ybins

    @property
    def xbin_centers(self):
        return self.tc.xbin_centers

    @property
    def ybin_centers(self):
        return self.tc.ybin_centers

    @property
    def bin_centers(self):
        return self.tc.bin_centers

    @property
    def bins(self):
        return self.tc.bins

    def normalize(self, **kwargs):
        return self.tc.normalize(**kwargs)

    @property
    def spatial_sparsity(self):
        return self.tc.spatial_sparsity

    @property
    def spatial_information(self):
        return self.tc.spatial_information

    @property
    def information_rate(self):
        return self.tc.information_rate

    @property
    def spatial_selectivity(self):
        return self.tc.spatial_selectivity

    def __sub__(self, other):
        return self.tc.__sub__(other)

    def __mul__(self, other):
        return self.tc.__mul__(other)

    def __rmul__(self, other):
        return self.tc.__rmul__(other)

    def __truediv__(self, other):
        return self.tc.__truediv__(other)

    def __iter__(self):
        return self.tc.__iter__()

    def __next__(self):
        return self.tc.__next__()

    def __getitem__(self, *idx):
        return self.tc.__getitem__(*idx)

    def _get_peak_firing_order_idx(self):
        return self.tc._get_peak_firing_order_idx()

    def get_peak_firing_order_ids(self):
        return self.tc.get_peak_firing_order_ids()

    def _reorder_units_by_idx(self):
        return self.tc._reorder_units_by_idx()

    def reorder_units_by_ids(self):
        return self.tc.reorder_units_by_ids()

    def reorder_units(self):
        return self.tc.reorder_units()


# class TuningCurve2DContinuous:
#     """
#     Tuning curves (2-dimensional) of multiple continous variables
#     """

#     __attributes__ = ["_ratemap", "_occupancy"]

#     def __init__(
#         self,
#         *,
#         asa=None,  # nelpy analog signal array of n signals
#         pos=None,  # nelpy positions. Needs time,x,y
#         ext_nx=None,  # number of x bins
#         ext_ny=None,  # number of y bins
#         ext_xmin=None,  # xmin
#         ext_ymin=None,  # ymin
#         ext_xmax=None,  # xmax
#         ext_ymax=None,  # ymax
#         min_duration=0.1,  # duration under this will be nan
#         bin_size=3,  # spatial bin size
#         sigma=None,
#         truncate=None,
#         mask=None
#     ):
#         if mask is None:
#             self._mask = None

#         x = np.interp(
#             asa.abscissa_vals,
#             pos.abscissa_vals,
#             pos.data[0],
#         )
#         y = np.interp(
#             asa.abscissa_vals,
#             pos.abscissa_vals,
#             pos.data[1],
#         )
#         if ext_xmin is None:
#             ext_xmin, ext_xmax = np.floor(x.min() / 10) * 10, np.ceil(x.max() / 10) * 10
#             ext_ymin, ext_ymax = np.floor(y.min() / 10) * 10, np.ceil(y.max() / 10) * 10

#         if ext_nx is None:
#             ext_nx = len(np.arange(ext_xmin, ext_xmax + bin_size, bin_size))
#             ext_ny = len(np.arange(ext_ymin, ext_ymax + bin_size, bin_size))

#         self._xbins = np.linspace(ext_xmin, ext_xmax, ext_nx + 1)
#         self._ybins = np.linspace(ext_ymin, ext_ymax, ext_ny + 1)

#         ext_bin_idx_x = np.squeeze(np.digitize(x, self._xbins, right=True))
#         ext_bin_idx_y = np.squeeze(np.digitize(y, self._ybins, right=True))

#         # n_xbins = len(self._xbins) - 1
#         # n_ybins = len(self._ybins) - 1

#         if ext_bin_idx_x.max() > self.n_xbins:
#             raise ValueError("ext values greater than 'ext_xmax'")
#         if ext_bin_idx_x.min() == 0:
#             raise ValueError("ext values less than 'ext_xmin'")
#         if ext_bin_idx_y.max() > self.n_ybins:
#             raise ValueError("ext values greater than 'ext_ymax'")
#         if ext_bin_idx_y.min() == 0:
#             raise ValueError("ext values less than 'ext_ymin'")

#         self._occupancy, _, _ = np.histogram2d(
#             x,
#             y,
#             bins=[self._xbins, self._ybins],
#             range=([[ext_xmin, ext_xmax], [ext_ymin, ext_ymax]]),
#         )
#         # occupancy = occupancy / asa.fs
#         self._n_signals = asa.n_signals

#         ratemap = np.zeros((self.n_signals, self.n_xbins, self.n_ybins))

#         for tt, (bidxx, bidxy) in enumerate(zip(ext_bin_idx_x, ext_bin_idx_y)):
#             ratemap[:, bidxx - 1, bidxy - 1] += asa.data[:, tt]

#         for uu in range(self.n_signals):
#             ratemap[uu][self._occupancy / asa.fs < min_duration] = 0

#         # ratemap = ratemap * asa.fs

#         denom = np.tile(self._occupancy, (self.n_signals, 1, 1))
#         denom[denom == 0] = 1
#         self._ratemap = ratemap / denom

#         if sigma is not None:
#             if sigma > 0:
#                 self.smooth(sigma=sigma, truncate=truncate, inplace=True)

#         for uu in range(self.n_signals):
#             self._ratemap[uu][self._occupancy / asa.fs < min_duration] = np.nan

#     @property
#     def n_bins(self):
#         """(int) Number of external correlates (bins)."""
#         return self.n_xbins * self.n_ybins

#     @property
#     def n_xbins(self):
#         """(int) Number of external correlates (bins)."""
#         return len(self._xbins) - 1

#     @property
#     def n_ybins(self):
#         """(int) Number of external correlates (bins)."""
#         return len(self._ybins) - 1

#     @property
#     def ratemap(self):
#         return self._ratemap

#     @property
#     def mask(self):
#         return self._mask

#     @property
#     def occupancy(self):
#         return self._occupancy

#     @property
#     def n_signals(self):
#         return self._n_signals

#     @property
#     def xbins(self):
#         """External correlate bins."""
#         return self._xbins

#     @property
#     def ybins(self):
#         """External correlate bins."""
#         return self._ybins

#     @property
#     def shape(self):
#         """(tuple) The shape of the TuningCurve2DContinuous ratemap."""
#         if self.isempty:
#             return (self.n_signals, 0, 0)
#         if len(self.ratemap.shape) == 1:
#             return (self.ratemap.shape[0], 1, 1)
#         return self.ratemap.shape

#     def __repr__(self):
#         address_str = " at " + str(hex(id(self)))
#         if self.isempty:
#             return "<empty TuningCurve2DContinuous" + address_str + ">"
#         shapestr = " with shape (%s, %s, %s)" % (
#             self.shape[0],
#             self.shape[1],
#             self.shape[2],
#         )
#         return "<TuningCurve2D%s>%s" % (address_str, shapestr)

#     @property
#     def isempty(self):
#         """(bool) True if TuningCurve1D is empty"""
#         try:
#             return len(self.ratemap) == 0
#         except TypeError:  # TypeError should happen if ratemap = []
#             return True

#     def __len__(self):
#         return self.n_signals

#     def smooth(self, *, sigma=None, truncate=None, inplace=False, mode=None, cval=None):
#         """Smooths the tuning curve with a Gaussian kernel.
#         mode : {‘reflect’, ‘constant’, ‘nearest’, ‘mirror’, ‘wrap’}, optional
#             The mode parameter determines how the array borders are handled,
#             where cval is the value when mode is equal to ‘constant’. Default is
#             ‘reflect’
#         cval : scalar, optional
#             Value to fill past edges of input if mode is ‘constant’. Default is 0.0
#         """
#         if sigma is None:
#             sigma = 0.1  # in units of extern
#         if truncate is None:
#             truncate = 4
#         if mode is None:
#             mode = "reflect"
#         if cval is None:
#             cval = 0.0

#         ds_x = (self._xbins[-1] - self._xbins[0]) / self.n_xbins
#         ds_y = (self._ybins[-1] - self._ybins[0]) / self.n_ybins
#         sigma_x = sigma / ds_x
#         sigma_y = sigma / ds_y

#         if not inplace:
#             out = copy.deepcopy(self)
#         else:
#             out = self

#         if self.mask is None:
#             if self.n_signals > 1:
#                 out._ratemap = scipy.ndimage.filters.gaussian_filter(
#                     self.ratemap,
#                     sigma=(0, sigma_x, sigma_y),
#                     truncate=truncate,
#                     mode=mode,
#                     cval=cval,
#                 )
#             else:
#                 out._ratemap = scipy.ndimage.filters.gaussian_filter(
#                     self.ratemap,
#                     sigma=(sigma_x, sigma_y),
#                     truncate=truncate,
#                     mode=mode,
#                     cval=cval,
#                 )
#         else:  # we have a mask!
#             # smooth, dealing properly with NANs
#             # NB! see https://stackoverflow.com/questions/18697532/gaussian-filtering-a-image-with-nan-in-python

#             masked_ratemap = self.ratemap.copy() * self.mask
#             V = masked_ratemap.copy()
#             V[masked_ratemap != masked_ratemap] = 0
#             W = 0 * masked_ratemap.copy() + 1
#             W[masked_ratemap != masked_ratemap] = 0

#             if self.n_units > 1:
#                 VV = scipy.ndimage.filters.gaussian_filter(
#                     V,
#                     sigma=(0, sigma_x, sigma_y),
#                     truncate=truncate,
#                     mode=mode,
#                     cval=cval,
#                 )
#                 WW = scipy.ndimage.filters.gaussian_filter(
#                     W,
#                     sigma=(0, sigma_x, sigma_y),
#                     truncate=truncate,
#                     mode=mode,
#                     cval=cval,
#                 )
#                 Z = VV / WW
#                 out._ratemap = Z * self.mask
#             else:
#                 VV = scipy.ndimage.filters.gaussian_filter(
#                     V, sigma=(sigma_x, sigma_y), truncate=truncate, mode=mode, cval=cval
#                 )
#                 WW = scipy.ndimage.filters.gaussian_filter(
#                     W, sigma=(sigma_x, sigma_y), truncate=truncate, mode=mode, cval=cval
#                 )
#                 Z = VV / WW
#                 out._ratemap = Z * self.mask

#         return out


# ########################################################################
# # class TuningCurve1D
# ########################################################################
# class TuningCurve1DContinous:
#     """Tuning curves (1-dimensional) of multiple signals.
#     Get in asa
#     Get in queriable object for external correlates
#     Get in bins, binlabels
#     Get in n_bins, xmin, xmax
#     Get in a transform function f
#     Parameters
#     ----------
#     Attributes
#     ----------
#     """

#     __attributes__ = [
#         "_ratemap",
#         "_occupancy",
#         "_signals_ids",
#         "_signal_labels",
#         "_signal_tags",
#         "_label",
#     ]

#     def __init__(
#         self,
#         *,
#         asa=None,
#         extern=None,
#         ratemap=None,
#         sigma=None,
#         truncate=None,
#         n_extern=None,
#         transform_func=None,
#         minbgrate=None,
#         extmin=None,
#         extmax=None,
#         extlabels=None,
#         signal_ids=None,
#         signal_labels=None,
#         signal_tags=None,
#         label=None,
#         min_duration=None,
#         empty=False
#     ):
#         """
#         If sigma is nonzero, then smoothing is applied.
#         We always require asa and extern, and then some combination of
#             (1) bin edges, transform_func*
#             (2) n_extern, transform_func*
#             (3) n_extern, x_min, x_max, transform_func*
#             transform_func operates on extern and returns a value that TuninCurve1D can interpret. If no transform is specified, the identity operator is assumed.
#         """
#         # TODO: input validation
#         if not empty:
#             if ratemap is None:
#                 assert (
#                     asa is not None
#                 ), "asa must be specified or ratemap must be specified!"
#                 assert (
#                     extern is not None
#                 ), "extern must be specified or ratemap must be specified!"
#             else:
#                 assert asa is None, "ratemap and asa cannot both be specified!"
#                 assert extern is None, "ratemap and extern cannot both be specified!"

#         # if an empty object is requested, return it:
#         if empty:
#             for attr in self.__attributes__:
#                 exec("self." + attr + " = None")
#             return

#         self._asa = asa
#         self._extern = extern

#         if minbgrate is None:
#             minbgrate = 0.01  # Hz minimum background firing rate

#         if n_extern is not None:
#             if extmin is not None and extmax is not None:
#                 self._bins = np.linspace(extmin, extmax, n_extern + 1)
#             else:
#                 raise NotImplementedError
#         else:
#             raise NotImplementedError

#         if min_duration is None:
#             min_duration = 0

#         self._min_duration = min_duration

#         self._signals_ids = np.arange(asa.n_signals)
#         self._signal_labels = None
#         self._signal_tags = None  # no input validation yet
#         self.label = label

#         if transform_func is None:
#             self.trans_func = self._trans_func

#         # compute occupancy
#         self._occupancy = self._compute_occupancy()
#         # compute ratemap (in Hz)
#         self._ratemap = self._compute_ratemap()
#         # normalize firing rate by occupancy
#         self._ratemap = self._normalize_firing_rate_by_occupancy()
#         # enforce minimum background firing rate
#         self._ratemap[
#             self._ratemap < minbgrate
#         ] = minbgrate  # background firing rate of 0.01 Hz

#         if sigma is not None:
#             if sigma > 0:
#                 self.smooth(sigma=sigma, truncate=truncate, inplace=True)

#         # optionally detach _asa and _extern to save space when pickling, for example
#         self._detach()

#     @property
#     def is2d(self):
#         return False

#     def spatial_information(self):
#         """Compute the spatial information and firing sparsity...
#         The specificity index examines the amount of information
#         (in bits) that a single spike conveys about the animal's
#         location (i.e., how well cell firing predicts the animal's
#         location).The spatial information content of cell discharge was
#         calculated using the formula:
#             information content = \Sum P_i(R_i/R)log_2(R_i/R)
#         where i is the bin number, P_i, is the probability for occupancy
#         of bin i, R_i, is the mean firing rate for bin i, and R is the
#         overall mean firing rate.
#         In order to account for the effects of low firing rates (with
#         fewer spikes there is a tendency toward higher information
#         content) or random bursts of firing, the spike firing
#         time-series was randomly offset in time from the rat location
#         time-series, and the information content was calculated. A
#         distribution of the information content based on 100 such random
#         shifts was obtained and was used to compute a standardized score
#         (Zscore) of information content for that cell. While the
#         distribution is not composed of independent samples, it was
#         nominally normally distributed, and a Z value of 2.29 was chosen
#         as a cut-off for significance (the equivalent of a one-tailed
#         t-test with P = 0.01 under a normal distribution).
#         Reference(s)
#         ------------
#         Markus, E. J., Barnes, C. A., McNaughton, B. L., Gladden, V. L.,
#             and Skaggs, W. E. (1994). "Spatial information content and
#             reliability of hippocampal CA1 neurons: effects of visual
#             input", Hippocampus, 4(4), 410-421.
#         Parameters
#         ----------
#         Returns
#         -------
#         si : array of shape (n_signals,)
#             spatial information (in bits) per unit
#         sparsity: array of shape (n_signals,)
#             sparsity (in percent) for each unit
#         """

#         # Pi = self.occupancy / np.sum(self.occupancy)
#         # R = self.ratemap.mean(axis=1) # mean firing rate
#         # Ri = self.ratemap.T
#         # si = np.sum((Pi*((Ri / R)*np.log2(Ri / R)).T), axis=1)

#         # sparsity = np.sum((Pi*Ri.T), axis=1)/(R**2)

#         return utils.spatial_information(ratemap=self.ratemap)

#     def spatial_sparsity(self):
#         """Compute the spatial information and firing sparsity...
#         The specificity index examines the amount of information
#         (in bits) that a single spike conveys about the animal's
#         location (i.e., how well cell firing redicts the animals
#         location).The spatial information content of cell discharge was
#         calculated using the formula:
#             information content = \Sum P_i(R_i/R)log_2(R_i/R)
#         where i is the bin number, P, is the probability for occupancy
#         of bin i, R, is the mean firing rate for bin i, and R is the
#         overall mean firing rate.
#         In order to account for the effects of low firing rates (with
#         fewer spikes there is a tendency toward higher information
#         content) or random bursts of firing, the spike firing
#         time-series was randomly offset in time from the rat location
#         time-series, and the information content was calculated. A
#         distribution of the information content based on 100 such random
#         shifts was obtained and was used to compute a standardized score
#         (Zscore) of information content for that cell. While the
#         distribution is not composed of independent samples, it was
#         nominally normally distributed, and a Z value of 2.29 was chosen
#         as a cut-off for significance (the equivalent of a one-tailed
#         t-test with P = 0.01 under a normal distribution).
#         Reference(s)
#         ------------
#         Markus, E. J., Barnes, C. A., McNaughton, B. L., Gladden, V. L.,
#             and Skaggs, W. E. (1994). "Spatial information content and
#             reliability of hippocampal CA1 neurons: effects of visual
#             input", Hippocampus, 4(4), 410-421.
#         Parameters
#         ----------
#         Returns
#         -------
#         si : array of shape (n_signals,)
#             spatial information (in bits) per unit
#         sparsity: array of shape (n_signals,)
#             sparsity (in percent) for each unit
#         """
#         return utils.spatial_sparsity(ratemap=self.ratemap)

#     def _init_from_ratemap(
#         self,
#         ratemap,
#         occupancy=None,
#         extmin=None,
#         extmax=None,
#         extlabels=None,
#         signal_ids=None,
#         signal_labels=None,
#         signal_tags=None,
#         label=None,
#     ):
#         """Initialize a TuningCurve1D object from a ratemap.
#         Parameters
#         ----------
#         ratemap : array
#             Array of shape (n_signals, n_extern)
#         Returns
#         -------
#         """
#         n_signals, n_extern = ratemap.shape

#         if occupancy is None:
#             # assume uniform occupancy
#             self._occupancy = np.ones(n_extern)

#         if extmin is None:
#             extmin = 0
#         if extmax is None:
#             extmax = extmin + 1

#         self._bins = np.linspace(extmin, extmax, n_extern + 1)
#         self._ratemap = ratemap

#         # inherit unit IDs if available, otherwise initialize to default
#         if signal_ids is None:
#             signal_ids = list(range(1, n_signals + 1))

#         signal_ids = np.array(signal_ids, ndmin=1)  # standardize signal_ids

#         # if signal_labels is empty, default to signal_ids
#         if signal_labels is None:
#             signal_labels = signal_ids

#         signal_labels = np.array(signal_labels, ndmin=1)  # standardize

#         self._signals_ids = signal_ids
#         self._signal_labels = signal_labels
#         self._signal_tags = signal_tags  # no input validation yet
#         if label is not None:
#             self.label = label

#         return self

#     def mean(self, *, axis=None):
#         """Returns the mean of firing rate (in Hz).
#         Parameters
#         ----------
#         axis : int, optional
#             When axis is None, the global mean firing rate is returned.
#             When axis is 0, the mean firing rates across signal, as a
#             function of the external correlate (e.g. position) are
#             returned.
#             When axis is 1, the mean firing rate for each unit is
#             returned.
#         Returns
#         -------
#         mean :
#         """
#         means = np.mean(self.ratemap, axis=axis).squeeze()
#         if means.size == 1:
#             return np.asscalar(means)
#         return means

#     def max(self, *, axis=None):
#         """Returns the mean of firing rate (in Hz).
#         Parameters
#         ----------
#         axis : int, optional
#             When axis is None, the global mean firing rate is returned.
#             When axis is 0, the mean firing rates across signal, as a
#             function of the external correlate (e.g. position) are
#             returned.
#             When axis is 1, the mean firing rate for each unit is
#             returned.
#         Returns
#         -------
#         mean :
#         """
#         maxes = np.max(self.ratemap, axis=axis).squeeze()
#         if maxes.size == 1:
#             return np.asscalar(maxes)
#         return maxes

#     def min(self, *, axis=None):
#         """Returns the mean of firing rate (in Hz).
#         Parameters
#         ----------
#         axis : int, optional
#             When axis is None, the global mean firing rate is returned.
#             When axis is 0, the mean firing rates across signal, as a
#             function of the external correlate (e.g. position) are
#             returned.
#             When axis is 1, the mean firing rate for each unit is
#             returned.
#         Returns
#         -------
#         mean :
#         """
#         mins = np.min(self.ratemap, axis=axis).squeeze()
#         if mins.size == 1:
#             return np.asscalar(mins)
#         return mins

#     @property
#     def ratemap(self):
#         return self._ratemap

#     @property
#     def n_bins(self):
#         """(int) Number of external correlates (bins)."""
#         return len(self.bins) - 1

#     @property
#     def occupancy(self):
#         return self._occupancy

#     @property
#     def bins(self):
#         """External correlate bins."""
#         return self._bins

#     @property
#     def bin_centers(self):
#         """External correlate bin centers."""
#         return (self.bins + (self.bins[1] - self.bins[0]) / 2)[:-1]

#     def _trans_func(self, extern, at):
#         """Default transform function to map extern into numerical bins"""

#         _, ext = extern.asarray(at=at)

#         return np.atleast_1d(ext)

#     def _compute_occupancy(self):

#         # Make sure that self._asa_centers fall within not only the support
#         # of extern, but also within the extreme sample times; otherwise,
#         # interpolation will yield NaNs at the extremes. Indeed, when we have
#         # sample times within a support epoch, we can assume that the signal
#         # stayed roughly constant for that one sample duration.

#         if self._asa._bin_centers[0] < self._extern.time[0]:
#             self._extern = copy.copy(self._extern)
#             self._extern.time[0] = self._asa._bin_centers[0]
#             self._extern._interp = None
#             # raise ValueError('interpolated sample requested before first sample of extern!')
#         if self._asa._bin_centers[-1] > self._extern.time[-1]:
#             self._extern = copy.copy(self._extern)
#             self._extern.time[-1] = self._asa._bin_centers[-1]
#             self._extern._interp = None
#             # raise ValueError('interpolated sample requested after last sample of extern!')

#         ext = self.trans_func(self._extern, at=self._asa.bin_centers)

#         xmin = self.bins[0]
#         xmax = self.bins[-1]
#         occupancy, _ = np.histogram(ext, bins=self.bins, range=(xmin, xmax))
#         # xbins = (bins + xmax/n_xbins)[:-1] # for plotting
#         return occupancy

#     def _compute_ratemap(self, min_duration=None):

#         if min_duration is None:
#             min_duration = self._min_duration

#         ext = self.trans_func(self._extern, at=self._asa.bin_centers)

#         ext_bin_idx = np.squeeze(np.digitize(ext, self.bins, right=True))
#         # make sure that all the events fit between extmin and extmax:
#         # TODO: this might rather be a warning, but it's a pretty serious warning...
#         if ext_bin_idx.max() > self.n_bins:
#             raise ValueError("ext values greater than 'ext_max'")
#         if ext_bin_idx.min() == 0:
#             raise ValueError("ext values less than 'ext_min'")

#         ratemap = np.zeros((self.n_signals, self.n_bins))

#         for tt, bidx in enumerate(ext_bin_idx):
#             ratemap[:, bidx - 1] += self._asa.data[:, tt]

#         # apply minimum observation duration
#         for uu in range(self.n_signals):
#             ratemap[uu][self.occupancy * self._asa.ds < min_duration] = 0

#         return ratemap

#     def normalize(self, inplace=False):

#         if not inplace:
#             out = copy.deepcopy(self)
#         else:
#             out = self
#         if self.n_signals > 1:
#             per_unit_max = np.max(out.ratemap, axis=1)[..., np.newaxis]
#             out._ratemap = self.ratemap / np.tile(per_unit_max, (1, out.n_bins))
#         else:
#             per_unit_max = np.max(out.ratemap)
#             out._ratemap = self.ratemap / np.tile(per_unit_max, out.n_bins)
#         return out

#     def _normalize_firing_rate_by_occupancy(self):
#         # normalize spike counts by occupancy:
#         denom = np.tile(self.occupancy, (self.n_signals, 1))
#         denom[denom == 0] = 1
#         ratemap = self.ratemap / denom
#         return ratemap

#     @property
#     def signal_ids(self):
#         """Unit IDs contained in the SpikeTrain."""
#         return list(self._signals_ids)

#     @signal_ids.setter
#     def signal_ids(self, val):
#         if len(val) != self.n_signals:
#             # print(len(val))
#             # print(self.n_signals)
#             raise TypeError("signal_ids must be of length n_signals")
#         elif len(set(val)) < len(val):
#             raise TypeError("duplicate signal_ids are not allowed")
#         else:
#             try:
#                 # cast to int:
#                 signal_ids = [int(id) for id in val]
#             except TypeError:
#                 raise TypeError("signal_ids must be int-like")
#         self._signals_ids = signal_ids

#     @property
#     def signal_labels(self):
#         """Labels corresponding to signal contained in the SpikeTrain."""
#         if self._signal_labels is None:
#             warnings.warn("unit labels have not yet been specified")
#         return self._signal_labels

#     @signal_labels.setter
#     def signal_labels(self, val):
#         if len(val) != self.n_signals:
#             raise TypeError("labels must be of length n_signals")
#         else:
#             try:
#                 # cast to str:
#                 labels = [str(label) for label in val]
#             except TypeError:
#                 raise TypeError("labels must be string-like")
#         self._signal_labels = labels

#     @property
#     def signal_tags(self):
#         """Tags corresponding to signal contained in the SpikeTrain"""
#         if self._signal_tags is None:
#             warnings.warn("unit tags have not yet been specified")
#         return self._signal_tags

#     @property
#     def label(self):
#         """Label pertaining to the source of the spike train."""
#         if self._label is None:
#             warnings.warn("label has not yet been specified")
#         return self._label

#     @label.setter
#     def label(self, val):
#         if val is not None:
#             try:  # cast to str:
#                 label = str(val)
#             except TypeError:
#                 raise TypeError("cannot convert label to string")
#         else:
#             label = val
#         self._label = label

#     def __add__(self, other):
#         out = copy.copy(self)

#         if isinstance(other, numbers.Number):
#             out._ratemap = out.ratemap + other
#         elif isinstance(other, TuningCurve1DContinous):
#             # TODO: this should merge two TuningCurve1D objects
#             raise NotImplementedError
#         else:
#             raise TypeError(
#                 "unsupported operand type(s) for +: 'TuningCurve1DContinous' and '{}'".format(
#                     str(type(other))
#                 )
#             )
#         return out

#     def __sub__(self, other):
#         out = copy.copy(self)
#         out._ratemap = out.ratemap - other
#         return out

#     def __mul__(self, other):
#         """overloaded * operator."""
#         out = copy.copy(self)
#         out._ratemap = out.ratemap * other
#         return out

#     def __rmul__(self, other):
#         return self * other

#     def __truediv__(self, other):
#         """overloaded / operator."""
#         out = copy.copy(self)
#         out._ratemap = out.ratemap / other
#         return out

#     def __len__(self):
#         return self.n_signals

#     def smooth(self, *, sigma=None, truncate=None, inplace=False, mode=None, cval=None):
#         """Smooths the tuning curve with a Gaussian kernel.
#         mode : {‘reflect’, ‘constant’, ‘nearest’, ‘mirror’, ‘wrap’}, optional
#             The mode parameter determines how the array borders are handled,
#             where cval is the value when mode is equal to ‘constant’. Default is
#             ‘reflect’
#         cval : scalar, optional
#             Value to fill past edges of input if mode is ‘constant’. Default is 0.0
#         """
#         if sigma is None:
#             sigma = 0.1  # in signal of extern
#         if truncate is None:
#             truncate = 4
#         if mode is None:
#             mode = "reflect"
#         if cval is None:
#             cval = 0.0

#         ds = (self.bins[-1] - self.bins[0]) / self.n_bins
#         sigma = sigma / ds

#         if not inplace:
#             out = copy.deepcopy(self)
#         else:
#             out = self

#         if self.n_signals > 1:
#             out._ratemap = scipy.ndimage.filters.gaussian_filter(
#                 self.ratemap, sigma=(0, sigma), truncate=truncate, mode=mode, cval=cval
#             )
#         else:
#             out._ratemap = scipy.ndimage.filters.gaussian_filter(
#                 self.ratemap, sigma=sigma, truncate=truncate, mode=mode, cval=cval
#             )

#         return out

#     @property
#     def n_signals(self):
#         """(int) The number of signal."""
#         try:
#             return len(self._signals_ids)
#         except TypeError:  # when signal_ids is an integer
#             return 1
#         except AttributeError:
#             return 0

#     @property
#     def shape(self):
#         """(tuple) The shape of the TuningCurve1DContinous ratemap."""
#         if self.isempty:
#             return (self.n_signals, 0)
#         if len(self.ratemap.shape) == 1:
#             return (1, self.ratemap.shape[0])
#         return self.ratemap.shape

#     def __repr__(self):
#         address_str = " at " + str(hex(id(self)))
#         if self.isempty:
#             return "<empty TuningCurve1DContinous" + address_str + ">"
#         shapestr = " with shape (%s, %s)" % (self.shape[0], self.shape[1])
#         return "<TuningCurve1DContinous%s>%s" % (address_str, shapestr)

#     @property
#     def isempty(self):
#         """(bool) True if TuningCurve1DContinous is empty"""
#         try:
#             return len(self.ratemap) == 0
#         except TypeError:  # TypeError should happen if ratemap = []
#             return True

#     def __iter__(self):
#         """TuningCurve1D iterator initialization"""
#         # initialize the internal index to zero when used as iterator
#         self._index = 0
#         return self

#     def __next__(self):
#         """TuningCurve1D iterator advancer."""
#         index = self._index
#         if index > self.n_signals - 1:
#             raise StopIteration
#         out = copy.copy(self)
#         out._ratemap = self.ratemap[index, :]
#         out._signals_ids = self.signal_ids[index]
#         out._signal_labels = self.signal_labels[index]
#         self._index += 1
#         return out

#     def __getitem__(self, *idx):
#         """TuningCurve1D index access.
#         Accepts integers, slices, and lists"""

#         idx = [ii for ii in idx]
#         if len(idx) == 1 and not isinstance(idx[0], int):
#             idx = idx[0]
#         if isinstance(idx, tuple):
#             idx = [ii for ii in idx]

#         if self.isempty:
#             return self
#         try:
#             out = copy.copy(self)
#             out._ratemap = self.ratemap[idx, :]
#             out._signals_ids = (np.asanyarray(out._signals_ids)[idx]).tolist()
#             out._signal_labels = (np.asanyarray(out._signal_labels)[idx]).tolist()
#             return out
#         except Exception:
#             raise TypeError("unsupported subsctipting type {}".format(type(idx)))

#     def _unit_subset(self, signal_list):
#         """Return a TuningCurve1D restricted to a subset of signal.
#         Parameters
#         ----------
#         signal_list : array-like
#             Array or list of signal_ids.
#         """
#         unit_subset_ids = []
#         for unit in signal_list:
#             try:
#                 id = self.signal_ids.index(unit)
#             except ValueError:
#                 warnings.warn(
#                     "unit_id " + str(unit) + " not found in TuningCurve1D; ignoring"
#                 )
#                 pass
#             else:
#                 unit_subset_ids.append(id)

#         new_signals_ids = (np.asarray(self.signal_ids)[unit_subset_ids]).tolist()
#         new_signal_labels = (np.asarray(self.signal_labels)[unit_subset_ids]).tolist()

#         if len(unit_subset_ids) == 0:
#             warnings.warn("no signal remaining in requested unit subset")
#             return TuningCurve1DContinous(empty=True)

#         newtuningcurve = copy.copy(self)
#         newtuningcurve._signals_ids = new_signals_ids
#         newtuningcurve._signal_labels = new_signal_labels
#         # TODO: implement tags
#         # newtuningcurve._signal_tags =
#         newtuningcurve._ratemap = self.ratemap[unit_subset_ids, :]
#         # TODO: shall we restrict _asa as well? This will require a copy to be made...
#         # newtuningcurve._asa =

#         return newtuningcurve

#     def _get_peak_firing_order_idx(self):
#         """Docstring goes here
#         ratemap has shape (n_signals, n_ext)
#         """
#         peakorder = np.argmax(self.ratemap, axis=1).argsort()

#         return peakorder.tolist()

#     def get_peak_firing_order_ids(self):
#         """Docstring goes here
#         ratemap has shape (n_signals, n_ext)
#         """
#         peakorder = np.argmax(self.ratemap, axis=1).argsort()

#         return (np.asanyarray(self.signal_ids)[peakorder]).tolist()

#     def _reorder_signal_by_idx(self, neworder=None, *, inplace=False):
#         """Reorder signal according to a specified order.
#         neworder must be list-like, of size (n_signals,) and in 0,..n_signals
#         and not in terms of signal_ids
#         Return
#         ------
#         out : reordered TuningCurve1D
#         """
#         if neworder is None:
#             neworder = self._get_peak_firing_order_idx()
#         if inplace:
#             out = self
#         else:
#             out = copy.deepcopy(self)

#         oldorder = list(range(len(neworder)))
#         for oi, ni in enumerate(neworder):
#             frm = oldorder.index(ni)
#             to = oi
#             utils.swap_rows(out._ratemap, frm, to)
#             out._signals_ids[frm], out._signals_ids[to] = (
#                 out._signals_ids[to],
#                 out._signals_ids[frm],
#             )
#             out._signal_labels[frm], out._signal_labels[to] = (
#                 out._signal_labels[to],
#                 out._signal_labels[frm],
#             )
#             # TODO: re-build unit tags (tag system not yet implemented)
#             oldorder[frm], oldorder[to] = oldorder[to], oldorder[frm]

#         return out

#     def reorder_signal_by_ids(self, neworder=None, *, inplace=False):
#         """Reorder signal according to a specified order.
#         neworder must be list-like, of size (n_signals,) and in terms of
#         signal_ids
#         Return
#         ------
#         out : reordered TuningCurve1D
#         """
#         if neworder is None:
#             neworder = self.get_peak_firing_order_ids()
#         if inplace:
#             out = self
#         else:
#             out = copy.deepcopy(self)

#         # signal_ids = list(signal_ids)
#         neworder = [self.signal_ids.index(x) for x in neworder]

#         oldorder = list(range(len(neworder)))
#         for oi, ni in enumerate(neworder):
#             frm = oldorder.index(ni)
#             to = oi
#             utils.swap_rows(out._ratemap, frm, to)
#             out._signals_ids[frm], out._signals_ids[to] = (
#                 out._signals_ids[to],
#                 out._signals_ids[frm],
#             )
#             out._signal_labels[frm], out._signal_labels[to] = (
#                 out._signal_labels[to],
#                 out._signal_labels[frm],
#             )
#             # TODO: re-build unit tags (tag system not yet implemented)
#             oldorder[frm], oldorder[to] = oldorder[to], oldorder[frm]

#         return out

#     def reorder_signal(self, inplace=False):
#         """Convenience function to reorder signal by peak firing location."""
#         return self.reorder_signal_by_ids(inplace=inplace)

#     def _detach(self):
#         """Detach asa and extern from tuning curve."""
#         self._asa = None
#         self._extern = None
