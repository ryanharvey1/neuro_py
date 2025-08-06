import copy
import logging
import multiprocessing
import os
import warnings
from typing import Any, List, Optional, Union

import nelpy as nel
import numpy as np
from joblib import Parallel, delayed
from scipy.io import savemat
from scipy.spatial.distance import pdist

from neuro_py.stats.stats import get_significant_events
from neuro_py.tuning import fields


class NDimensionalBinner:
    """
    Base class for N-dimensional binning of data (point process or continuous)
    over arbitrary dimensions.

    This class provides low-level functionality to create N-dimensional tuning curves
    by binning spike train or continuous data along multiple spatial or behavioral dimensions.
    """

    def __init__(self):
        pass

    def create_nd_tuning_curve(
        self,
        st_data: object,
        pos_data: object,
        bin_edges: List[np.ndarray],
        min_duration: float = 0.1,
        minbgrate: Union[int, float] = 0,
        tuning_curve_sigma: Optional[
            Union[int, float, List[Union[int, float]], np.ndarray]
        ] = None,
        smooth_mode: str = "reflect",
    ) -> tuple:
        """
        Create an N-dimensional tuning curve from spike and position data.

        Parameters
        ----------
        st_data : object
            Spike train data (nelpy.SpikeTrain or nelpy.AnalogSignal).
        pos_data : object
            Position data (nelpy.AnalogSignal or nel.PositionArray).
        bin_edges : List[np.ndarray]
            List of bin edges for each dimension.
        min_duration : float, optional
            Minimum duration for a valid bin. Default is 0.1.
        minbgrate : Union[int, float], optional
            Minimum background firing rate. Default is 0.
        tuning_curve_sigma : Optional[Union[int, float, List[Union[int, float]], np.ndarray]], optional
            Sigma for smoothing. Can be a single value (used for all dimensions)
            or an array/list with sigma values for each dimension. Default is None (no smoothing).
        smooth_mode : str, optional
            Smoothing mode. Default is "reflect".

        Returns
        -------
        tuple
            A tuple containing (tuning_curve, occupancy, ratemap)
        """
        # Determine number of dimensions
        n_dims = len(bin_edges)

        if n_dims != pos_data.n_signals:
            raise ValueError(
                f"Number of bin_edges ({n_dims}) must match position dimensions ({pos_data.n_signals})"
            )

        # Compute occupancy
        occupancy = self._compute_occupancy_nd(pos_data, bin_edges)

        # Compute ratemap
        ratemap = self._compute_ratemap_nd(st_data, pos_data, occupancy, bin_edges)

        # Apply minimum background firing rate (before minimum duration check)
        ratemap[ratemap < minbgrate] = minbgrate

        # Apply minimum background occupancy (after minimum background firing rate)
        for uu in range(st_data.data.shape[0]):
            ratemap[uu][occupancy < min_duration] = 0

        # Create appropriate tuning curve object based on dimensions
        if n_dims == 1:
            tc = nel.TuningCurve1D(
                ratemap=ratemap,
                extmin=bin_edges[0][0],
                extmax=bin_edges[0][-1],
            )
        elif n_dims == 2:
            tc = nel.TuningCurve2D(
                ratemap=ratemap,
                ext_xmin=bin_edges[0][0],
                ext_ymin=bin_edges[1][0],
                ext_xmax=bin_edges[0][-1],
                ext_ymax=bin_edges[1][-1],
                ext_ny=len(bin_edges[1]) - 1,
                ext_nx=len(bin_edges[0]) - 1,
            )
        else:
            # For N-dimensional (N > 2), use nelpy's TuningCurveND class
            # Calculate extents for each dimension
            ext_min = [bin_edges[i][0] for i in range(n_dims)]
            ext_max = [bin_edges[i][-1] for i in range(n_dims)]
            tc = nel.TuningCurveND(ratemap=ratemap, ext_min=ext_min, ext_max=ext_max)

        tc._occupancy = occupancy

        # Apply smoothing if requested
        if tuning_curve_sigma is not None:
            # Handle array or scalar sigma values
            if np.isscalar(tuning_curve_sigma):
                # Single value: use for all dimensions
                if tuning_curve_sigma > 0 and hasattr(tc, "smooth"):
                    tc.smooth(sigma=tuning_curve_sigma, inplace=True, mode=smooth_mode)
            else:
                # Array/list: convert to numpy array
                sigma_array = np.asarray(tuning_curve_sigma)
                if len(sigma_array) != n_dims:
                    raise ValueError(
                        f"Length of tuning_curve_sigma array ({len(sigma_array)}) must match "
                        f"number of dimensions ({n_dims})"
                    )

                # Check if any sigma values are positive and smooth if so
                if np.any(sigma_array > 0) and hasattr(tc, "smooth"):
                    tc.smooth(sigma=sigma_array, inplace=True, mode=smooth_mode)

        return tc, occupancy, ratemap

    def _compute_occupancy_nd(
        self, pos_data: object, bin_edges: List[np.ndarray]
    ) -> np.ndarray:
        """
        Compute N-dimensional occupancy.

        Parameters
        ----------
        pos_data : object
            Position data.
        bin_edges : List[np.ndarray]
            Bin edges for each dimension.

        Returns
        -------
        np.ndarray
            N-dimensional occupancy array.
        """
        n_dims = len(bin_edges)

        if n_dims == 1:
            occupancy, _ = np.histogram(pos_data.data[0, :], bins=bin_edges[0])
        elif n_dims == 2:
            occupancy, _, _ = np.histogram2d(
                pos_data.data[0, :],
                pos_data.data[1, :],
                bins=(bin_edges[0], bin_edges[1]),
            )
        else:
            # For N-dimensional histograms
            occupancy, _ = np.histogramdd(
                pos_data.data.T,  # Transpose to get (n_samples, n_dims)
                bins=bin_edges,
            )

        return occupancy / pos_data.fs

    def _compute_ratemap_nd(
        self,
        st_data: object,
        pos_data: object,
        occupancy: np.ndarray,
        bin_edges: List[np.ndarray],
    ) -> np.ndarray:
        """
        Compute N-dimensional ratemap.

        Parameters
        ----------
        st_data : object
            Spike train data.
        pos_data : object
            Position data.
        occupancy : np.ndarray
            Occupancy array.
        bin_edges : List[np.ndarray]
            Bin edges for each dimension.

        Returns
        -------
        np.ndarray
            N-dimensional ratemap.
        """
        n_dims = len(bin_edges)

        # Initialize ratemap with proper shape
        ratemap_shape = [st_data.data.shape[0]] + list(occupancy.shape)
        ratemap = np.zeros(ratemap_shape)

        if st_data.isempty:
            return ratemap

        # Remove NaNs from position data for interpolation
        mask = ~np.isnan(pos_data.data).any(axis=0)
        pos_clean = pos_data.data[:, mask]
        ts_clean = pos_data.abscissa_vals[mask]

        # Handle point process data (spike trains)
        if isinstance(st_data, nel.core._eventarray.SpikeTrainArray):
            for i in range(st_data.data.shape[0]):
                # Interpolate spike positions
                spike_positions = []
                for dim in range(n_dims):
                    spike_pos_dim = np.interp(st_data.data[i], ts_clean, pos_clean[dim])
                    spike_positions.append(spike_pos_dim)

                # Bin spikes
                if n_dims == 1:
                    counts, _ = np.histogram(spike_positions[0], bins=bin_edges[0])
                    ratemap[i] = counts
                elif n_dims == 2:
                    counts, _, _ = np.histogram2d(
                        spike_positions[0],
                        spike_positions[1],
                        bins=(bin_edges[0], bin_edges[1]),
                    )
                    ratemap[i] = counts
                else:
                    counts, _ = np.histogramdd(
                        np.array(spike_positions).T, bins=bin_edges
                    )
                    ratemap[i] = counts

        # Handle continuous data (analog signals)
        elif isinstance(st_data, nel.core._analogsignalarray.AnalogSignalArray):
            # Interpolate position at signal timestamps
            signal_positions = []
            for dim in range(n_dims):
                pos_interp = np.interp(st_data.abscissa_vals, ts_clean, pos_clean[dim])
                signal_positions.append(pos_interp)

            # Get bin indices for each time point
            bin_indices = []
            for dim in range(n_dims):
                bin_idx = np.digitize(signal_positions[dim], bin_edges[dim], right=True)
                bin_indices.append(bin_idx)

            # Accumulate signal values in appropriate bins
            for tt in range(len(st_data.abscissa_vals)):
                # Convert to 0-based indexing and ensure within bounds
                indices = []
                valid = True
                for dim in range(n_dims):
                    idx = bin_indices[dim][tt] - 1
                    if idx < 0 or idx >= len(bin_edges[dim]) - 1:
                        valid = False
                        break
                    indices.append(idx)

                if valid:
                    if n_dims == 1:
                        ratemap[:, indices[0]] += st_data.data[:, tt]
                    elif n_dims == 2:
                        ratemap[:, indices[0], indices[1]] += st_data.data[:, tt]
                    else:
                        # For N-dimensional indexing
                        full_indices = tuple([slice(None)] + indices)
                        ratemap[full_indices] += st_data.data[:, tt]

            # Convert from counts to rate
            ratemap = ratemap * st_data.fs

        # Normalize by occupancy
        # Handle division by zero
        occupancy_expanded = np.expand_dims(occupancy, 0)  # Add unit dimension
        occupancy_tiled = np.tile(
            occupancy_expanded, (ratemap.shape[0],) + (1,) * n_dims
        )

        np.divide(ratemap, occupancy_tiled, where=occupancy_tiled != 0, out=ratemap)

        # Remove NaNs and infs
        bad_idx = np.isnan(ratemap) | np.isinf(ratemap)
        ratemap[bad_idx] = 0

        return ratemap


class SpatialMap(NDimensionalBinner):
    """
    SpatialMap: make a spatial map tuning curve
        maps timestamps or continuous signals onto positions

    Parameters
    ----------
    pos : object
        Position data (nelpy.AnalogSignal or nel.PositionArray).
    st : object
        Spike train data (nelpy.SpikeTrain or nelpy.AnalogSignal).
    speed : Optional[object]
        Speed data (nelpy.AnalogSignal), recommended input: from non-epoched data.
    dim : Optional[int]
        Dimension of the map (1 or 2) *deprecated*.
    dir_epoch : Optional[object]
        Epochs of the running direction, for linear data (nelpy.Epoch) *deprecated*.
    speed_thres : Union[int, float], optional
        Speed threshold for running. Default is 4.
    s_binsize : Union[int, float, List[Union[int, float]], np.ndarray], optional
        Bin size for the spatial map. Can be a single value (used for all dimensions)
        or an array/list with bin sizes for each dimension. Default is 3.
    x_minmax : Optional[List[Union[int, float]]], optional
        Min and max x values for the spatial map.
    y_minmax : Optional[List[Union[int, float]]], optional
        Min and max y values for the spatial map.
    dim_minmax : Optional[List[List[Union[int, float]]]], optional
        Min and max values for each dimension. Should be a list of [min, max] pairs,
        one for each dimension. If provided, takes precedence over x_minmax and y_minmax.
        Example: [[0, 100], [-50, 50], [0, 200]] for 3D data.
    tuning_curve_sigma : Union[int, float, List[Union[int, float]], np.ndarray], optional
        Sigma for the tuning curve. Can be a single value (used for all dimensions)
        or an array/list with sigma values for each dimension. Default is 3.
    smooth_mode : str, optional
        Mode for smoothing curve (str) reflect, constant, nearest, mirror, wrap. Default is "reflect".
    min_duration : float, optional
        Minimum duration for a tuning curve. Default is 0.1.
    minbgrate : Union[int, float], optional
        Minimum firing rate for tuning curve; will set to this if lower. Default is 0.
    n_shuff : int, optional
        Number of position shuffles for spatial information. Default is 500.
    parallel_shuff : bool, optional
        Parallelize shuffling. Default is True.
    place_field_thres : Union[int, float], optional
        Percent of continuous region of peak firing rate. Default is 0.2.
    place_field_min_size : Optional[Union[int, float]]
        Minimum size of place field (cm).
    place_field_max_size : Optional[Union[int, float]]
        Maximum size of place field (cm).
    place_field_min_peak : Union[int, float], optional
        Minimum peak rate of place field. Default is 3.
    place_field_sigma : Union[int, float], optional
        Extra smoothing sigma to apply before field detection. Default is 2.

    Attributes
    ----------
    tc : nelpy.TuningCurve
        Tuning curves.
    st_run : nelpy.SpikeTrain
        Spike train restricted to running epochs.
    bst_run : nelpy.binnedSpikeTrain
        Binned spike train restricted to running epochs.
    speed : Optional[nnelpy.AnalogSignal]
        Speed data.
    run_epochs : nelpy.EpochArray
        Running epochs.

    Notes
    -----
    Place field detector (.find_fields()) is sensitive to many parameters.
    For 2D, it is highly recommended to have good environmental sampling.
    In brief testing with 300cm linear track, optimal 1D parameters were:
        place_field_min_size=15
        place_field_max_size=None
        place_field_min_peak=3
        place_field_sigma=None
        place_field_thres=.33

    TODO
    ----
    Place field detector currently collects field width and peak rate for peak place field.
    In the future, these should be stored for all sub fields.

    Examples
    --------
    >>> import nelpy as nel
    >>> from neuro_py.tuning.maps import SpatialMap
    >>> # Create synthetic position and spike data
    >>> pos = nel.AnalogSignalArray(data=np.random.rand(2, 1000)*20, timestamps=np.linspace(0, 100, 1000))
    >>> st = nel.SpikeTrainArray(time=np.sort(np.random.rand(10, 1000), axis=1), fs=1000.0)
    >>> # Create a spatial map with 3 cm bins
    >>> spatial_map = SpatialMap(pos=pos, st=st, s_binsize=3)
    >>> print(spatial_map)
    <TuningCurve2D at 0x21ea37a9110> with shape (10, 7, 7)
    """

    def __init__(
        self,
        pos: object,
        st: object,
        speed: Optional[object] = None,
        dim: Optional[int] = None,  # deprecated
        dir_epoch: Optional[object] = None,  # deprecated
        speed_thres: Union[int, float] = 4,
        s_binsize: Union[int, float, List[Union[int, float]], np.ndarray] = 3,
        tuning_curve_sigma: Union[int, float, List[Union[int, float]], np.ndarray] = 3,
        x_minmax: Optional[List[Union[int, float]]] = None,
        y_minmax: Optional[List[Union[int, float]]] = None,
        dim_minmax: Optional[List[List[Union[int, float]]]] = None,
        smooth_mode: str = "reflect",
        min_duration: float = 0.1,
        minbgrate: Union[int, float] = 0,
        n_shuff: int = 500,
        parallel_shuff: bool = True,
        place_field_thres: Union[int, float] = 0.2,
        place_field_min_size: Optional[Union[int, float]] = None,
        place_field_max_size: Optional[Union[int, float]] = None,
        place_field_min_peak: Union[int, float] = 3,
        place_field_sigma: Union[int, float] = 2,
    ) -> None:
        # Initialize the parent class
        super().__init__()

        # add all the inputs to self
        self.__dict__.update(locals())
        del self.__dict__["self"]

        # Handle s_binsize input: normalize to array format
        self.dim = pos.n_signals
        if np.isscalar(s_binsize):
            # Single value: use for all dimensions
            self.s_binsize_array = np.full(self.dim, s_binsize)
        else:
            # Array/list: convert to numpy array
            self.s_binsize_array = np.asarray(s_binsize)
            if len(self.s_binsize_array) != self.dim:
                raise ValueError(
                    f"Length of s_binsize array ({len(self.s_binsize_array)}) must match "
                    f"number of position dimensions ({self.dim})"
                )

        # Keep original s_binsize for backward compatibility in some methods
        if np.isscalar(s_binsize):
            self.s_binsize = s_binsize
        else:
            # For backward compatibility, use the first dimension's bin size
            self.s_binsize = self.s_binsize_array[0]

        # Handle dim_minmax input: normalize min/max values for each dimension
        if dim_minmax is not None:
            # Convert to numpy array for easier handling
            self.dim_minmax_array = np.asarray(dim_minmax)
            if self.dim_minmax_array.shape != (self.dim, 2):
                raise ValueError(
                    f"dim_minmax must be a list of [min, max] pairs with shape ({self.dim}, 2), "
                    f"got shape {self.dim_minmax_array.shape}"
                )
            # Override x_minmax and y_minmax if provided
            if self.dim >= 1:
                self.x_minmax = list(self.dim_minmax_array[0])
            if self.dim >= 2:
                self.y_minmax = list(self.dim_minmax_array[1])
        else:
            # Create dim_minmax_array from existing x_minmax, y_minmax, or auto-determine
            self.dim_minmax_array = np.zeros((self.dim, 2))
            for dim_idx in range(self.dim):
                if dim_idx == 0 and self.x_minmax is not None:
                    self.dim_minmax_array[dim_idx] = self.x_minmax
                elif dim_idx == 1 and self.y_minmax is not None:
                    self.dim_minmax_array[dim_idx] = self.y_minmax
                else:
                    # Auto-determine min/max for this dimension
                    self.dim_minmax_array[dim_idx, 0] = np.floor(
                        np.nanmin(pos.data[dim_idx, :])
                    )
                    self.dim_minmax_array[dim_idx, 1] = np.ceil(
                        np.nanmax(pos.data[dim_idx, :])
                    )

        # Handle tuning_curve_sigma input: normalize to array format
        if np.isscalar(tuning_curve_sigma):
            # Single value: use for all dimensions
            self.tuning_curve_sigma_array = np.full(self.dim, tuning_curve_sigma)
        else:
            # Array/list: convert to numpy array
            self.tuning_curve_sigma_array = np.asarray(tuning_curve_sigma)
            if len(self.tuning_curve_sigma_array) != self.dim:
                raise ValueError(
                    f"Length of tuning_curve_sigma array ({len(self.tuning_curve_sigma_array)}) must match "
                    f"number of position dimensions ({self.dim})"
                )

        # Keep original tuning_curve_sigma for backward compatibility in some methods
        if np.isscalar(tuning_curve_sigma):
            self.tuning_curve_sigma = tuning_curve_sigma
        else:
            # For backward compatibility, use the first dimension's sigma
            self.tuning_curve_sigma = self.tuning_curve_sigma_array[0]

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

        # check data is not empty
        if pos.isempty or st.isempty:
            raise ValueError("pos and st must not be empty")

        # check if pos all nan
        if np.all(np.isnan(pos.data)):
            raise ValueError("Position data cannot contain all NaN values")

        # get speed and running epochs (highly recommended you calculate
        #   speed before hand on non epoched data)
        if speed_thres > 0:
            if self.speed is None:
                self.speed = nel.utils.ddt_asa(
                    self.pos, smooth=True, sigma=0.1, norm=True
                )

            self.run_epochs = nel.utils.get_run_epochs(
                self.speed, v1=self.speed_thres, v2=self.speed_thres
            ).merge()
        else:
            self.run_epochs = self.pos.support.copy()

        # calculate maps, 1d, 2d, or N-dimensional
        self.dim = pos.n_signals
        if pos.n_signals == 2:
            self.tc, self.st_run = self.map_2d()
        elif pos.n_signals == 1:
            self.tc, self.st_run = self.map_1d()
        elif pos.n_signals > 2:
            self.tc, self.st_run = self.map_nd()
        else:
            raise ValueError("pos dims must be >= 1")

        # find place fields. Currently only collects metrics from peak field
        # self.find_fields()

    def map_1d(
        self, pos: Optional[object] = None, use_base_class: bool = False
    ) -> tuple:
        """Maps 1D data for the spatial tuning curve.

        Parameters
        ----------
        pos : Optional[object]
            Position data for shuffling.
        use_base_class : bool, optional
            Whether to use the new NDimensionalBinner base class functionality.
            Default is False to maintain backward compatibility.

        Returns
        -------
        tuple
            A tuple containing the tuning curve and restricted spike train.
        """
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

        # log warning if st_run is empty following restriction
        if st_run.isempty:
            logging.warning(
                "No spike trains during running epochs"
            )  # This will log it but not raise a warning
            warnings.warn("No spike trains during running epochs", UserWarning)

        # take pos as input for case of shuffling
        if pos is not None:
            pos_run = pos[self.run_epochs]
        else:
            pos_run = self.pos[self.run_epochs]

        # Use dimension-specific min/max values (dimension 0 for 1D)
        x_min, x_max = self.dim_minmax_array[0]

        # Use dimension-specific bin size for x-axis (dimension 0)
        x_binsize = self.s_binsize_array[0]
        self.x_edges = np.arange(x_min, x_max + x_binsize, x_binsize)

        # Use new base class method if requested
        if use_base_class:
            bin_edges = [self.x_edges]
            tc, occupancy, ratemap = self.create_nd_tuning_curve(
                st_data=st_run,
                pos_data=pos_run,
                bin_edges=bin_edges,
                min_duration=self.min_duration,
                minbgrate=self.minbgrate,
                tuning_curve_sigma=self.tuning_curve_sigma_array,
                smooth_mode=self.smooth_mode,
            )
            return tc, st_run

        # Original implementation
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

    def compute_occupancy_1d(self, pos_run: object) -> np.ndarray:
        """Computes the occupancy for 1D position data.

        Parameters
        ----------
        pos_run : object
            Restricted position data for running.

        Returns
        -------
        np.ndarray
            Occupancy values per bin.
        """
        occupancy, _ = np.histogram(pos_run.data[0, :], bins=self.x_edges)
        return occupancy / pos_run.fs

    def compute_ratemap_1d(
        self, st_run: object, pos_run: object, occupancy: np.ndarray
    ) -> np.ndarray:
        """Computes the ratemap for 1D data.

        Parameters
        ----------
        st_run : object
            Spike train data restricted to running epochs.
        pos_run : object
            Position data restricted to running epochs.
        occupancy : np.ndarray
            Occupancy values per bin.

        Returns
        -------
        np.ndarray
            Ratemap values for the given spike and position data.
        """
        # initialize ratemap
        ratemap = np.zeros((st_run.data.shape[0], occupancy.shape[0]))

        if st_run.isempty:
            return ratemap

        mask = ~np.isnan(pos_run.data).any(axis=0)
        x_pos, ts = (
            pos_run.data[0, mask],
            pos_run.abscissa_vals[mask],
        )
        # if data to map is spike train (point process)
        if isinstance(st_run, nel.core._eventarray.SpikeTrainArray):
            for i in range(st_run.data.shape[0]):
                # get spike counts in each bin
                (
                    ratemap[i, : len(self.x_edges)],
                    _,
                ) = np.histogram(
                    np.interp(st_run.data[i], ts, x_pos),
                    bins=self.x_edges,
                )

        # if data to map is analog signal (continuous)
        elif isinstance(st_run, nel.core._analogsignalarray.AnalogSignalArray):
            # get x location for every bin center
            x = np.interp(st_run.abscissa_vals, ts, x_pos)
            # get indices location within bin edges
            ext_bin_idx = np.squeeze(np.digitize(x, self.x_edges, right=True))
            # iterate over each time step and add data values to ratemap
            for tt, bidx in enumerate(ext_bin_idx):
                ratemap[:, bidx - 1] += st_run.data[:, tt]
            # divide by sampling rate
            ratemap = ratemap * st_run.fs

        # divide by occupancy
        np.divide(ratemap, occupancy, where=occupancy != 0, out=ratemap)

        # remove nans and infs
        bad_idx = np.isnan(ratemap) | np.isinf(ratemap)
        ratemap[bad_idx] = 0

        return ratemap

    def map_2d(
        self, pos: Optional[object] = None, use_base_class: bool = False
    ) -> tuple:
        """Maps 2D data for the spatial tuning curve.

        Parameters
        ----------
        pos : Optional[object]
            Position data for shuffling.
        use_base_class : bool, optional
            Whether to use the new NDimensionalBinner base class functionality.
            Default is False to maintain backward compatibility.

        Returns
        -------
        tuple
            A tuple containing the tuning curve and restricted spike train.
        """
        # restrict spike trains to those epochs during which the animal was running
        st_run = self.st[self.run_epochs]

        # log warning if st_run is empty following restriction
        if st_run.isempty:
            logging.warning(
                "No spike trains during running epochs"
            )  # This will log it but not raise a warning
            warnings.warn("No spike trains during running epochs", UserWarning)

        # take pos as input for case of shuffling
        if pos is not None:
            pos_run = pos[self.run_epochs]
        else:
            pos_run = self.pos[self.run_epochs]

        # Use dimension-specific min/max values
        ext_xmin, ext_xmax = self.dim_minmax_array[0]
        ext_ymin, ext_ymax = self.dim_minmax_array[1]

        # create bin edges
        # Use dimension-specific bin sizes
        x_binsize = self.s_binsize_array[0]
        y_binsize = self.s_binsize_array[1]
        self.x_edges = np.arange(ext_xmin, ext_xmax + x_binsize, x_binsize)
        self.y_edges = np.arange(ext_ymin, ext_ymax + y_binsize, y_binsize)

        # Use new base class method if requested
        if use_base_class:
            bin_edges = [self.x_edges, self.y_edges]
            tc, occupancy, ratemap = self.create_nd_tuning_curve(
                st_data=st_run,
                pos_data=pos_run,
                bin_edges=bin_edges,
                min_duration=self.min_duration,
                minbgrate=self.minbgrate,
                tuning_curve_sigma=self.tuning_curve_sigma_array,
                smooth_mode=self.smooth_mode,
            )
            return tc, st_run

        # Original implementation
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
        """Computes the occupancy for 2D position data.

        Parameters
        ----------
        pos_run : object
            Restricted position data for running.

        Returns
        -------
        np.ndarray
            Occupancy values per bin.
        """
        occupancy, _, _ = np.histogram2d(
            pos_run.data[0, :], pos_run.data[1, :], bins=(self.x_edges, self.y_edges)
        )
        return occupancy / pos_run.fs

    def compute_ratemap_2d(
        self, st_run: object, pos_run: object, occupancy: np.ndarray
    ) -> np.ndarray:
        """Computes the ratemap for 2D data.

        Parameters
        ----------
        st_run : object
            Spike train data restricted to running epochs.
        pos_run : object
            Position data restricted to running epochs.
        occupancy : np.ndarray
            Occupancy values per bin.

        Returns
        -------
        np.ndarray
            Ratemap values for the given spike and position data.
        """
        ratemap = np.zeros(
            (st_run.data.shape[0], occupancy.shape[0], occupancy.shape[1])
        )
        if st_run.isempty:
            return ratemap

        # remove nans from position data for interpolation
        mask = ~np.isnan(pos_run.data).any(axis=0)
        x_pos, y_pos, ts = (
            pos_run.data[0, mask],
            pos_run.data[1, mask],
            pos_run.abscissa_vals[mask],
        )

        if isinstance(st_run, nel.core._eventarray.SpikeTrainArray):
            for i in range(st_run.data.shape[0]):
                ratemap[i, : len(self.x_edges), : len(self.y_edges)], _, _ = (
                    np.histogram2d(
                        np.interp(st_run.data[i], ts, x_pos),
                        np.interp(st_run.data[i], ts, y_pos),
                        bins=(self.x_edges, self.y_edges),
                    )
                )

        elif isinstance(st_run, nel.core._analogsignalarray.AnalogSignalArray):
            x = np.interp(st_run.abscissa_vals, ts, x_pos)
            y = np.interp(st_run.abscissa_vals, ts, y_pos)
            ext_bin_idx_x = np.squeeze(np.digitize(x, self.x_edges, right=True))
            ext_bin_idx_y = np.squeeze(np.digitize(y, self.y_edges, right=True))
            for tt, (bidxx, bidxy) in enumerate(zip(ext_bin_idx_x, ext_bin_idx_y)):
                ratemap[:, bidxx - 1, bidxy - 1] += st_run.data[:, tt]
            ratemap = ratemap * st_run.fs

        np.divide(ratemap, occupancy, where=occupancy != 0, out=ratemap)

        bad_idx = np.isnan(ratemap) | np.isinf(ratemap)
        ratemap[bad_idx] = 0

        return ratemap

    def map_nd(self, pos: Optional[object] = None) -> tuple:
        """Maps N-dimensional data for the spatial tuning curve using the base class.

        Parameters
        ----------
        pos : Optional[object]
            Position data for shuffling.

        Returns
        -------
        tuple
            A tuple containing the tuning curve and restricted spike train.
        """
        # restrict spike trains to those epochs during which the animal was running
        st_run = self.st[self.run_epochs]

        # log warning if st_run is empty following restriction
        if st_run.isempty:
            logging.warning(
                "No spike trains during running epochs"
            )  # This will log it but not raise a warning
            warnings.warn("No spike trains during running epochs", UserWarning)

        # take pos as input for case of shuffling
        if pos is not None:
            pos_run = pos[self.run_epochs]
        else:
            pos_run = self.pos[self.run_epochs]

        # Create bin edges for each dimension
        bin_edges = []
        for dim_idx in range(self.dim):
            # Use dimension-specific min/max from dim_minmax_array
            dim_min, dim_max = self.dim_minmax_array[dim_idx]

            # Use dimension-specific bin size
            dim_binsize = self.s_binsize_array[dim_idx]
            edges = np.arange(dim_min, dim_max + dim_binsize, dim_binsize)
            bin_edges.append(edges)

        # Store bin edges for compatibility with existing code
        self.x_edges = bin_edges[0]
        if len(bin_edges) > 1:
            self.y_edges = bin_edges[1]

        # Use the base class method to create the tuning curve
        tc, occupancy, ratemap = self.create_nd_tuning_curve(
            st_data=st_run,
            pos_data=pos_run,
            bin_edges=bin_edges,
            min_duration=self.min_duration,
            minbgrate=self.minbgrate,
            tuning_curve_sigma=self.tuning_curve_sigma_array,
            smooth_mode=self.smooth_mode,
        )

        return tc, st_run

    def shuffle_spatial_information(self) -> np.ndarray:
        """Shuffle spatial information and compute p-values for observed vs. null.

        This method creates shuffled coordinates of the position data and computes
        spatial information for each shuffle. The p-values for the observed
        spatial information against the null distribution are calculated.

        Returns
        -------
        np.ndarray
            P-values for the spatial information.
        """

        def create_shuffled_coordinates(
            X: np.ndarray, n_shuff: int = 500
        ) -> List[np.ndarray]:
            """Create shuffled coordinates by rolling the original coordinates.

            Parameters
            ----------
            X : np.ndarray
                Original position data.
            n_shuff : int, optional
                Number of shuffles to create (default is 500).

            Returns
            -------
            List[np.ndarray]
                List of shuffled coordinates.
            """
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

        def get_spatial_infos(pos_shuff: np.ndarray, ts: np.ndarray, dim: int) -> float:
            """Get spatial information for shuffled position data.

            Parameters
            ----------
            pos_shuff : np.ndarray
                Shuffled position data.
            ts : np.ndarray
                Timestamps corresponding to the shuffled data.
            dim : int
                Dimension of the spatial data (1 or 2).

            Returns
            -------
            float
                Spatial information calculated from the tuning curve.
            """
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
        _, self.spatial_information_pvalues, self.spatial_information_zscore = (
            get_significant_events(
                self.tc.spatial_information(), np.array(shuffle_spatial_info)
            )
        )

        return self.spatial_information_pvalues

    def find_fields(self) -> None:
        """Find place fields in the spatial maps.

        This method detects place fields from the spatial maps and calculates
        their properties, including width, peak firing rate, and a mask for
        each detected field.
        """
        from skimage import measure

        field_width = []
        peak_rate = []
        mask = []

        if self.place_field_max_size is None and self.dim == 1:
            # For 1D, use the bin size for dimension 0
            self.place_field_max_size = self.tc.n_bins * self.s_binsize_array[0]
        elif self.place_field_max_size is None and self.dim == 2:
            # For 2D, use the average of both dimensions or the maximum
            avg_binsize = np.mean(self.s_binsize_array[:2])
            self.place_field_max_size = self.tc.n_bins * avg_binsize

        if self.dim == 1:
            # Use bin size for dimension 0 (x-axis)
            x_binsize = self.s_binsize_array[0]
            for ratemap_ in self.tc.ratemap:
                map_fields = fields.map_stats2(
                    ratemap_,
                    threshold=self.place_field_thres,
                    min_size=self.place_field_min_size / x_binsize,
                    max_size=self.place_field_max_size / x_binsize,
                    min_peak=self.place_field_min_peak,
                    sigma=self.place_field_sigma,
                )
                if len(map_fields["sizes"]) == 0:
                    field_width.append(np.nan)
                    peak_rate.append(np.nan)
                    mask.append(map_fields["fields"])
                else:
                    field_width.append(
                        np.array(map_fields["sizes"]).max() * len(ratemap_) * x_binsize
                    )
                    peak_rate.append(np.array(map_fields["peaks"]).max())
                    mask.append(map_fields["fields"])

        if self.dim == 2:
            # Use average of x and y bin sizes for 2D field calculations
            avg_binsize = np.mean(self.s_binsize_array[:2])
            for ratemap_ in self.tc.ratemap:
                peaks = fields.compute_2d_place_fields(
                    ratemap_,
                    min_firing_rate=self.place_field_min_peak,
                    thresh=self.place_field_thres,
                    min_size=(self.place_field_min_size / avg_binsize),
                    max_size=(self.place_field_max_size / avg_binsize),
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
                    field_width.append(np.max(pdist(bc[0], "euclidean")) * avg_binsize)
                    # field_ids = np.unique(peaks)
                    peak_rate.append(ratemap_[peaks == 1].max())
                    mask.append(peaks)

        self.tc.field_width = np.array(field_width)
        self.tc.field_peak_rate = np.array(peak_rate)
        self.tc.field_mask = np.array(mask)
        self.tc.n_fields = np.array(
            [len(np.unique(mask_)) - 1 for mask_ in self.tc.field_mask]
        )

    def save_mat_file(self, basepath: str, UID: Optional[Any] = None) -> None:
        """Save firing rate map data to a .mat file in MATLAB format.

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

        The file will be saved to a .mat file with the name `basepath.ratemap.firingRateMap.mat`,
        where `basepath` is the base path of the data.

        Parameters
        ----------
        basepath : str
            The base path for saving the .mat file.
        UID : Optional[Any], optional
            A unique identifier for the data (default is None).

        Returns
        -------
        None
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
            firingRateMap["spatial_information_pvalues"] = (
                self.spatial_information_pvalues.tolist()
            )
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
