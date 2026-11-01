import nelpy as nel
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter
from scipy.stats import binned_statistic_dd

from neuro_py.tuning.maps import SpatialMap


def _make_map(
    dimensions: int,
    limits: tuple[float, float] | None,
    binsize: float,
    continuous: bool = False,
    dimension_bounds: bool = False,
    sigma: float = 0,
) -> SpatialMap:
    timestamps = np.arange(80) / 40
    low, high = (0.1, 4.4) if limits is None else limits
    coordinates = np.linspace(low, high, len(timestamps))
    pos = nel.AnalogSignalArray(
        np.tile(coordinates, (dimensions, 1)), timestamps=timestamps, fs=40
    )
    if continuous:
        data = nel.AnalogSignalArray(
            np.vstack((2 + coordinates**2, 5 - coordinates)),
            timestamps=timestamps,
            fs=40,
        )
    else:
        data = nel.SpikeTrainArray(
            [np.array([0.15, 0.5, 1, 1.5]), np.array([0.3, 0.9, 1.7])], fs=20000
        )
    kwargs = {}
    if limits is not None:
        if dimension_bounds:
            kwargs["dim_minmax"] = [list(limits)] * dimensions
        else:
            kwargs["x_minmax"] = list(limits)
            if dimensions == 2:
                kwargs["y_minmax"] = list(limits)
    return SpatialMap(
        pos,
        data,
        speed_thres=0,
        s_binsize=binsize,
        tuning_curve_sigma=sigma,
        min_duration=0,
        **kwargs,
    )


@pytest.mark.parametrize("dimensions", [1, 2])
@pytest.mark.parametrize("continuous", [False, True])
@pytest.mark.parametrize("dimension_bounds", [False, True])
@pytest.mark.parametrize(
    "limits,binsize,expected_edges",
    [
        ((0, 4.5), 2.5, [0, 2.5, 5]),
        ((0, 5), 2.5, [0, 2.5, 5]),
        ((2, 6.5), 2.5, [2, 4.5, 7]),
        ((-4, 0.5), 2.5, [-4, -1.5, 1]),
        ((0, 1), 2.5, [0, 2.5]),
        ((0, 0.45), 0.25, [0, 0.25, 0.5]),
        ((0.1, 0.4), 0.1, [0.1, 0.2, 0.3, 0.4]),
        (None, 3, [0, 3, 6]),
    ],
)
def test_coordinates_and_values_match_histogram_bins(
    dimensions: int,
    continuous: bool,
    dimension_bounds: bool,
    limits: tuple[float, float] | None,
    binsize: float,
    expected_edges: list[float],
) -> None:
    spatial = _make_map(dimensions, limits, binsize, continuous, dimension_bounds)
    edges = [spatial.x_edges] + ([spatial.y_edges] if dimensions == 2 else [])
    reported_edges = (
        [spatial.tc.bins] if dimensions == 1 else [spatial.xbins, spatial.ybins]
    )
    reported_centers = (
        [spatial.bin_centers]
        if dimensions == 1
        else [spatial.xbin_centers, spatial.ybin_centers]
    )
    expected_centers = (np.asarray(expected_edges[:-1]) + expected_edges[1:]) / 2
    for counted, reported, centers in zip(edges, reported_edges, reported_centers):
        np.testing.assert_allclose(counted, expected_edges, atol=1e-12)
        np.testing.assert_allclose(reported, expected_edges, atol=1e-12)
        np.testing.assert_allclose(centers, expected_centers, atol=1e-12)

    # Independent histograms check that correcting coordinates preserves counts,
    # occupancy, unsmoothed rates and continuous-signal bin means.
    pos = spatial.pos[spatial.run_epochs]
    occupancy = np.histogramdd(pos.data.T, bins=edges)[0] / pos.fs
    np.testing.assert_allclose(spatial.occupancy, occupancy)
    for unit, values in enumerate(spatial.st_run.data):
        sample_times = spatial.st_run.abscissa_vals if continuous else values
        positions = np.column_stack(
            [
                np.interp(sample_times, pos.abscissa_vals, coordinate)
                for coordinate in pos.data
            ]
        )
        if continuous:
            expected = binned_statistic_dd(
                positions, values, statistic="mean", bins=edges
            )[0]
            expected = np.nan_to_num(expected)
            # The configured background floor also applies to continuous maps.
            expected = np.maximum(expected, spatial.minbgrate)
        else:
            counts = np.histogramdd(positions, bins=edges)[0]
            expected = np.divide(
                counts, occupancy, out=np.zeros_like(counts), where=occupancy > 0
            )
        np.testing.assert_allclose(spatial.ratemap[unit], expected)


@pytest.mark.parametrize("dimensions", [1, 2])
@pytest.mark.parametrize("continuous", [False, True])
def test_smoothing_uses_histogram_bin_widths(dimensions: int, continuous: bool) -> None:
    raw = _make_map(dimensions, (0, 4.5), 2.5, continuous)
    smoothed = _make_map(dimensions, (0, 4.5), 2.5, continuous, sigma=1.25)
    expected = gaussian_filter(
        raw.ratemap, sigma=(0,) + (1.25 / 2.5,) * dimensions, mode="reflect"
    )
    np.testing.assert_allclose(smoothed.ratemap, expected)
    np.testing.assert_array_equal(smoothed.occupancy, raw.occupancy)


def test_anisotropic_bins_keep_independent_axis_coordinates() -> None:
    inputs = _make_map(2, (0, 4.5), 2.5)
    spatial = SpatialMap(
        inputs.pos,
        inputs.st,
        s_binsize=[2.5, 4],
        dim_minmax=[[0, 4.5], [-3, 6.3]],
        speed_thres=0,
        tuning_curve_sigma=0,
        min_duration=0,
    )
    tc = spatial.tc
    np.testing.assert_allclose(tc.xbins, [0, 2.5, 5])
    np.testing.assert_allclose(tc.ybins, [-3, 1, 5, 9])
    np.testing.assert_allclose(tc.xbin_centers, [1.25, 3.75])
    np.testing.assert_allclose(tc.ybin_centers, [-1, 3, 7])
    expected = gaussian_filter(
        tc.ratemap, sigma=(0, 1.25 / 2.5, 1.25 / 4), mode="reflect"
    )
    np.testing.assert_allclose(tc.smooth(sigma=1.25).ratemap, expected)


@pytest.mark.parametrize("dimensions", [1, 2])
def test_default_and_generic_binners_agree_on_coordinates(dimensions: int) -> None:
    spatial = _make_map(dimensions, (0, 4.5), 2.5)
    generic, _ = (
        spatial.map_1d(use_base_class=True)
        if dimensions == 1
        else spatial.map_2d(use_base_class=True)
    )
    if dimensions == 1:
        np.testing.assert_allclose(generic.bins, spatial.tc.bins)
    else:
        np.testing.assert_allclose(generic.xbins, spatial.xbins)
        np.testing.assert_allclose(generic.ybins, spatial.ybins)
    np.testing.assert_allclose(generic.ratemap, spatial.ratemap)
    np.testing.assert_array_equal(generic.occupancy, spatial.occupancy)


def test_nd_binner_keeps_histogram_coordinates() -> None:
    spatial = _make_map(3, (0, 4.5), 2.5, dimension_bounds=True)
    for edges in spatial.tc.bins:
        np.testing.assert_allclose(edges, [0, 2.5, 5])
