import time
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import ndimage
from scipy.spatial.distance import pdist
from skimage import measure

from neuro_py.tuning import fields
from neuro_py.tuning.maps import SpatialMap


def _gaussian_blob(
    shape: tuple[int, int],
    center: tuple[float, float],
    amplitude: float,
    sigma: float,
) -> np.ndarray:
    x_idx, y_idx = np.indices(shape)
    return amplitude * np.exp(
        -(((x_idx - center[0]) ** 2) + ((y_idx - center[1]) ** 2)) / (2 * sigma**2)
    )


def _make_spatial_map_stub(ratemaps: np.ndarray) -> SpatialMap:
    spatial_map = SpatialMap.__new__(SpatialMap)
    spatial_map.dim = 2
    spatial_map.s_binsize_array = np.array([1.0, 1.0])
    spatial_map.place_field_thres = 0.2
    spatial_map.place_field_min_size = 5
    spatial_map.place_field_max_size = 10_000
    spatial_map.place_field_min_peak = 1
    spatial_map.place_field_sigma = None
    spatial_map.tc = SimpleNamespace(ratemap=ratemaps, n_bins=ratemaps.shape[-1])
    return spatial_map


def _legacy_compute_2d_place_fields(
    firing_rate: np.ndarray,
    min_firing_rate: float = 1,
    thresh: float = 0.2,
    min_size: int = 100,
    max_size: int = 200,
    sigma: float | None = None,
    filter_size: int = 3,
) -> np.ndarray:
    firing_rate_orig = firing_rate.copy()

    if sigma is not None:
        firing_rate = ndimage.gaussian_filter(firing_rate, sigma)

    local_maxima_inds = firing_rate == ndimage.maximum_filter(
        firing_rate, size=filter_size
    )
    receptive_fields = np.zeros(firing_rate.shape, dtype=int)
    n_receptive_fields = 0
    firing_rate = firing_rate.copy()
    for local_max in np.flipud(np.sort(firing_rate[local_maxima_inds])):
        labeled_image, num_labels = ndimage.label(
            firing_rate > max(local_max * thresh, min_firing_rate)
        )

        if not num_labels:
            continue
        for i in range(1, num_labels + 1):
            image_label = labeled_image == i
            if local_max in firing_rate[image_label]:
                break
            if np.sum(image_label) >= min_size:
                n_receptive_fields += 1
                receptive_fields[image_label] = n_receptive_fields
                firing_rate[image_label] = 0

    receptive_fields = fields.remove_fields_by_area(
        receptive_fields, int(min_size), maximum_field_area=max_size
    )
    if n_receptive_fields > 0:
        receptive_fields = fields.sort_fields_by_rate(
            firing_rate_orig, receptive_fields, func=np.max
        )
    return receptive_fields


def test_compute_2d_place_fields_single_blob():
    ratemap = _gaussian_blob((40, 40), center=(20, 20), amplitude=8.0, sigma=4.0)

    labels = fields.compute_2d_place_fields(
        ratemap,
        min_firing_rate=1.0,
        thresh=0.2,
        min_size=5,
        max_size=1000,
        sigma=None,
    )
    legacy_labels = _legacy_compute_2d_place_fields(
        ratemap,
        min_firing_rate=1.0,
        thresh=0.2,
        min_size=5,
        max_size=1000,
        sigma=None,
    )

    assert labels.shape == ratemap.shape
    assert np.array_equal(labels, legacy_labels)


def test_compute_2d_place_fields_two_blobs_sorted_by_peak():
    ratemap = _gaussian_blob((60, 60), center=(18, 18), amplitude=9.0, sigma=4.0)
    ratemap += _gaussian_blob((60, 60), center=(42, 42), amplitude=5.0, sigma=4.5)

    labels = fields.compute_2d_place_fields(
        ratemap,
        min_firing_rate=1.0,
        thresh=0.2,
        min_size=5,
        max_size=2000,
        sigma=None,
    )
    legacy_labels = _legacy_compute_2d_place_fields(
        ratemap,
        min_firing_rate=1.0,
        thresh=0.2,
        min_size=5,
        max_size=2000,
        sigma=None,
    )

    assert np.array_equal(labels, legacy_labels)


def test_compute_2d_place_fields_merges_connected_hotspots():
    ratemap = _gaussian_blob((70, 70), center=(28, 30), amplitude=8.0, sigma=5.0)
    ratemap += _gaussian_blob((70, 70), center=(40, 38), amplitude=7.0, sigma=5.0)

    labels = fields.compute_2d_place_fields(
        ratemap,
        min_firing_rate=1.0,
        thresh=0.2,
        min_size=5,
        max_size=4000,
        sigma=1.5,
    )
    legacy_labels = _legacy_compute_2d_place_fields(
        ratemap,
        min_firing_rate=1.0,
        thresh=0.2,
        min_size=5,
        max_size=4000,
        sigma=1.5,
    )

    assert np.array_equal(labels, legacy_labels)


def test_compute_2d_place_fields_matches_legacy_behavior():
    ratemap = _gaussian_blob((70, 70), center=(22, 24), amplitude=8.0, sigma=4.5)
    ratemap += _gaussian_blob((70, 70), center=(40, 45), amplitude=7.5, sigma=5.5)
    ratemap += _gaussian_blob((70, 70), center=(46, 52), amplitude=4.5, sigma=2.5)
    ratemap += 0.15

    labels = fields.compute_2d_place_fields(
        ratemap,
        min_firing_rate=1.0,
        thresh=0.2,
        min_size=5,
        max_size=4000,
        sigma=1.5,
    )
    legacy_labels = _legacy_compute_2d_place_fields(
        ratemap,
        min_firing_rate=1.0,
        thresh=0.2,
        min_size=5,
        max_size=4000,
        sigma=1.5,
    )

    assert np.array_equal(labels, legacy_labels)


def test_compute_2d_place_fields_respects_size_filters():
    ratemap = np.zeros((50, 50), dtype=float)
    ratemap[10:12, 10:12] = 6.0
    ratemap[25:35, 25:35] = 6.0

    labels = fields.compute_2d_place_fields(
        ratemap,
        min_firing_rate=1.0,
        thresh=0.2,
        min_size=10,
        max_size=60,
        sigma=None,
    )
    legacy_labels = _legacy_compute_2d_place_fields(
        ratemap,
        min_firing_rate=1.0,
        thresh=0.2,
        min_size=10,
        max_size=60,
        sigma=None,
    )

    assert np.array_equal(labels, legacy_labels)


def test_compute_2d_place_fields_smoothing_preserves_label_convention():
    ratemap = _gaussian_blob((50, 50), center=(25, 25), amplitude=10.0, sigma=3.0)
    ratemap += 0.2 * _gaussian_blob((50, 50), center=(15, 35), amplitude=3.0, sigma=2.0)

    labels_no_smooth = fields.compute_2d_place_fields(
        ratemap,
        min_firing_rate=1.0,
        thresh=0.2,
        min_size=5,
        max_size=1000,
        sigma=None,
    )
    labels_smooth = fields.compute_2d_place_fields(
        ratemap,
        min_firing_rate=1.0,
        thresh=0.2,
        min_size=5,
        max_size=1000,
        sigma=1.5,
    )
    legacy_no_smooth = _legacy_compute_2d_place_fields(
        ratemap,
        min_firing_rate=1.0,
        thresh=0.2,
        min_size=5,
        max_size=1000,
        sigma=None,
    )
    legacy_smooth = _legacy_compute_2d_place_fields(
        ratemap,
        min_firing_rate=1.0,
        thresh=0.2,
        min_size=5,
        max_size=1000,
        sigma=1.5,
    )

    assert labels_no_smooth.shape == ratemap.shape
    assert labels_smooth.shape == ratemap.shape
    assert np.array_equal(labels_no_smooth, legacy_no_smooth)
    assert np.array_equal(labels_smooth, legacy_smooth)


def test_spatial_map_find_fields_2d_empty_outputs():
    spatial_map = _make_spatial_map_stub(np.array([np.zeros((50, 50), dtype=float)]))

    SpatialMap.find_fields(spatial_map)

    assert spatial_map.tc.field_mask.shape == (1, 50, 50)
    assert spatial_map.tc.n_fields.tolist() == [0]
    assert np.isnan(spatial_map.tc.field_peak_rate[0])
    assert np.isnan(spatial_map.tc.field_width[0])


def test_spatial_map_find_fields_2d_uses_primary_field_boundary(
    monkeypatch: pytest.MonkeyPatch,
):
    ratemap = np.zeros((30, 30), dtype=float)
    ratemap[3:9, 3:9] = 5.0
    ratemap[12:28, 16:28] = 2.0
    peaks = np.zeros((30, 30), dtype=int)
    peaks[3:9, 3:9] = 1
    peaks[12:28, 16:28] = 2

    spatial_map = _make_spatial_map_stub(np.array([ratemap]))

    monkeypatch.setattr(
        fields, "compute_2d_place_fields", lambda *args, **kwargs: peaks
    )

    SpatialMap.find_fields(spatial_map)

    expected_contours = measure.find_contours(
        (peaks == 1).astype(float),
        0.5,
        fully_connected="low",
        positive_orientation="low",
    )
    expected_width = np.max(
        pdist(max(expected_contours, key=lambda contour: contour.shape[0]), "euclidean")
    )

    assert spatial_map.tc.field_peak_rate.tolist() == [5.0]
    assert spatial_map.tc.n_fields.tolist() == [2]
    assert spatial_map.tc.field_width[0] == pytest.approx(expected_width)


def test_compute_2d_place_fields_large_noisy_map_smoke():
    rng = np.random.default_rng(0)
    ratemap = rng.random((180, 240)) * 5.0
    ratemap += _gaussian_blob((180, 240), center=(80, 120), amplitude=6.0, sigma=10.0)

    start = time.perf_counter()
    labels = fields.compute_2d_place_fields(
        ratemap,
        min_firing_rate=3.0,
        thresh=0.2,
        min_size=10,
        max_size=20_000,
        sigma=2,
    )
    elapsed = time.perf_counter() - start

    assert labels.shape == ratemap.shape
    assert elapsed < 30.0
