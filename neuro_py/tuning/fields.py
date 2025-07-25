from math import sqrt
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d, label
from scipy.ndimage.filters import gaussian_filter, maximum_filter


def detect_firing_fields(
    image_gray: np.ndarray,
    max_sigma: int = 30,
    log_num_sigma: int = 10,
    log_thres: float = 0.1,
    dog_thres: float = 0.1,
    doh_thres: float = 0.01,
) -> None:
    """
    Detect firing fields in a grayscale image using different blob detection methods.

    Parameters
    ----------
    image_gray : np.ndarray
        Grayscale image to analyze.
    max_sigma : int, optional
        The maximum standard deviation for Gaussian filter.
    log_num_sigma : int, optional
        The number of sigma values for the Laplacian of Gaussian.
    log_thres : float, optional
        Threshold for Laplacian of Gaussian blobs.
    dog_thres : float, optional
        Threshold for Difference of Gaussian blobs.
    doh_thres : float, optional
        Threshold for Determinant of Hessian blobs.
    """
    from skimage.feature import blob_dog, blob_doh, blob_log

    plt.imshow(image_gray, origin="lower")

    blobs_log = blob_log(
        image_gray, max_sigma=max_sigma, num_sigma=log_num_sigma, threshold=log_thres
    )
    # Compute radii in the 3rd column.
    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

    blobs_dog = blob_dog(image_gray, max_sigma=max_sigma, threshold=dog_thres)
    blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)

    blobs_doh = blob_doh(image_gray, max_sigma=max_sigma, threshold=doh_thres)

    blobs_list = [blobs_log, blobs_dog, blobs_doh]
    colors = ["yellow", "lime", "red"]
    titles = [
        "Laplacian of Gaussian",
        "Difference of Gaussian",
        "Determinant of Hessian",
    ]
    sequence = zip(blobs_list, colors, titles)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
    ax = axes.ravel()

    for idx, (blobs, color, title) in enumerate(sequence):
        ax[idx].set_title(title)
        ax[idx].imshow(image_gray, interpolation="nearest", origin="lower")
        for blob in blobs:
            y, x, r = blob
            c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
            ax[idx].add_patch(c)
        ax[idx].set_axis_off()

    plt.tight_layout()


def find_peaks(image: np.ndarray) -> np.ndarray:
    """
    Find peaks sorted by distance from the center of the image.

    Parameters
    ----------
    image : np.ndarray
        The input image.

    Returns
    -------
    np.ndarray
        Coordinates for peaks in the image as [row, column].
    """
    image = image.copy()
    image[~np.isfinite(image)] = 0
    image_max = filters.maximum_filter(image, 3)
    is_maxima = image == image_max
    labels, num_objects = ndimage.label(is_maxima)
    indices = np.arange(1, num_objects + 1)
    peaks = ndimage.maximum_position(image, labels=labels, index=indices)
    peaks = np.array(peaks)
    center = (np.array(image.shape) - 1) / 2
    distances = np.linalg.norm(peaks - center, axis=1)
    peaks = peaks[distances.argsort()]
    return peaks


def sort_fields_by_rate(
    rate_map: np.ndarray,
    fields: np.ndarray,
    func: Optional[Callable[[np.ndarray], Any]] = None,
) -> np.ndarray:
    """
    Sort fields by the rate value of each field.

    Parameters
    ----------
    rate_map : np.ndarray
        The rate map.
    fields : np.ndarray
        The fields of the same shape as rate_map.
    func : Callable, optional
        Function returning value to sort after, default is np.max.

    Returns
    -------
    np.ndarray
        Sorted fields.
    """
    indx = np.sort(np.unique(fields.ravel()))
    func = func or np.max
    # Sort by largest peak
    rate_means = ndimage.labeled_comprehension(
        rate_map, fields, indx, func, np.float64, 0
    )
    sort = np.argsort(rate_means)[::-1]

    sorted_fields = np.zeros_like(fields)
    for indx_i, indx_ in enumerate(indx[sort]):
        if indx_ == 0:
            continue
        sorted_fields[fields == indx_] = np.max(sorted_fields) + 1

    # new rate map with fields > min_size, sorted
    # sorted_fields = np.zeros_like(fields)
    # for i in range(indx.max() + 1):
    #     sorted_fields[fields == sort[i] + 1] = i + 1

    return sorted_fields


def remove_fields_by_area(
    fields: np.ndarray,
    minimum_field_area: int,
    maximum_field_area: Optional[int] = None,
) -> np.ndarray:
    """
    Sets fields below minimum area to zero, measured as the number of bins in a field.

    Parameters
    ----------
    fields : np.ndarray
        The fields.
    minimum_field_area : int
        Minimum field area (number of bins in a field).
    maximum_field_area : Optional[int]
        Maximum field area (number of bins in a field). Default is None.

    Returns
    -------
    np.ndarray
        Fields with number of bins below minimum_field_area are set to zero.

    Raises
    ------
    ValueError
        If minimum_field_area is not an integer.
    """
    if not isinstance(minimum_field_area, (int, np.integer)):
        raise ValueError("'minimum_field_area' should be int")

    if maximum_field_area is None:
        maximum_field_area = len(fields.flatten())
    ## variant
    # fields_areas = scipy.ndimage.measurements.sum(
    #     np.ones_like(fields), fields, index=np.arange(fields.max() + 1))
    # fields_area = fields_areas[fields]
    # fields[fields_area < minimum_field_area] = 0

    labels, counts = np.unique(fields, return_counts=True)
    for lab, count in zip(labels, counts):
        if lab != 0:
            if (count < minimum_field_area) | (count > maximum_field_area):
                fields[fields == lab] = 0
    return fields


def separate_fields_by_laplace(
    rate_map: np.ndarray, threshold: float = 0, minimum_field_area: Optional[int] = None
) -> np.ndarray:
    """
    Separates fields using the Laplacian to identify fields separated by
    a negative second derivative.

    Parameters
    ----------
    rate_map : np.ndarray
        2D array representing firing rate in each bin.
    threshold : float
        Value of Laplacian to separate fields by relative to the minima.
        Should be on the interval 0 to 1, where 0 cuts off at 0 and
        1 cuts off at min(laplace(rate_map)). Default is 0.
    minimum_field_area : Optional[int]
        Minimum number of bins to consider it a field. Default is None (all fields are kept).

    Returns
    -------
    np.ndarray
        Labels with areas filled with the same value, corresponding to fields
        in rate_map. The fill values are in range(1, nFields + 1), sorted by size of the
        field (sum of all field values) with 0 elsewhere.
    """

    laplacian = ndimage.laplace(rate_map)

    laplacian[laplacian > threshold * np.min(laplacian)] = 0

    # Labels areas of the laplacian not connected by values > 0.
    fields, _ = ndimage.label(laplacian)
    fields = sort_fields_by_rate(rate_map, fields)
    if minimum_field_area is not None:
        fields = remove_fields_by_area(fields, minimum_field_area)
    return fields


def separate_fields_by_dilation(
    rate_map: np.ndarray,
    seed: float = 2.5,
    sigma: float = 2.5,
    minimum_field_area: Optional[int] = None,
) -> np.ndarray:
    """
    Separates fields by the Laplace of Gaussian (LoG)
    on the rate map subtracted by a reconstruction of the rate map using
    dilation.

    Parameters
    ----------
    rate_map : np.ndarray
        2D array representing firing rate in each bin.
    seed : float
        Magnitude of dilation.
    sigma : float
        Standard deviation of Gaussian to separate fields. Default is 2.5.
    minimum_field_area : Optional[int]
        Minimum number of bins to consider it a field. Default is None (all fields are kept).

    Returns
    -------
    np.ndarray
        Labels with areas filled with the same value, corresponding to fields
        in rate_map. The fill values are in range(1, nFields + 1), sorted by size of the
        field (sum of all field values) with 0 elsewhere.

    References
    ----------
    .. [1] https://scikit-image.org/docs/stable/auto_examples/color_exposure/plot_regional_maxima.html
    """
    from skimage.morphology import reconstruction

    rate_map_norm = (rate_map - rate_map.mean()) / rate_map.std()
    dilated = reconstruction(rate_map_norm - seed, rate_map_norm, method="dilation")
    rate_map_reconstructed = rate_map_norm - dilated

    laplacian = ndimage.gaussian_laplace(rate_map_reconstructed, sigma)
    laplacian[laplacian > 0] = 0
    fields, _ = ndimage.label(laplacian)
    fields = sort_fields_by_rate(rate_map, fields)
    if minimum_field_area is not None:
        fields = remove_fields_by_area(fields, minimum_field_area)
    return fields


def separate_fields_by_laplace_of_gaussian(
    rate_map: np.ndarray, sigma: float = 2, minimum_field_area: Optional[int] = None
) -> np.ndarray:
    """
    Separates fields using the Laplace of Gaussian (LoG) to identify fields
    separated by a negative second derivative. Works best if no smoothing is
    applied to the rate map, preferably with interpolated NaNs.

    Parameters
    ----------
    rate_map : np.ndarray
        2D array representing firing rate in each bin.
    sigma : float
        Standard deviation of Gaussian to separate fields. Default is 2.
    minimum_field_area : Optional[int]
        Minimum number of bins to consider it a field. Default is None (all fields are kept).

    Returns
    -------
    np.ndarray
        Labels with areas filled with the same value, corresponding to fields
        in rate_map. The fill values are in range(1, nFields + 1), sorted by size of the
        field (sum of all field values) with 0 elsewhere.
    """
    laplacian = ndimage.gaussian_laplace(rate_map, sigma)
    laplacian[laplacian > 0] = 0

    # Labels areas of the laplacian not connected by values > 0.
    fields, _ = ndimage.label(laplacian)

    fields = sort_fields_by_rate(rate_map, fields)
    if minimum_field_area is not None:
        fields = remove_fields_by_area(fields, minimum_field_area)
    return fields


def calculate_field_centers(
    rate_map: np.ndarray, labels: np.ndarray, center_method: str = "maxima"
) -> np.ndarray:
    """
    Finds center of fields at labels.

    Parameters
    ----------
    rate_map : np.ndarray
        2D array representing firing rate in each bin.
    labels : np.ndarray
        Labeled fields.
    center_method : str
        Method to calculate the center; either 'maxima' or 'center_of_mass'. Default is 'maxima'.

    Returns
    -------
    np.ndarray
        Coordinates of the center for each field.

    Raises
    ------
    ValueError
        If an invalid center_method is provided.
    """
    indices = np.arange(1, np.max(labels) + 1)
    if center_method == "maxima":
        bc = ndimage.maximum_position(rate_map, labels=labels, index=indices)
    elif center_method == "center_of_mass":
        bc = ndimage.center_of_mass(rate_map, labels=labels, index=indices)
    else:
        raise ValueError("invalid center_method flag '{}'".format(center_method))

    if not bc:
        # empty list
        return bc

    bc = np.array(bc)
    bc[:, [0, 1]] = bc[:, [1, 0]]  # y, x -> x, y
    return bc


def which_field(
    x: np.ndarray, y: np.ndarray, fields: np.ndarray, box_size: List[float]
) -> np.ndarray:
    """
    Returns which spatial field each (x, y) position is in.

    Parameters
    ----------
    x : np.ndarray
        X-coordinates.
    y : np.ndarray
        Y-coordinates, must have the same length as x.
    fields : np.ndarray
        Labeled fields, where each field is defined by an area separated by
        zeros. The fields are labeled with indices from [1:].
    box_size : List[float]
        Extents of the arena.

    Returns
    -------
    np.ndarray
        Array-like x and y with fields-labeled indices.

    Raises
    ------
    ValueError
        If x and y do not have the same length.
    """
    if len(x) != len(y):
        raise ValueError("x and y must have same length")

    sx, sy = fields.shape
    # bin sizes
    dx = box_size[0] / sx
    dy = box_size[1] / sy
    x_bins = dx + np.arange(0, box_size[0] + dx, dx)
    y_bins = dy + np.arange(0, box_size[1] + dx, dy)
    # x_bins = np.arange(0, box_size[0] + dx, dx)
    # y_bins = np.arange(0, box_size[1] + dx, dy)
    ix = np.digitize(x, x_bins)
    iy = np.digitize(y, y_bins)

    # fix for boundaries:
    ix[ix == sx] = sx - 1
    iy[iy == sy] = sy - 1
    return np.array(fields[ix, iy])


def compute_crossings(field_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute indices at which a field is entered or exited.

    Parameters
    ----------
    field_indices : np.ndarray
        1D array, typically obtained with in_field.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Indices at which fields are entered and exited.
    """
    # make sure to start and end outside fields
    field_indices = np.concatenate(([0], field_indices.astype(bool).astype(int), [0]))
    (enter,) = np.where(np.diff(field_indices) == 1)
    (exit,) = np.where(np.diff(field_indices) == -1)
    assert len(enter) == len(exit), (len(enter), len(exit))
    return enter, exit


def distance_to_edge_function(
    x_c: float,
    y_c: float,
    field: np.ndarray,
    box_size: Tuple[float, float],
    interpolation: str = "linear",
) -> Callable[[float], float]:
    """
    Returns a function which, for a given angle, returns the distance to
    the edge of the field from the center.

    Parameters
    ----------
    x_c : float
        X-coordinate of the center.
    y_c : float
        Y-coordinate of the center.
    field : np.ndarray
        2D array with ones at field bins and zeros elsewhere.
    box_size : Tuple[float, float]
        Size of the box (arena).
    interpolation : str, optional
        Type of interpolation to use. Default is "linear".

    Returns
    -------
    Callable[[float], float]
        A function that takes an angle and returns the distance to the edge of the field.
    """
    from skimage import measure

    contours = measure.find_contours(field, 0.8)

    box_dim = np.array(box_size)
    edge_x, edge_y = (contours[0] * box_dim / (np.array(field.shape) - (1, 1))).T

    # # angle between 0 and 2\pi
    angles = np.arctan2((edge_y - y_c), (edge_x - x_c)) % (2 * np.pi)
    a_sort = np.argsort(angles)
    angles = angles[a_sort]
    edge_x = edge_x[a_sort]
    edge_y = edge_y[a_sort]

    # # Fill in edge values for the interpolation
    pad_a = np.pad(angles, 2, mode="linear_ramp", end_values=(0, 2 * np.pi))
    ev_x = (edge_x[0] + edge_x[-1]) / 2
    pad_x = np.pad(edge_x, 2, mode="linear_ramp", end_values=ev_x)
    ev_y = (edge_y[0] + edge_y[-1]) / 2
    pad_y = np.pad(edge_y, 2, mode="linear_ramp", end_values=ev_y)

    if interpolation == "cubic":
        mask = np.where(np.diff(pad_a) == 0)
        pad_a = np.delete(pad_a, mask)
        pad_x = np.delete(pad_x, mask)
        pad_y = np.delete(pad_y, mask)

    x_func = interp1d(pad_a, pad_x, kind=interpolation)
    y_func = interp1d(pad_a, pad_y, kind=interpolation)

    def dist_func(angle: float) -> float:
        """
        Computes the distance from the center to the edge of the field at a given angle.

        Parameters
        ----------
        angle : float
            Angle in radians.

        Returns
        -------
        float
            Distance to the edge of the field from the center.
        """
        x = x_func(angle)
        y = y_func(angle)
        dist = np.sqrt((x - x_c) ** 2 + (y - y_c) ** 2)
        return dist

    return dist_func


def map_pass_to_unit_circle(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    x_c: float,
    y_c: float,
    field: Optional[np.ndarray] = None,
    box_size: Optional[Tuple[float, float]] = None,
    dist_func: Optional[Callable[[float], float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Uses three vectors {v, p, q} to map the passes to the unit circle. v
    is the average velocity vector of the pass, p is the vector from the
    position (x, y) to the center of the field, and q is the vector from the
    center to the edge through (x, y).

    Parameters
    ----------
    x : np.ndarray
        X-coordinates.
    y : np.ndarray
        Y-coordinates.
    t : np.ndarray
        Time data.
    x_c : float
        X-coordinate of the center of the field.
    y_c : float
        Y-coordinate of the center of the field.
    field : Optional[np.ndarray], optional
        2D array indicating the location of the field.
    box_size : Optional[Tuple[float, float]], optional
        Size of the box (arena).
    dist_func : Optional[Callable[[float], float]], optional
        Function that computes distance to bump edge from center. Default is
        distance_to_edge_function with linear interpolation.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        r : Array of distances to origin on unit circle.
        theta : Array of angles to axis defined by mean velocity vector.
        pdcd : Array of distances to peak projected onto the current direction.
        pdmd : Array of distances to peak projected onto the mean direction.

    Raises
    ------
    AssertionError
        If neither dist_func nor both field and box_size are provided.

    References:
    -----------
    .. [1] Jeewajee A, Barry C, Douchamps V, Manson D, Lever C, Burgess N. Theta
           phase precession of grid and place cell firing in open environments.
           Philos Trans R Soc Lond B Biol Sci. 2013 Dec 23;369(1635):20120532
    """
    if dist_func is None:
        assert field is not None and box_size is not None, (
            'either provide "dist_func" or "field" and "box_size"'
        )
        dist_func = distance_to_edge_function(
            x_c, y_c, field, box_size, interpolation="linear"
        )
    pos = np.array((x, y))

    # vector from pos to center p
    p_vec = ((x_c, y_c) - pos.T).T
    # angle between x-axis and negative vector p
    angle = (np.arctan2(p_vec[1], p_vec[0]) + np.pi) % (2 * np.pi)
    # distance from center to edge at each angle
    q = dist_func(angle)
    # distance from center to pos
    p = np.linalg.norm(p_vec, axis=0)
    # r-coordinate on unit circle
    r = p / q

    dpos = np.gradient(pos, axis=1)
    dt = np.gradient(t)
    velocity = np.divide(dpos, dt)

    # mean velocity vector v
    mean_velocity = np.average(velocity, axis=1)
    # angle on unit circle, run is rotated such that mean velocity vector
    # is toward positive x
    theta = (angle - np.arctan2(mean_velocity[1], mean_velocity[0])) % (2 * np.pi)

    w_pdcd = angle - np.arctan2(velocity[1], velocity[0])
    pdcd = r * np.cos(w_pdcd)

    w_pdmd = angle - np.arctan2(mean_velocity[1], mean_velocity[0])
    pdmd = r * np.cos(w_pdmd)
    return r, theta, pdcd, pdmd


def consecutive(array: np.ndarray, stepsize: float) -> List[np.ndarray]:
    """
    Splits array when distance between neighboring points is further than the stepsize.

    Parameters
    ----------
    array : np.ndarray
        Array to be split.
    stepsize : float
        Minimum distance to consider points as separate.

    Returns
    -------
    List[np.ndarray]
        List of arrays, split when jump greater than stepsize.
    """
    return np.split(array, np.where(np.diff(array) > stepsize)[0] + 1)


def find_fields_1d(
    tuning: List[np.ndarray],
    hz_thresh: float = 5,
    min_length: int = 1,
    max_length: int = 20,
    max_mean_firing: float = 10,
) -> dict:
    """
    Finds the location of maximum spiking.

    Parameters
    ----------
    tuning : List[np.ndarray]
        Each inner array contains the tuning curves for an individual neuron.
    hz_thresh : float, optional
        Any bin with firing above this value is considered to be part of a field. Default is 5.
    min_length : int, optional
        Minimum length of field (in tuning curve bin units). Default is 1.
    max_length : int, optional
        Maximum length of field (in tuning curve bin units). Default is 20.
    max_mean_firing : float, optional
        Only neurons with a mean firing rate less than this amount are considered for
        having place fields. Default is 10.

    Returns
    -------
    dict
        Where the key is the neuron number (int), and the value is a list of arrays (int)
        that are indices into the tuning curve where the field occurs.
        Each inner array contains the indices for a given place field.
    """
    fields = []
    for neuron_tc in tuning:
        if np.mean(neuron_tc) < max_mean_firing:
            neuron_field = np.zeros(neuron_tc.shape[0])
            for i, this_bin in enumerate(neuron_tc):
                if this_bin > hz_thresh:
                    neuron_field[i] = 1
            fields.append(neuron_field)
        else:
            fields.append(np.array([]))

    fields_idx = dict()
    for i, neuron_fields in enumerate(fields):
        field_idx = np.nonzero(neuron_fields)[0]
        fields_idx[i] = consecutive(field_idx, stepsize=1)

    with_fields = dict()
    for key in fields_idx:
        for field in fields_idx[key]:
            if len(field) > max_length:
                continue
            elif min_length <= len(field):
                with_fields[key] = fields_idx[key]
                continue
    return with_fields


def compute_linear_place_fields(
    firing_rate: np.ndarray,
    min_window_size: int = 5,
    min_firing_rate: float = 1.0,
    thresh: float = 0.5,
) -> np.ndarray:
    """
    Find consecutive bins where all are >= threshold of local max firing rate
    and the local max is > min_firing_rate.

    Parameters
    ----------
    firing_rate : np.ndarray
        Array of firing rates.
    min_window_size : int, optional
        Minimum size of the window. Default is 5.
    min_firing_rate : float, optional
        Minimum firing rate to consider a bin. Default is 1.0.
    thresh : float, optional
        Threshold percentage of local max. Default is 0.5.

    Returns
    -------
    np.ndarray
        Boolean array indicating place fields.
    """
    is_place_field = np.zeros(len(firing_rate), dtype="bool")
    for start in range(len(firing_rate) - min_window_size):
        for fin in range(start + min_window_size, len(firing_rate)):
            window = firing_rate[start:fin]
            mm = max(window)
            if mm > min_firing_rate and all(window > thresh * mm):
                is_place_field[start:fin] = True
            else:
                break

    return is_place_field


def compute_2d_place_fields(
    firing_rate: np.ndarray,
    min_firing_rate: float = 1,
    thresh: float = 0.2,
    min_size: int = 100,
    max_size: int = 200,
    sigma: Optional[float] = None,
) -> np.ndarray:
    """
    Compute place fields from the firing rate.

    Parameters
    ----------
    firing_rate : np.ndarray
        2D array of firing rates (NxN).
    min_firing_rate : float, optional
        Minimum firing rate in Hz. Default is 1.
    thresh : float, optional
        Percentage of local max. Default is 0.2.
    min_size : int, optional
        Minimum size of place field in pixels. Default is 100.
    max_size : int, optional
        Maximum size of place field in pixels. Default is 200.
    sigma : Optional[float], optional
        Standard deviation for Gaussian smoothing. Default is None.

    Returns
    -------
    np.ndarray
        2D array of receptive fields labeled with unique integers.
    """
    firing_rate_orig = firing_rate.copy()

    if sigma is not None:
        firing_rate = gaussian_filter(firing_rate, sigma)

    local_maxima_inds = firing_rate == maximum_filter(firing_rate, 3)
    receptive_fields = np.zeros(firing_rate.shape, dtype=int)
    n_receptive_fields = 0
    firing_rate = firing_rate.copy()
    for local_max in np.flipud(np.sort(firing_rate[local_maxima_inds])):
        labeled_image, num_labels = label(
            firing_rate > max(local_max * thresh, min_firing_rate)
        )

        if not num_labels:  # nothing above min_firing_thresh
            continue
        for i in range(1, num_labels + 1):
            image_label = labeled_image == i
            if local_max in firing_rate[image_label]:
                break
            if np.sum(image_label) >= min_size:
                n_receptive_fields += 1
                receptive_fields[image_label] = n_receptive_fields
                firing_rate[image_label] = 0

    receptive_fields = remove_fields_by_area(
        receptive_fields, int(min_size), maximum_field_area=max_size
    )
    if n_receptive_fields > 0:
        receptive_fields = sort_fields_by_rate(
            firing_rate_orig, receptive_fields, func=np.max
        )
    return receptive_fields


def find_field(
    firing_rate: np.ndarray, threshold: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the field in the firing rate that exceeds the threshold.

    Parameters
    ----------
    firing_rate : np.ndarray
        Array of firing rates.
    threshold : float
        Threshold for detection.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple containing the image label and the same label.
    """
    mm = np.max(firing_rate)

    labeled_image, num_labels = label(firing_rate > threshold)
    for i in range(1, num_labels + 1):
        image_label = labeled_image == i
        if mm in firing_rate[image_label]:
            return image_label, image_label


def find_field2(
    firing_rate: np.ndarray, thresh: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the field in a 1D firing rate array that exceeds the threshold.

    Parameters
    ----------
    firing_rate : np.ndarray
        1D array of firing rates.
    thresh : float
        Threshold for detection.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple containing two boolean arrays:
        the first indicates the buffer area and the second indicates the field.
    """
    firing_rate = np.array(firing_rate)
    imm = np.argmax(firing_rate)
    mm = np.max(firing_rate)
    # TODO: make more efficient by using argmax instead of where()[0]
    first = np.where(np.diff(firing_rate[:imm]) < 0)[0]
    if len(first) == 0:
        first = 0
    else:
        first = first[-1] + 2

    last = np.where(np.diff(firing_rate[imm:]) > 0)[0]

    if len(last) == 0:
        last = len(firing_rate)
    else:
        last = last[0] + imm + 1
    field_buffer = np.zeros(firing_rate.shape, dtype="bool")
    field_buffer[first:last] = True
    field = field_buffer & (firing_rate > thresh * mm)

    return field_buffer, field


def map_stats2(
    firing_rate: np.ndarray,
    threshold: float = 0.1,
    min_size: int = 5,
    max_size: Optional[int] = None,
    min_peak: float = 1.0,
    sigma: Optional[float] = None,
) -> Dict[str, List[float]]:
    """
    Map statistics of firing rate fields.

    Parameters
    ----------
    firing_rate : np.ndarray
        1D array of firing rates.
    threshold : float, optional
        Threshold for field detection. Default is 0.1.
    min_size : int, optional
        Minimum size of detected fields. Default is 5.
    max_size : Optional[int], optional
        Maximum size of detected fields. Default is None, which sets it to the length of firing_rate.
    min_peak : float, optional
        Minimum peak firing rate to consider a field valid. Default is 1.0.
    sigma : Optional[float], optional
        Standard deviation for Gaussian smoothing. Default is None.

    Returns
    -------
    Dict[str, List[float]]
        A dictionary containing the sizes, peaks, means, and fields of detected firing rate fields.
    """
    if sigma is not None:
        firing_rate = gaussian_filter1d(firing_rate, sigma)

    if max_size is None:
        max_size = len(firing_rate)

    firing_rate = firing_rate.copy()
    firing_rate = firing_rate - np.min(firing_rate)
    out = dict(sizes=list(), peaks=list(), means=list())
    out["fields"] = np.zeros(firing_rate.shape)
    field_counter = 1
    while True:
        peak = np.max(firing_rate)
        if peak < min_peak:
            break
        field_buffer, field = find_field(firing_rate, threshold)
        field_size = np.sum(field)
        if (
            (field_size > min_size)
            and (field_size < max_size)
            and (np.max(firing_rate[field]) > (2 * np.min(firing_rate[field_buffer])))
        ):
            out["fields"][field] = field_counter
            out["sizes"].append(float(field_size) / len(firing_rate))
            out["peaks"].append(peak)
            out["means"].append(np.mean(firing_rate[field]))
            field_counter += 1
        firing_rate[field_buffer] = 0

    return out
