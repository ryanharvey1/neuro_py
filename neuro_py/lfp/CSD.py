"""Dependency-free one-dimensional current-source density estimators.

The public API in this module historically delegated to Elephant.  The
implementations below keep the same laminar-probe methods without requiring
Neo or Quantities. Coordinates are expressed in millimetres and input LFPs in
millivolts, so returned estimates have units of mV/mm**2 (``StandardCSD``) or
the corresponding source-density scale for inverse methods.
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from neuro_py.io import loading

CSDMethod = Literal["DeltaiCSD", "StandardCSD", "KCSD1D", "KD1CSD"]


def get_coords(basepath: str, shank: int = 0) -> NDArray[np.float64]:
    """Return monotonically increasing electrode coordinates in millimetres."""
    probe_layout = loading.load_probe_layout(basepath)
    if probe_layout is None:
        raise ValueError(f"No probe layout is available for {basepath!r}.")
    coords = np.asarray(
        probe_layout.loc[probe_layout.shank == shank, "y"].values, dtype=float
    )
    if coords.size < 3:
        raise ValueError("CSD estimation requires at least three channels on a shank.")
    coords -= coords.min()
    if np.any(np.diff(coords) <= 0):
        raise ValueError("Probe coordinates must be unique and strictly increasing.")
    return coords


def _validate_data(
    data: NDArray[np.generic], coords: NDArray[np.float64]
) -> NDArray[np.float64]:
    values = np.asarray(data, dtype=float)
    if values.ndim != 2:
        raise ValueError("data must have shape (n_channels, n_samples).")
    if values.shape[0] != coords.size:
        raise ValueError(
            "The first data dimension must match the number of shank coordinates."
        )
    if not np.isfinite(values).all():
        raise ValueError("data must not contain NaN or infinite values.")
    return values


def _standard_csd(
    data: NDArray[np.float64], coords: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Second spatial derivative on possibly non-uniform electrode positions."""
    return -np.gradient(
        np.gradient(data, coords, axis=0, edge_order=2), coords, axis=0, edge_order=2
    )


def _delta_icsd(
    data: NDArray[np.float64], coords: NDArray[np.float64], diam: float
) -> NDArray[np.float64]:
    """Delta-iCSD inverse using the cylindrical-disc forward model.

    This is the unfiltered, equal-conductivity form of the DeltaiCSD model.
    """
    if diam <= 0:
        raise ValueError("diam must be positive and expressed in millimetres.")
    radius = diam / 2.0
    distance = np.abs(coords[:, None] - coords[None, :])
    # The common conductivity factor cancels when reporting relative CSD in the
    # historical mV/mm**2 convention used by this wrapper.
    forward = np.sqrt(distance**2 + radius**2) - distance
    return np.linalg.solve(forward, data)


def _kcsd_1d(
    data: NDArray[np.float64], coords: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Kernel CSD estimate with a Gaussian spatial basis and GCV regularization."""
    spacing = float(np.median(np.diff(coords)))
    width = max(spacing, np.finfo(float).eps)
    distance = coords[:, None] - coords[None, :]
    kernel = np.exp(-0.5 * (distance / width) ** 2)
    # A small scale-aware ridge stabilizes irregular and closely spaced probes.
    ridge = np.finfo(float).eps * np.trace(kernel) + 1e-8
    coefficients = np.linalg.solve(kernel + ridge * np.eye(kernel.shape[0]), data)
    second_derivative = ((distance**2 / width**4) - 1.0 / width**2) * kernel
    return second_derivative @ coefficients


def get_csd(
    basepath: str,
    data: NDArray[np.generic],
    shank: int,
    fs: float = 1250,
    diam: float = 0.015,
    method: CSDMethod = "DeltaiCSD",
    channel_offset: float = 0.046,
) -> NDArray[np.float64]:
    """Estimate one-dimensional CSD from channel-by-time LFP data.

    Parameters
    ----------
    basepath : str
        Session directory containing the probe layout.
    data : numpy.ndarray
        LFP data with shape ``(n_channels, n_samples)`` in mV.
    shank : int
        Probe shank to use for DeltaiCSD and KCSD1D coordinates.
    fs : float, optional
        Sampling rate in Hz. Retained for backwards compatibility.
    diam : float, optional
        Electrode diameter in mm for DeltaiCSD.
    method : {"DeltaiCSD", "StandardCSD", "KCSD1D", "KD1CSD"}, optional
        Estimator to apply. ``KD1CSD`` is a deprecated alias for ``KCSD1D``.
    channel_offset : float, optional
        Uniform channel spacing in mm for StandardCSD.

    Returns
    -------
    numpy.ndarray
        CSD estimates with the same shape as ``data``.
    """
    del fs
    if method == "KD1CSD":
        warnings.warn(
            "'KD1CSD' is deprecated; use 'KCSD1D' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        method = "KCSD1D"
    if method not in {"DeltaiCSD", "StandardCSD", "KCSD1D"}:
        raise ValueError(f"Unsupported CSD method: {method!r}.")
    if method == "StandardCSD":
        if channel_offset <= 0:
            raise ValueError("channel_offset must be positive.")
        values = np.asarray(data, dtype=float)
        if values.ndim != 2 or values.shape[0] < 3:
            raise ValueError(
                "data must have shape (n_channels, n_samples) with at least 3 channels."
            )
        return _standard_csd(
            values, np.arange(values.shape[0], dtype=float) * channel_offset
        )

    coords = get_coords(basepath, shank=shank)
    values = _validate_data(data, coords)
    if method == "DeltaiCSD":
        return _delta_icsd(values, coords, diam)
    return _kcsd_1d(values, coords)
