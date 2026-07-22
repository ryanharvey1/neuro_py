"""Version-aware MATLAB MAT-file helpers."""

from pathlib import Path
from typing import Any, Literal

import h5py
from scipy import io as sio

MatFormat = Literal["v7", "v7.3"]


def _v73_dependency_error(dependency: str) -> ImportError:
    """Create an actionable error for optional v7.3 MAT-file support."""
    return ImportError(
        f"MATLAB v7.3 support requires {dependency}. "
        "For an editable checkout, run: python -m pip install -e ."
    )


def load_mat(filename: str | Path, *, simplify_cells: bool = True) -> dict[str, Any]:
    """Load a legacy or MATLAB v7.3 MAT-file.

    Parameters
    ----------
    filename : str or pathlib.Path
        MAT-file to load.
    simplify_cells : bool, optional
        Forwarded to :func:`scipy.io.loadmat` for legacy MAT-files. MATLAB v7.3
        files are read by ``pymatreader``, whose output is already simplified.

    Returns
    -------
    dict
        Variables stored in the MAT-file.
    """
    path = str(filename)
    if h5py.is_hdf5(path):
        try:
            from pymatreader import read_mat
        except ModuleNotFoundError as error:
            raise _v73_dependency_error("pymatreader") from error
        return read_mat(path)
    return sio.loadmat(path, simplify_cells=simplify_cells)


def save_mat(
    filename: str | Path,
    data: dict[str, Any],
    *,
    format: MatFormat = "v7",
) -> None:
    """Save MATLAB-compatible data in legacy or v7.3 MAT format.

    Parameters
    ----------
    filename : str or pathlib.Path
        Destination MAT-file.
    data : dict
        MATLAB-compatible variables to store.
    format : {"v7", "v7.3"}, optional
        MAT-file format. The default preserves the legacy SciPy writer.
    """
    if format == "v7":
        sio.savemat(filename, data, long_field_names=True)
        return
    if format == "v7.3":
        try:
            import hdf5storage
        except ModuleNotFoundError as error:
            raise _v73_dependency_error("hdf5storage") from error
        hdf5storage.savemat(
            filename,
            data,
            fmt="7.3",
            matlab_compatible=True,
            truncate_existing=True,
        )
        return
    raise ValueError("format must be either 'v7' or 'v7.3'")
