"""Regression tests for version-aware MATLAB I/O."""

from unittest.mock import patch

import h5py
import nelpy as nel
import numpy as np
import pytest
import scipy.io as sio

from neuro_py.io import epoch_to_mat, load_events, load_mat, save_mat


@pytest.mark.parametrize("format", ["v7", "v7.3"])
def test_save_mat_roundtrips_simple_nested_data(tmp_path, format):
    """Both supported MAT formats preserve a representative nested payload."""
    filename = tmp_path / f"data_{format}.mat"
    payload = {"events": {"timestamps": np.array([[1.0, 2.0]]), "peaks": 1.5}}

    save_mat(filename, payload, format=format)
    loaded = load_mat(filename)

    assert "events" in loaded
    assert np.asarray(loaded["events"]["timestamps"]).squeeze().tolist() == [1.0, 2.0]


def test_load_mat_preserves_legacy_scipy_behavior(tmp_path):
    """Legacy files continue to use SciPy's simplify_cells contract."""
    filename = tmp_path / "legacy.mat"
    sio.savemat(filename, {"events": {"peaks": np.array([1.0, 2.0])}})

    loaded = load_mat(filename)

    assert np.asarray(loaded["events"]["peaks"]).tolist() == [1.0, 2.0]


def test_save_mat_v73_uses_matlab_compatible_backend(tmp_path):
    """The large-file path selects hdf5storage without a huge test allocation."""
    filename = tmp_path / "large.mat"
    with patch("hdf5storage.savemat") as save:
        save_mat(filename, {"large": np.empty(0)}, format="v7.3")

    assert save.call_args.kwargs["fmt"] == "7.3"
    assert save.call_args.kwargs["matlab_compatible"] is True


def test_load_events_reads_v73_file(tmp_path):
    """A public IO loader handles the HDF5-backed MAT-file path end to end."""
    basepath = tmp_path / "session"
    basepath.mkdir()
    filename = basepath / "session.ripples.events.mat"
    save_mat(
        filename,
        {
            "ripples": {
                "timestamps": np.array([[1.0, 2.0], [3.0, 4.0]]),
                "peaks": np.array([1.5, 3.5]),
            }
        },
        format="v7.3",
    )

    events = load_events(str(basepath), "ripples", load_pandas=True)

    assert events["starts"].tolist() == [1.0, 3.0]
    assert events["stops"].tolist() == [2.0, 4.0]


def test_save_mat_v73_replaces_existing_legacy_file(tmp_path):
    """Converting a legacy MAT-file in place creates a valid v7.3 file."""
    filename = tmp_path / "session.mat"
    sio.savemat(filename, {"session": {"timestamps": np.array([[1.0, 2.0]])}})

    save_mat(filename, load_mat(filename), format="v7.3")

    assert h5py.is_hdf5(filename)
    assert "session" in load_mat(filename)


def test_save_mat_rejects_unknown_format(tmp_path):
    """Unsupported MAT formats fail clearly rather than silently changing output."""
    with pytest.raises(ValueError, match="format"):
        save_mat(tmp_path / "data.mat", {}, format="v7.4")  # type: ignore[arg-type]


def test_epoch_to_mat_supports_legacy_and_v73_formats(tmp_path):
    """CellExplorer event output retains its schema in both MAT-file formats."""
    epoch = nel.EpochArray([[1.0, 2.0]])
    for format in ("v7", "v7.3"):
        basepath = tmp_path / format
        basepath.mkdir()
        epoch_to_mat(epoch, str(basepath), "events", format=format)
        filename = basepath / f"{format}.events.events.mat"

        assert filename.exists()
        assert h5py.is_hdf5(filename) is (format == "v7.3")
        assert "events" in load_mat(filename)
