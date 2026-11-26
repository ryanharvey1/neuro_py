"""Registration, metadata fidelity, real-loader integration, and output bounds."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat

from neuro_py.mcp.dataset import MAX_RESPONSE_BYTES, Dataset, _bounded, _json_safe


def registered(catalog):
    dataset = Dataset(catalogs=[catalog])
    session_id = dataset.list_sessions()["rows"][0]["session_id"]
    return dataset, session_id


def test_catalog_deduplicates_sessions_preserving_each_registration(catalog, session):
    dataset, session_id = registered(catalog)
    rows = dataset.list_sessions()["rows"]
    assert len(rows) == 1
    assert rows[0]["basepath"] == str(session)
    assert len(rows[0]["registrations"]) == 2
    assert rows[0]["registrations"][0]["metadata"]["animal"] == "rat01"
    assert (
        Dataset(sessions=[session]).list_sessions()["rows"][0]["session_id"]
        == session_id
    )
    assert dataset.list_sessions("LINEAR")["total_rows"] == 1
    assert dataset.list_sessions(".*")["total_rows"] == 0


def test_root_registration_and_file_inventory(session):
    dataset = Dataset(roots=[session.parent])
    row = dataset.list_sessions()["rows"][0]
    result = dataset.get_session(row["session_id"])
    assert result["status"] == "ok"
    assert all(file["available"] for file in result["files"].values())
    assert result["files"]["epochs"]["size_bytes"] > 0


def test_path_maps_are_prefix_bounded_and_preserve_catalog_identity(tmp_path):
    filename = tmp_path / "sessions.csv"
    pd.DataFrame({"basepath": [r"Z:\Data\rat\day01", r"Z:\Database\rat\day01"]}).to_csv(
        filename, index=False
    )
    original = Dataset(catalogs=[filename]).list_sessions()["rows"]
    remapped = Dataset(
        catalogs=[filename], path_maps=[(r"z:\data", str(tmp_path / "mounted"))]
    ).list_sessions()["rows"]
    assert [r["session_id"] for r in remapped] == [r["session_id"] for r in original]
    assert remapped[0]["basepath"] == str(tmp_path / "mounted" / "rat" / "day01")
    assert remapped[1]["basepath"] == original[1]["basepath"]
    dataset = Dataset(catalogs=[filename])
    assert dataset.list_sessions(r"Data\rat")["total_rows"] == 1
    assert dataset.list_sessions(r"Z:\Data\rat\day01")["total_rows"] == 1


def test_epochs_keep_original_labels_times_and_loader_rows(catalog):
    dataset, session_id = registered(catalog)
    first = dataset.read_table(session_id, "epochs", limit=2)
    second = dataset.read_table(session_id, "epochs", offset=first["next_offset"])
    assert first["total_rows"] == 3
    assert first["rows"][1]["name"] == "maze"
    assert first["rows"][1]["startTime"] == 1.0
    assert second["rows"][0]["source_row"] == 2
    assert second["rows"][0]["environment"] == "sleep"
    assert second["next_offset"] is None
    assert first["time_units"] == "seconds"
    assert first["source_size_bytes"] > 0
    assert first["rows"][0]["manipulation"] is None


def test_cells_keep_uids_regions_and_default_bad_unit_exclusion(catalog):
    dataset, session_id = registered(catalog)
    result = dataset.read_table(session_id, "cells")
    assert [row["UID"] for row in result["rows"]] == [7, 99]
    assert [row["source_row"] for row in result["rows"]] == [0, 2]
    assert result["rows"][0]["brainRegion"] == "CA1sp"
    assert result["rows"][1]["firingRate"] is None
    assert "firingRate" in result["available_columns"]
    assert (
        dataset.read_table(session_id, "cells", include_bad_units=True)["total_rows"]
        == 3
    )
    selected = dataset.read_table(
        session_id, "cells", region="ca1", columns=["brainRegion"]
    )
    assert selected["rows"] == [{"source_row": 0, "UID": 7, "brainRegion": "CA1sp"}]
    assert dataset.read_table(session_id, "cells", region="CA1.*")["status"] == "empty"
    assert (
        dataset.read_table(session_id, "cells", cell_type="int")["rows"][0]["UID"] == 99
    )


def test_trials_preserve_labels(catalog):
    dataset, session_id = registered(catalog)
    result = dataset.read_table(session_id, "trials")
    assert [r["trialsID"] for r in result["rows"]] == ["out", "in"]
    assert result["rows"][0]["startTime"] == 1.1


def test_ripples_use_loader_flagged_and_manual_event_behavior(catalog):
    dataset, session_id = registered(catalog)
    original = dataset.read_table(session_id, "ripples", manual_events=False)
    assert original["total_rows"] == 2
    assert [r["peaks"] for r in original["rows"]] == [0.15, 0.55]
    assert original["rows"][0]["detectorName"] == "synthetic"
    manual = dataset.read_table(session_id, "ripples", manual_events=True)
    assert manual["total_rows"] == 3
    assert manual["parameters"]["manual_events"] is True


def test_sleep_intervals_missing_state_and_empty_state_are_distinct(catalog):
    dataset, session_id = registered(catalog)
    result = dataset.read_table(session_id, "sleep_states")
    assert result["total_rows"] == 2
    assert result["rows"][1] == {
        "source_row": 1,
        "start": 3.0,
        "stop": 4.0,
        "state": "NREMstate",
    }
    assert (
        dataset.read_table(session_id, "sleep_states", state="THETA")["status"]
        == "missing_state"
    )
    assert (
        dataset.read_table(session_id, "sleep_states", state="REMstate")["status"]
        == "empty"
    )


def test_missing_file_empty_table_and_unavailable_session_are_distinct(
    catalog, session
):
    dataset, session_id = registered(catalog)
    behavior = session / (session.name + ".animal.behavior.mat")
    savemat(behavior, {"behavior": {"trials": np.empty((0, 2))}})
    assert dataset.read_table(session_id, "trials")["status"] == "empty"
    behavior.unlink()
    assert dataset.read_table(session_id, "trials")["status"] == "missing_file"
    missing = Dataset(sessions=[session / "offline"])
    missing_id = missing.list_sessions()["rows"][0]["session_id"]
    assert missing.get_session(missing_id)["status"] == "unavailable"
    assert missing.read_table(missing_id, "cells")["status"] == "unavailable"


@pytest.mark.parametrize("options", [{"limit": 0}, {"limit": 201}, {"offset": -1}])
def test_invalid_pagination_rejected_before_file_access(catalog, options):
    dataset, session_id = registered(catalog)
    with pytest.raises(ValueError, match="offset"):
        dataset.read_table(session_id, "cells", **options)
    with pytest.raises(ValueError, match="offset"):
        dataset.list_sessions(**options)


def test_unknown_paths_products_columns_and_states_rejected(catalog):
    dataset, session_id = registered(catalog)
    with pytest.raises(ValueError, match="Unknown session_id"):
        dataset.read_table("../../private", "epochs")
    with pytest.raises(ValueError, match="Unknown data product"):
        dataset.read_table(session_id, "../../private")
    with pytest.raises(ValueError, match="Unknown columns"):
        dataset.read_table(session_id, "cells", columns=["not_a_metric"])
    with pytest.raises(ValueError, match="Unsupported sleep-state"):
        dataset.read_table(session_id, "sleep_states", state="anything")


def test_read_does_not_modify_source_files(catalog, session):
    dataset, session_id = registered(catalog)
    before = {p: hashlib.sha256(p.read_bytes()).digest() for p in session.iterdir()}
    for product in ["epochs", "cells", "trials", "ripples", "sleep_states"]:
        dataset.read_table(session_id, product)
    assert before == {
        p: hashlib.sha256(p.read_bytes()).digest() for p in session.iterdir()
    }


def test_json_nulls_numpy_scalars_and_payload_bound():
    result = _json_safe({"n": np.array(7), "bad": [np.nan, np.inf, pd.NA]})
    assert json.loads(json.dumps(result, allow_nan=False)) == {
        "n": 7,
        "bad": [None] * 3,
    }
    with pytest.raises(ValueError, match="128 KB"):
        _bounded({"text": "x" * MAX_RESPONSE_BYTES})


def test_empty_registration_and_invalid_catalog_fail_clearly(tmp_path):
    with pytest.raises(ValueError, match="No sessions"):
        Dataset()
    filename = tmp_path / "bad.csv"
    filename.write_text("other\nvalue\n")
    with pytest.raises(ValueError, match="basepath"):
        Dataset(catalogs=[filename])


def test_base_package_and_catalog_work_without_optional_mcp_sdk():
    script = """
import importlib.abc
import sys
class BlockSdk(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'mcp' or fullname.startswith('mcp.'):
            raise ModuleNotFoundError('MCP SDK intentionally unavailable')
sys.meta_path.insert(0, BlockSdk())
import neuro_py
from neuro_py import mcp
from neuro_py.mcp.dataset import Dataset
assert not any(name == 'mcp' or name.startswith('mcp.') for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
