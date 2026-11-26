"""Directory-alias identity, recording access, and physical cache containment."""

import os
import shutil
from pathlib import Path

import pytest

from neuro_py.mcp.dataset import Dataset
from neuro_py.mcp.index import IndexBuilder, MetadataIndex


@pytest.mark.parametrize(
    "registration", ["relative_catalog", "absolute_catalog", "session", "root"]
)
def test_symlinked_registrations_match_index_scope(
    session, tmp_path, registration, directory_alias
):
    alias = tmp_path / "mounted"
    directory_alias(alias, session.parent)
    aliased_session = alias / session.name
    cache = tmp_path / "private" / "index.sqlite"
    IndexBuilder(cache).build(roots=[alias])

    if registration.endswith("catalog"):
        catalog = tmp_path / "sessions.csv"
        basepath = (
            f"mounted/{session.name}"
            if registration == "relative_catalog"
            else str(aliased_session)
        )
        catalog.write_text(f"basepath\n{basepath}\n")
        dataset = Dataset(catalogs=[catalog], index_db=cache)
    elif registration == "session":
        dataset = Dataset(sessions=[aliased_session], index_db=cache)
    else:
        dataset = Dataset(roots=[alias], index_db=cache)

    listed = dataset.list_sessions()["rows"][0]
    assert listed["catalog_basepath"] == str(aliased_session)
    assert listed["basepath"] == str(aliased_session)
    indexed = dataset.search_sessions(scope="session_list")
    assert indexed["total_rows"] == 1
    assert indexed["rows"][0]["in_session_list"]
    assert indexed["rows"][0]["session_id"] == listed["session_id"]
    assert dataset.summarize_index(scope="session_list")["sessions_in_scope"] == 1
    assert dataset.read_table(listed["session_id"], "epochs")["total_rows"] == 3


@pytest.mark.parametrize("registration", ["root", "session"])
@pytest.mark.parametrize("cache_prefix", ["real", "alias"])
def test_cache_inside_symlinked_recordings_is_rejected_before_writes(
    tmp_path, registration, cache_prefix, directory_alias
):
    recordings = tmp_path / "recordings"
    (recordings / "day01").mkdir(parents=True)
    alias = tmp_path / "mounted"
    directory_alias(alias, recordings)
    prefix = recordings if cache_prefix == "real" else alias
    cache = prefix / "day01" / "private" / "index.sqlite"
    builder = IndexBuilder(cache)

    with pytest.raises(ValueError, match="outside recording roots"):
        if registration == "root":
            builder.build(roots=[alias])
        else:
            builder.build(sessions=[alias / "day01"])

    assert not cache.parent.exists()
    assert not cache.exists()
    assert not cache.with_name(cache.name + ".lock").exists()


@pytest.mark.parametrize(
    "registration",
    ["relative_catalog", "absolute_catalog", "mapped_catalog", "session", "root"],
)
@pytest.mark.parametrize("index_mode", ["root", "session"])
def test_renamed_session_alias_preserves_identity_and_loads_recording(
    session, tmp_path, registration, index_mode, directory_alias
):
    alias = tmp_path / "session_shortcut"
    directory_alias(alias, session)
    cache = tmp_path / "private" / "index.sqlite"
    inputs = {"roots": [alias]} if index_mode == "root" else {"sessions": [alias]}
    status = IndexBuilder(cache).build(**inputs)
    assert status["session_count"] == 1
    assert status["metadata_status_counts"] == {"ok": 1}

    if registration.endswith("catalog"):
        catalog = tmp_path / "sessions.csv"
        basepath = str(alias) if registration == "absolute_catalog" else alias.name
        options = {}
        if registration == "mapped_catalog":
            basepath = r"Z:\session_shortcut"
            options["path_maps"] = [("Z:\\", str(tmp_path))]
        catalog.write_text(f"basepath\n{basepath}\n")
        dataset = Dataset(catalogs=[catalog], index_db=cache, **options)
    elif registration == "session":
        dataset = Dataset(sessions=[alias], index_db=cache)
    else:
        dataset = Dataset(roots=[alias], index_db=cache)

    listed = dataset.list_sessions()["rows"][0]
    assert listed["basepath"] == str(alias)
    found = dataset.search_sessions(
        scope="session_list", environments=["linear"], regions=["CA1"]
    )
    assert found["total_rows"] == 1
    assert found["rows"][0]["session_id"] == listed["session_id"]
    assert found["rows"][0]["event_types"] == ["ripples"]
    assert dataset.summarize_index(scope="session_list")["sessions_in_scope"] == 1
    inventory = dataset.get_session(listed["session_id"])
    assert inventory["basename"] == session.name
    assert all(info["available"] for info in inventory["files"].values())
    for product in ["epochs", "cells", "trials", "ripples", "sleep_states"]:
        page = dataset.read_table(listed["session_id"], product)
        assert page["status"] == "ok"
        assert page["basepath"] == str(alias)
        assert page["basename"] == session.name
        assert Path(page["source"]).parent == session.resolve()
    assert [
        row["UID"] for row in dataset.read_table(listed["session_id"], "cells")["rows"]
    ] == [7, 99]

    # Discovered IDs work without a catalog registration, too.
    discovered = Dataset(index_db=cache)
    indexed_id = discovered.search_sessions()["rows"][0]["session_id"]
    assert discovered.read_table(indexed_id, "epochs")["total_rows"] == 3


def test_catalog_under_aliased_parent_preserves_relative_session_identity(
    session, tmp_path, directory_alias
):
    alias = tmp_path / "mounted"
    directory_alias(alias, session.parent)
    (session.parent / "sessions.csv").write_text(f"basepath\n{session.name}\n")
    cache = tmp_path / "private" / "index.sqlite"
    IndexBuilder(cache).build(roots=[alias])
    catalog = alias / "sessions.csv"
    dataset = Dataset(catalogs=[catalog], index_db=cache)
    listed = dataset.list_sessions()["rows"][0]
    assert listed["basepath"] == str(alias / session.name)
    assert listed["registrations"][0]["source"] == str(catalog)
    assert dataset.search_sessions(scope="session_list")["total_rows"] == 1
    assert dataset.summarize_index(scope="session_list")["sessions_in_scope"] == 1
    assert dataset.read_table(listed["session_id"], "epochs")["total_rows"] == 3


def test_retargeted_session_alias_refreshes_metadata_with_identical_source_stats(
    session, tmp_path, directory_alias, monkeypatch
):
    from neuro_py.io import loading

    other = tmp_path / "other" / session.name
    shutil.copytree(session, other)
    alias = tmp_path / "session_shortcut"
    directory_alias(alias, session)
    cache = tmp_path / "private" / "index.sqlite"
    IndexBuilder(cache).build(sessions=[alias])
    index = MetadataIndex(cache)
    first = index.search()["rows"][0]

    # Remove only the test's link/junction; both recordings remain in place.
    if os.name == "nt":
        alias.rmdir()
    else:
        alias.unlink()
    directory_alias(alias, other)
    original = loading.load_epoch
    calls = []

    def track(basepath):
        calls.append(basepath)
        return original(basepath)

    monkeypatch.setattr(loading, "load_epoch", track)
    IndexBuilder(cache).build(sessions=[alias])
    refreshed = index.search()["rows"][0]
    assert calls == [str(other.resolve())]
    assert refreshed["session_id"] == first["session_id"]
    assert refreshed["files"] == first["files"]
    assert refreshed["recording_path"] == str(other.resolve())
    assert first["recording_path"] == str(session.resolve())
