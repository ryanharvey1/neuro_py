"""Scientific metadata fidelity, resumable discovery, and private cached queries."""

import json
import shutil
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat

from neuro_py.mcp.dataset import Dataset
from neuro_py.mcp.index import IndexBuilder, MetadataIndex


@pytest.fixture
def indexed(session, tmp_path):
    cache = tmp_path / "private" / "index.sqlite"
    status = IndexBuilder(cache).build(roots=[session.parent])
    assert status["discovery_complete"]
    return MetadataIndex(cache)


def test_recording_labels_counts_and_file_types(indexed, session):
    found = indexed.search(environments=["linear"], regions=["ca1"], cell_type="pyr")
    assert found["total_rows"] == 1
    row = found["rows"][0]
    assert row["basepath"] == str(session)
    assert row["metadata"]["animals"] == ["rat01"]
    assert row["metadata"]["environments"] == ["linear", "sleep"]
    assert row["metadata"]["cell_count_total"] == 3
    assert row["metadata"]["cell_count_good"] == 2
    assert row["metadata"]["recording_regions"] == ["CA1", "PFC"]
    assert row["metadata"]["recording"]["sr"] == 30000.0
    assert indexed.search(recording_regions=["CA1", "PFC"])["total_rows"] == 1
    assert row["matching_cell_count"] == 1
    assert row["event_types"] == ["ripples"]
    assert row["files"]["behavior"]["present"]
    assert not row["files"]["lfp"]["present"]
    assert row["observed_at"]
    assert indexed.search(query="RAT01")["total_rows"] == 1
    assert indexed.search(query=".*")["total_rows"] == 0
    assert (
        indexed.search(products=["cells", "ripples", "sleep_states"])["total_rows"] == 1
    )
    assert indexed.search(products=["lfp"])["total_rows"] == 0


def test_joint_region_type_filters_and_bad_units(indexed):
    assert indexed.search(regions=["CA1", "PFC"])["total_rows"] == 1
    assert indexed.search(regions=["CA1"], cell_type="Int")["total_rows"] == 0
    assert indexed.search(regions=["PFC"], cell_type="Int")["total_rows"] == 1
    assert indexed.search(regions=["CA1", "PFC"], cell_type="Pyr")["total_rows"] == 0
    assert indexed.search(regions=["CA1so"])["total_rows"] == 0
    assert indexed.search(regions=["CA1so"], include_bad_units=True)["total_rows"] == 1
    assert indexed.search(regions=["CA1.*"])["total_rows"] == 0


def test_summary_counts_sessions_not_cells_and_preserves_labels(indexed):
    assert indexed.summarize("region")["rows"] == [
        {"label": "CA1sp", "session_count": 1},
        {"label": "PFC", "session_count": 1},
    ]
    assert indexed.summarize("environment")["rows"] == [
        {"label": "linear", "session_count": 1},
        {"label": "sleep", "session_count": 1},
    ]
    assert indexed.summarize("animal")["rows"] == [
        {"label": "rat01", "session_count": 1}
    ]
    assert indexed.summarize("cell_type")["sessions_without_label"] == 0
    assert indexed.summarize("event_type")["rows"] == [
        {"label": "ripples", "session_count": 1}
    ]


def test_scope_preserves_session_list_and_allows_reads_of_discovered_sessions(
    session, tmp_path
):
    other = tmp_path / "rat02" / "day01"
    shutil.copytree(session, other)
    cache = tmp_path / "private" / "index.sqlite"
    IndexBuilder(cache).build(roots=[session.parent, other.parent])
    csv = tmp_path / "sessions.csv"
    pd.DataFrame({"basepath": [str(session)]}).to_csv(csv, index=False)
    dataset = Dataset(catalogs=[csv], index_db=cache)
    assert dataset.list_sessions()["total_rows"] == 1
    assert dataset.search_sessions()["total_rows"] == 2
    scoped = dataset.search_sessions(scope="session_list")
    assert scoped["total_rows"] == 1
    assert scoped["rows"][0]["in_session_list"]
    assert (
        scoped["rows"][0]["session_id"]
        == dataset.list_sessions()["rows"][0]["session_id"]
    )
    discovered = next(
        row for row in dataset.search_sessions()["rows"] if not row["in_session_list"]
    )
    assert dataset.read_table(discovered["session_id"], "epochs")["total_rows"] == 3
    assert dataset.summarize_index(scope="session_list")["sessions_in_scope"] == 1
    assert Dataset(index_db=cache).search_sessions()["total_rows"] == 2
    first = dataset.search_sessions(limit=1)
    second = dataset.search_sessions(
        offset=first["next_offset"], limit=1, page_token=first["page_token"]
    )
    assert first["total_rows"] == second["total_rows"] == 2
    assert first["rows"][0]["session_id"] != second["rows"][0]["session_id"]
    assert second["next_offset"] is None


def test_pagination_rejects_refresh_even_when_total_count_is_unchanged(tmp_path):
    paths = [tmp_path / name for name in ("a", "b", "c", "e")]
    for path in paths:
        path.mkdir()
    a, b, c, e = paths
    cache = tmp_path / "private" / "index.sqlite"
    IndexBuilder(cache).build(sessions=[a, c, e])
    index = MetadataIndex(cache)
    first = index.search(limit=2)
    assert [Path(row["basepath"]).name for row in first["rows"]] == ["a", "c"]
    IndexBuilder(cache).build(sessions=[a, b, c])
    assert index.search()["total_rows"] == first["total_rows"] == 3
    with pytest.raises(ValueError, match="restart at offset=0"):
        index.search(
            offset=first["next_offset"], limit=2, page_token=first["page_token"]
        )
    restarted = index.search(limit=2)
    last = index.search(
        offset=restarted["next_offset"], limit=2, page_token=restarted["page_token"]
    )
    assert [Path(row["basepath"]).name for row in restarted["rows"] + last["rows"]] == [
        "a",
        "b",
        "c",
    ]


def test_summary_pagination_detects_metadata_only_changes(indexed):
    first = indexed.summarize("region", limit=1)
    with sqlite3.connect(indexed.path) as connection:
        key, payload = connection.execute(
            "SELECT session_id, payload FROM sessions"
        ).fetchone()
        data = json.loads(payload)
        data["metadata"]["cell_groups"][0]["region"] = "CA2"
        connection.execute(
            "UPDATE sessions SET payload=? WHERE session_id=?", (json.dumps(data), key)
        )
    with pytest.raises(ValueError, match="stale"):
        indexed.summarize("region", offset=1, limit=1, page_token=first["page_token"])


def test_tokens_require_same_filters_projection_and_catalog(indexed, session, tmp_path):
    first = indexed.search(regions=["CA1"], fields=["basepath"])
    with pytest.raises(ValueError, match="requires page_token"):
        indexed.search(offset=1)
    with pytest.raises(ValueError, match="stale"):
        indexed.search(
            regions=["PFC"], fields=["basepath"], page_token=first["page_token"]
        )
    with pytest.raises(ValueError, match="stale"):
        indexed.search(regions=["CA1"], page_token=first["page_token"])
    with pytest.raises(ValueError, match="requires page_token"):
        indexed.summarize(offset=1)
    with pytest.raises(ValueError, match="stale"):
        indexed.summarize(page_token=first["page_token"])
    csv = tmp_path / "catalog.csv"
    pd.DataFrame({"basepath": [str(session)]}).to_csv(csv, index=False)
    registered = Dataset(catalogs=[csv], index_db=indexed.path)
    unregistered = Dataset(index_db=indexed.path)
    page = registered.search_sessions(fields=["in_session_list"])
    assert page["rows"] == [{"in_session_list": True}]
    with pytest.raises(ValueError, match="stale"):
        unregistered.search_sessions(
            fields=["in_session_list"], page_token=page["page_token"]
        )
    page = registered.search_sessions(scope="all")
    with pytest.raises(ValueError, match="stale"):
        registered.search_sessions(scope="session_list", page_token=page["page_token"])


def test_each_query_uses_one_snapshot_during_concurrent_refresh(indexed, monkeypatch):
    original = indexed._rows

    def refresh_during_read(connection):
        for row in original(connection):
            with sqlite3.connect(indexed.path) as writer:
                writer.execute("DELETE FROM sessions")
                writer.execute("DELETE FROM memberships")
            yield row

    monkeypatch.setattr(indexed, "_rows", refresh_during_read)
    page = indexed.search()
    assert page["total_rows"] == page["coverage"]["session_count"] == 1
    assert indexed.status()["session_count"] == 0


def test_compact_fields_apply_before_response_limit_and_preserve_ids(
    indexed, session, tmp_path
):
    with sqlite3.connect(indexed.path) as connection:
        key, payload = connection.execute(
            "SELECT session_id,payload FROM sessions"
        ).fetchone()
        data = json.loads(payload)
        data["metadata"]["large_note"] = "x" * 128_000
        connection.execute(
            "UPDATE sessions SET payload=? WHERE session_id=?", (json.dumps(data), key)
        )
    with pytest.raises(ValueError, match="128 KB"):
        indexed.search()
    assert indexed.search(fields=["basepath"])["rows"] == [{"basepath": str(session)}]
    csv = tmp_path / "mapped.csv"
    pd.DataFrame({"basepath": [r"Z:\day01"]}).to_csv(csv, index=False)
    dataset = Dataset(
        catalogs=[csv], path_maps=[("Z:\\", str(session.parent))], index_db=indexed.path
    )
    full_id = dataset.list_sessions()["rows"][0]["session_id"]
    projected = dataset.search_sessions(
        fields=["session_id", "basepath", "in_session_list"], regions=["CA1"]
    )
    assert projected["rows"] == [
        {"session_id": full_id, "basepath": str(session), "in_session_list": True}
    ]
    assert dataset.search_sessions(fields=["basepath"])["rows"] == [
        {"basepath": str(session)}
    ]
    for fields in ([], ["unknown"], ["basepath", "unknown"]):
        with pytest.raises(ValueError, match="fields"):
            dataset.search_sessions(fields=fields)


def test_compact_paths_return_200_rows_and_complete_stable_pagination(tmp_path):
    cache = tmp_path / "private" / "index.sqlite"
    paths = [tmp_path / "recordings" / f"day{index:03}" for index in range(205)]
    IndexBuilder(cache).build(sessions=paths)
    dataset = Dataset(index_db=cache)
    first = dataset.search_sessions(fields=["basepath"], limit=200)
    second = dataset.search_sessions(
        fields=["basepath"],
        limit=200,
        offset=first["next_offset"],
        page_token=first["page_token"],
    )
    rows = first["rows"] + second["rows"]
    assert first["columns"] == ["basepath"]
    assert first["total_rows"] == second["total_rows"] == len(rows) == 205
    assert {row["basepath"] for row in rows} == {str(path) for path in paths}
    assert all(set(row) == {"basepath"} for row in rows)
    assert second["next_offset"] is None


def test_partial_region_matching_preserves_ca1_layer_labels(session, tmp_path):
    labels = ["ca1so", "ca1sp", "ca1sr", "ca1slm"]
    savemat(
        session / (session.name + ".cell_metrics.cellinfo.mat"),
        {
            "cell_metrics": {
                "UID": [1, 2, 3, 4],
                "brainRegion": np.asarray(labels, dtype=object),
                "putativeCellType": ["Pyramidal Cell"] * 4,
            }
        },
    )
    cache = tmp_path / "private" / "index.sqlite"
    IndexBuilder(cache).build(sessions=[session])
    result = MetadataIndex(cache).search(regions=["CA1"])
    assert result["total_rows"] == 1
    assert result["rows"][0]["matching_cell_count"] == 4
    assert {
        group["region"] for group in result["rows"][0]["metadata"]["cell_groups"]
    } == set(labels)


def test_queries_never_read_source_files(indexed, monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("Queries must not access recording files")

    monkeypatch.setattr("neuro_py.mcp.index._observe", no_network)
    assert indexed.status()["session_count"] == 1
    assert indexed.search(environments=["linear"])["total_rows"] == 1
    assert indexed.summarize("region")["total_rows"] == 2
    with indexed._connect() as connection:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            connection.execute("DELETE FROM sessions")


def test_resume_checkpoints_and_fair_root_progress(session, tmp_path):
    other = tmp_path / "rat02" / "day01"
    shutil.copytree(session, other)
    cache = tmp_path / "private" / "index.sqlite"
    roots = [session.parent, other.parent]
    status = IndexBuilder(cache).build(roots=roots, max_directories=2)
    assert not status["discovery_complete"]
    assert status["pending_directories"] == 2
    assert [root["directories_seen"] for root in status["roots"]] == [1, 1]
    # A fresh builder resumes the durable queue rather than restarting the roots.
    status = IndexBuilder(cache).build(roots=roots)
    assert status["discovery_complete"]
    assert status["session_count"] == 2
    assert [root["directories_seen"] for root in status["roots"]] == [2, 2]


def test_session_list_indexing_also_resumes(session, tmp_path):
    other = tmp_path / "rat02" / "day01"
    shutil.copytree(session, other)
    cache = tmp_path / "private" / "index.sqlite"
    first = IndexBuilder(cache).build(sessions=[session, other], max_directories=1)
    assert first["session_count"] == 1
    assert not first["discovery_complete"]
    second = IndexBuilder(cache).build(sessions=[session, other])
    assert second["session_count"] == 2
    assert second["discovery_complete"]
    assert second["roots"][0]["sessions_seen"] == 2


def test_small_batches_finish_all_roots_without_restarting_completed_roots(
    session, tmp_path
):
    other = tmp_path / "rat02" / "day01"
    shutil.copytree(session, other)
    cache = tmp_path / "private" / "index.sqlite"
    for _ in range(5):
        status = IndexBuilder(cache).build(
            roots=[session.parent, other.parent], max_directories=1
        )
        if status["discovery_complete"]:
            break
        if _ == 1:
            assert [root["directories_seen"] for root in status["roots"]] == [1, 1]
    assert status["discovery_complete"]
    assert status["session_count"] == 2
    assert [root["directories_seen"] for root in status["roots"]] == [2, 2]


def test_unchanged_sources_use_cache_and_changed_sources_refresh(
    indexed, session, monkeypatch
):
    from neuro_py.io import loading

    original = loading.load_epoch
    calls = []

    def track(basepath):
        calls.append(basepath)
        return original(basepath)

    monkeypatch.setattr(loading, "load_epoch", track)
    IndexBuilder(indexed.path).build(roots=[session.parent])
    assert calls == []
    savemat(
        session / "day01.session.mat",
        {
            "session": {
                "animal": {"name": "rat01"},
                "epochs": {
                    "name": "maze",
                    "environment": "cheeseboard",
                    "startTime": 0.0,
                    "stopTime": 2.0,
                },
            }
        },
    )
    IndexBuilder(indexed.path).build(roots=[session.parent])
    assert calls == [str(session)]
    assert indexed.search(environments=["cheeseboard"])["total_rows"] == 1
    assert indexed.search(environments=["linear"])["total_rows"] == 0


def test_missing_fields_large_sources_and_corrupt_files_are_not_false_absence(
    session, tmp_path
):
    cache = tmp_path / "private" / "index.sqlite"
    IndexBuilder(cache, max_source_bytes=1).build(roots=[session.parent])
    index = MetadataIndex(cache)
    assert index.status()["metadata_status_counts"] == {"partial": 1}
    row = index.search()["rows"][0]
    assert row["errors"]["cells"] == "source_exceeds_size_limit"
    assert index.summarize("region")["sessions_without_label"] == 1
    assert row["matching_cell_count"] is None
    # Changing the size limit invalidates the cache, even if sources did not change.
    IndexBuilder(cache).build(roots=[session.parent])
    assert index.search(regions=["CA1"])["total_rows"] == 1
    (session / "day01.cell_metrics.cellinfo.mat").write_bytes(b"not a MAT file")
    IndexBuilder(cache).build(roots=[session.parent])
    assert index.search(environments=["linear"])["total_rows"] == 1
    assert index.search()["rows"][0]["status"] == "partial"
    assert index.search()["rows"][0]["errors"]["cells"]
    assert index.search(regions=["CA1"])["total_rows"] == 0
    assert index.status()["metadata_status_counts"] == {"partial": 1}


def test_size_limit_changes_reuse_sources_with_unchanged_eligibility(
    indexed, session, monkeypatch
):
    def no_reload(*args, **kwargs):
        raise AssertionError("Unchanged eligible metadata should remain cached")

    monkeypatch.setattr("neuro_py.mcp.index._extract", no_reload)
    IndexBuilder(indexed.path, max_source_bytes=512_000_000).build(
        roots=[session.parent]
    )
    assert indexed.search(regions=["CA1"])["total_rows"] == 1
    assert indexed.status()["metadata_status_counts"] == {"ok": 1}


def test_inaccessible_roots_preserve_previous_discoveries(
    indexed, session, monkeypatch
):
    import os

    original = os.scandir

    def denied(path):
        if Path(path) == session.parent:
            raise PermissionError("Test share is offline")
        return original(path)

    monkeypatch.setattr("neuro_py.mcp.index.os.scandir", denied)
    status = IndexBuilder(indexed.path).build(roots=[session.parent])
    assert not status["discovery_complete"]
    assert status["session_count"] == 1
    assert status["roots"][0]["error_count"] == 1
    assert status["roots"][0]["status"] == "incomplete"
    assert indexed.search()["total_rows"] == 1


def test_successful_rescan_prunes_removed_discoveries(indexed, session):
    (session / "day01.session.mat").unlink()
    (session / "day01.cell_metrics.cellinfo.mat").unlink()
    status = IndexBuilder(indexed.path).build(roots=[session.parent])
    assert status["discovery_complete"]
    assert status["session_count"] == 0


def test_overlap_deduplicates_and_missing_list_sessions_are_explicit(session, tmp_path):
    cache = tmp_path / "private" / "index.sqlite"
    status = IndexBuilder(cache).build(
        roots=[session.parent, session], sessions=[session]
    )
    assert status["session_count"] == 1
    assert len(MetadataIndex(cache).search()["rows"][0]["roots"]) == 3
    missing = tmp_path / "offline"
    status = IndexBuilder(cache).build(sessions=[missing])
    assert status["metadata_status_counts"]["unavailable"] == 1


def test_extra_event_and_binary_files_are_inventoried_without_parsing(indexed, session):
    (session / "day01.lfp").write_bytes(b"not read by the indexer")
    (session / "day01.dentateSpikes.events.mat").write_bytes(b"not loaded")
    IndexBuilder(indexed.path).build(roots=[session.parent])
    assert indexed.search(products=["lfp"], event_types=["dentate"])["total_rows"] == 1


def test_cache_cannot_be_created_in_recording_folders(session):
    cache = session / "private-index.sqlite"
    with pytest.raises(ValueError, match="outside recording roots"):
        IndexBuilder(cache).build(roots=[session.parent])
    assert not cache.exists()
    with pytest.raises(ValueError, match="outside recording roots"):
        IndexBuilder(cache).build(sessions=[session])
    assert not cache.exists()


def test_recycling_snapshots_and_extra_excluded_dirs_are_not_discovered(
    session, tmp_path
):
    root = tmp_path / "share"
    root.mkdir()
    for name in ["#recycle", "#snapshot", "@eaDir", ".phy", "custom_archive"]:
        shutil.copytree(session, root / name / "day01")
    shutil.copytree(session, root / "rat01" / "day01")
    cache = tmp_path / "private" / "index.sqlite"
    status = IndexBuilder(cache, exclude_dirs=["custom_archive"]).build(roots=[root])
    assert status["session_count"] == 1
    assert status["discovery_complete"]
    assert "#recycle" in status["discovery_policy"]["excluded_directory_names"]


def test_new_exclusions_remove_queued_descendants_without_network_reads(
    session, tmp_path, monkeypatch
):
    root = session.parent
    cache = tmp_path / "private" / "index.sqlite"
    IndexBuilder(cache).build(roots=[root], max_directories=1)
    # Simulate an old queue that had already descended into a large cache.
    with sqlite3.connect(cache) as connection:
        for path in [root / "rat02" / ".PHY" / "a", root / "cache" / "nested"]:
            connection.execute(
                "INSERT INTO pending VALUES (?,?)", (str(root), str(path))
            )
        connection.execute(
            "UPDATE settings SET value=? WHERE key='discovery_policy'",
            (json.dumps({"excluded_directory_names": []}),),
        )
    import os

    original = os.scandir

    def no_cache_reads(path):
        assert ".phy" not in str(path).casefold()
        assert "cache" not in Path(path).parts
        return original(path)

    monkeypatch.setattr("neuro_py.mcp.index.os.scandir", no_cache_reads)
    status = IndexBuilder(cache, exclude_dirs=["cache"]).build(roots=[root])
    assert status["discovery_complete"]
    assert status["session_count"] == 1
    assert status["pending_directories"] == 0


def test_exclusions_preserve_explicit_session_list_entries(session, tmp_path):
    cache = tmp_path / "private" / "index.sqlite"
    builder = IndexBuilder(cache, exclude_dirs=[session.name])
    # Keep a pending explicit registration while changing the exclusion policy.
    builder.build(sessions=[session, tmp_path / "offline"], max_directories=1)
    status = IndexBuilder(cache, exclude_dirs=["offline"]).build(
        sessions=[session, tmp_path / "offline"]
    )
    assert status["session_count"] == 2
    assert status["metadata_status_counts"]["unavailable"] == 1


def test_second_writer_is_rejected_without_changing_the_index(indexed, session):
    builder = IndexBuilder(indexed.path)
    with builder._writer_lock():
        with pytest.raises(OSError):
            IndexBuilder(indexed.path).build(roots=[session.parent])
        assert indexed.status()["session_count"] == 1


def test_invalid_queries_disabled_index_and_bad_schema(indexed, tmp_path):
    with pytest.raises(ValueError, match="Unknown products"):
        indexed.search(products=["../../private"])
    with pytest.raises(ValueError, match="facet"):
        indexed.summarize("not_a_facet")
    with pytest.raises(ValueError, match="offset"):
        indexed.search(limit=201)
    dataset = Dataset(index_db=indexed.path)
    with pytest.raises(ValueError, match="scope"):
        dataset.search_sessions(scope="invalid")
    plain = Dataset(sessions=[tmp_path / "example"])
    assert plain.index_status()["status"] == "disabled"
    with pytest.raises(ValueError, match="No index"):
        plain.search_sessions()
    invalid = tmp_path / "invalid.sqlite"
    with sqlite3.connect(invalid) as connection:
        connection.execute("PRAGMA user_version=1000")
    with pytest.raises(ValueError, match="schema"):
        MetadataIndex(invalid)


def test_index_config_cli(session, tmp_path):
    import subprocess
    import sys

    cache = tmp_path / "private" / "index.sqlite"
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({"index_db": str(cache), "roots": [str(session.parent)]})
    )
    result = subprocess.run(
        [sys.executable, "-m", "neuro_py.mcp.index", "--config", str(config)],
        capture_output=True,
        text=True,
        timeout=40,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["session_count"] == 1
