"""POSIX path identity and physical cache containment."""

import os

import pytest

from neuro_py.mcp.dataset import Dataset
from neuro_py.mcp.index import IndexBuilder

pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX symbolic-link cases")


@pytest.mark.parametrize(
    "registration", ["relative_catalog", "absolute_catalog", "session", "root"]
)
def test_symlinked_registrations_match_index_scope(session, tmp_path, registration):
    alias = tmp_path / "mounted"
    alias.symlink_to(session.parent, target_is_directory=True)
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
    tmp_path, registration, cache_prefix
):
    recordings = tmp_path / "recordings"
    (recordings / "day01").mkdir(parents=True)
    alias = tmp_path / "mounted"
    alias.symlink_to(recordings, target_is_directory=True)
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
