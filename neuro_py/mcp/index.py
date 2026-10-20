"""Resumable metadata discovery with a private local SQLite cache.

The builder writes only the configured cache; MCP consumers open it read-only.
Session data is interpreted by the existing neuro_py loaders.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import stat
import sys
import time
import uuid
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from neuro_py.mcp.dataset import SOURCES, _bounded, _identity, _json_safe, _page

SCHEMA_VERSION = 1
EXCLUDED_DIRS = {
    ".git",
    ".venv",
    ".phy",
    "__pycache__",
    "#recycle",
    "#snapshot",
    "@eadir",
    "$recycle.bin",
    "system volume information",
    ".snapshot",
    ".snapshots",
}
INDEX_FILES = {key: suffix for key, suffix in SOURCES.items() if key != "trials"}
INDEX_FILES.update(
    {
        "behavior": ".animal.behavior.mat",
        "spikes": ".spikes.cellinfo.mat",
        "lfp": ".lfp",
        "eeg": ".eeg",
        "xml": ".xml",
    }
)
FACETS = {
    "environment",
    "region",
    "recording_region",
    "cell_type",
    "animal",
    "product",
    "event_type",
    "root",
    "status",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _labels(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, np.ndarray):
        values = values.reshape(-1).tolist()
    # Scalar strings and missing values occur in legacy MATLAB metadata.
    items = values if isinstance(values, (list, tuple, pd.Series)) else [values]
    return sorted(
        {str(item) for item in items if isinstance(item, str) and item.strip()}
    )


def _matches(labels: list[str], text: str) -> bool:
    return any(text.casefold() in label.casefold() for label in labels)


def _observe(path: Path) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for product, suffix in INDEX_FILES.items():
        source = path / (path.name + suffix)
        try:
            info = source.stat(follow_symlinks=False)
            files[product] = {
                "present": stat.S_ISREG(info.st_mode),
                "size_bytes": info.st_size,
                "mtime_ns": info.st_mtime_ns,
            }
        except FileNotFoundError:
            files[product] = {"present": False}
    with os.scandir(path) as entries:
        events = sorted(
            {
                entry.name[len(path.name) + 1 : -len(".events.mat")]
                for entry in entries
                if entry.name.startswith(path.name + ".")
                and entry.name.endswith(".events.mat")
                and entry.is_file(follow_symlinks=False)
            }
        )
    return {"files": files, "event_types": events}


def _extract(
    path: Path, observed: dict[str, Any], max_source_bytes: int
) -> dict[str, Any]:
    from neuro_py.io import loading
    from neuro_py.io.matlab import load_mat

    metadata: dict[str, Any] = {"environments": [], "animals": [], "cell_groups": []}
    errors: dict[str, str] = {}
    for product in ("epochs", "cells"):
        source = observed["files"][product]
        if not source["present"]:
            errors[product] = "missing_file"
            continue
        if source["size_bytes"] > max_source_bytes:
            errors[product] = "source_exceeds_size_limit"
            continue
        try:
            if product == "epochs":
                epochs = loading.load_epoch(str(path))
                metadata["environments"] = _labels(epochs.get("environment"))
                metadata["epoch_count"] = len(epochs)
                session = load_mat(path / (path.name + ".session.mat"))["session"]
                brain_regions = session.get("brainRegions", {})
                if isinstance(brain_regions, dict):
                    metadata["recording_regions"] = sorted(brain_regions)
                extracellular = session.get("extracellular", {})
                if isinstance(extracellular, dict):
                    metadata["recording"] = {
                        key: extracellular[key]
                        for key in ("sr", "srLfp", "nChannels")
                        if key in extracellular
                    }
                animal = session.get("animal", {})
                if isinstance(animal, dict):
                    metadata["animals"] = _labels(animal.get("name"))
            else:
                cells = loading.load_cell_metrics(str(path), only_metrics=True)
                if (
                    not isinstance(cells, pd.DataFrame)
                    or "UID" not in cells
                    or "bad_unit" not in cells
                ):
                    raise ValueError("Cell loader did not supply UID and bad_unit.")
                metadata["cell_count_total"] = len(cells)
                metadata["cell_count_good"] = int((~cells.bad_unit.astype(bool)).sum())
                groups: Counter[tuple[str, str, bool]] = Counter()
                group_columns = [
                    name
                    for name in ("brainRegion", "putativeCellType", "bad_unit")
                    if name in cells
                ]
                for row in cells.loc[:, group_columns].to_dict(orient="records"):
                    region = row.get("brainRegion")
                    cell_type = row.get("putativeCellType")
                    groups[
                        (
                            region if isinstance(region, str) else "",
                            cell_type if isinstance(cell_type, str) else "",
                            bool(row["bad_unit"]),
                        )
                    ] += 1
                metadata["cell_groups"] = [
                    {
                        "region": region,
                        "cell_type": cell_type,
                        "bad_unit": bad,
                        "count": count,
                    }
                    for (region, cell_type, bad), count in sorted(groups.items())
                ]
        except Exception as error:
            # One malformed session must not abort discovery of the other shares.
            errors[product] = f"{type(error).__name__}: {error}"[:1000]
    return {
        "metadata": metadata,
        "errors": errors,
        "status": "partial" if errors else "ok",
    }


class MetadataIndex:
    """Query a local metadata snapshot without opening recording files.

    Parameters
    ----------
    path : pathlib.Path
        Existing private SQLite index created by :class:`IndexBuilder`.
    """

    def __init__(self, path: Path) -> None:
        self.path = path.resolve(strict=True)
        with self._connect() as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version != SCHEMA_VERSION:
                raise ValueError(
                    "Unsupported metadata index schema. Rebuild the index."
                )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _rows(self) -> Iterator[dict[str, Any]]:
        with self._connect() as connection:
            memberships: dict[str, list[str]] = {}
            for item in connection.execute("SELECT session_id, root FROM memberships"):
                memberships.setdefault(item["session_id"], []).append(item["root"])
            for row in connection.execute("SELECT * FROM sessions ORDER BY basepath"):
                data = json.loads(row["payload"])
                data.update(
                    {
                        "session_id": row["session_id"],
                        "basepath": row["basepath"],
                        "roots": sorted(memberships.get(row["session_id"], [])),
                    }
                )
                yield data

    def get_entry(self, session_id: str) -> dict[str, Any] | None:
        """Resolve an indexed session ID to a registered session folder.

        Parameters
        ----------
        session_id : str
            ID returned by an index query.

        Returns
        -------
        dict or None
            Session registration or None for an unknown ID.
        """
        with self._connect() as connection:
            row = connection.execute(
                "SELECT basepath FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        if row is None:
            return None
        return {
            "session_id": session_id,
            "catalog_basepath": row[0],
            "basepath": row[0],
            "basename": Path(row[0]).name,
            "registrations": [{"source": str(self.path), "metadata": {}}],
        }

    def status(self) -> dict[str, Any]:
        """Report cached discovery coverage and resumable scan progress.

        Returns
        -------
        dict
            Per-root progress, metadata completeness, and last observation times.
        """
        with self._connect() as connection:
            roots = [
                dict(row)
                for row in connection.execute("SELECT * FROM roots ORDER BY root")
            ]
            total = connection.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            statuses = dict(
                connection.execute(
                    "SELECT json_extract(payload, '$.status'), COUNT(*) "
                    "FROM sessions GROUP BY 1"
                ).fetchall()
            )
            pending = connection.execute("SELECT COUNT(*) FROM pending").fetchone()[0]
            settings_exist = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='settings'"
            ).fetchone()
            policy = (
                connection.execute(
                    "SELECT value FROM settings WHERE key='discovery_policy'"
                ).fetchone()
                if settings_exist
                else None
            )
        for root in roots:
            root["errors"] = json.loads(root["errors"])
            root.pop("generation", None)
        return {
            "status": "enabled",
            "index_path": str(self.path),
            "session_count": total,
            "cached_not_live": True,
            "discovery_policy": json.loads(policy[0]) if policy else {},
            "pending_directories": pending,
            "discovery_complete": bool(roots)
            and all(
                root["status"] == "complete" and root["error_count"] == 0
                for root in roots
            ),
            "metadata_status_counts": statuses,
            "roots": roots,
        }

    def search(
        self,
        query: str = "",
        environments: list[str] | None = None,
        regions: list[str] | None = None,
        cell_type: str = "",
        products: list[str] | None = None,
        event_types: list[str] | None = None,
        include_bad_units: bool = False,
        paths: set[str] | None = None,
        offset: int = 0,
        limit: int = 50,
        recording_regions: list[str] | None = None,
    ) -> dict[str, Any]:
        """Find sessions using cached metadata and AND-combined literal filters.

        Parameters
        ----------
        query : str, optional
            Literal substring of path, animal name, or metadata labels.
        environments, regions, products, event_types : list of str, optional
            Every requested value must match. Product names are exact keys;
            other filters are case-insensitive literal substrings.
        cell_type : str, optional
            Required cell-type substring within every requested region.
        include_bad_units : bool, optional
            Count tagged bad cells for region/type matching; default False.
        paths : set of str, optional
            Normalized session-list paths to restrict scope, or None for all.
        recording_regions : list of str, optional
            Recorded region labels from session.brainRegions, independent of
            whether sorted cells exist or their metadata could be indexed.
        offset, limit : int, optional
            Page offset and row limit (1..200).

        Returns
        -------
        dict
            Matching sessions, filtered total, coverage, and pagination.
        """
        _page(pd.DataFrame(), offset, limit)
        unknown = set(products or []) - set(INDEX_FILES)
        if unknown:
            raise ValueError(f"Unknown products: {sorted(unknown)}")
        found = []
        total = 0
        for row in self._rows():
            if paths is not None and _identity(row["basepath"]) not in paths:
                continue
            metadata = row.get("metadata", {})
            if any(
                not _matches(metadata.get("recording_regions", []), term)
                for term in recording_regions or []
            ):
                continue
            groups = [
                g
                for g in metadata.get("cell_groups", [])
                if include_bad_units or not g["bad_unit"]
            ]
            if cell_type:
                groups = [g for g in groups if _matches([g["cell_type"]], cell_type)]
            if any(
                not _matches(metadata.get("environments", []), term)
                for term in environments or []
            ):
                continue
            if any(
                not _matches([g["region"] for g in groups], term)
                for term in regions or []
            ):
                continue
            if cell_type and not groups:
                continue
            if any(
                not row.get("files", {}).get(p, {}).get("present", False)
                for p in products or []
            ):
                continue
            if any(
                not _matches(row.get("event_types", []), term)
                for term in event_types or []
            ):
                continue
            text = "\n".join(
                [
                    row["basepath"],
                    *metadata.get("animals", []),
                    *metadata.get("environments", []),
                    *[g["region"] for g in groups],
                    *[g["cell_type"] for g in groups],
                ]
            )
            if query.casefold() not in text.casefold():
                continue
            selected = [
                g
                for g in groups
                if not regions or any(_matches([g["region"]], term) for term in regions)
            ]
            row["matching_cell_count"] = (
                sum(g["count"] for g in selected)
                if "cell_count_total" in metadata
                else None
            )
            if offset <= total < offset + limit:
                found.append(row)
            total += 1
        return _bounded(
            {
                "status": "ok",
                "coverage": self.status(),
                "total_rows": total,
                "offset": offset,
                "limit": limit,
                "next_offset": offset + limit if offset + limit < total else None,
                "columns": list(dict.fromkeys(key for row in found for key in row)),
                "rows": found,
            }
        )

    def summarize(
        self,
        facet: str = "environment",
        paths: set[str] | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Count sessions by a metadata category without accessing the shares.

        Parameters
        ----------
        facet : str, optional
            environment, region, recording_region, cell_type, animal, product,
            event_type, root, or status. Region/type counts exclude bad units.
        paths : set of str, optional
            Restrict to normalized session-list paths, or None for all sessions.
        offset, limit : int, optional
            Page offset and row limit (1..200).

        Returns
        -------
        dict
            Session counts per source label, unknown count, and scan coverage.
        """
        if facet not in FACETS:
            raise ValueError(f"facet must be one of {sorted(FACETS)}")
        counts: Counter[str] = Counter()
        unknown = total = 0
        for row in self._rows():
            if paths is not None and _identity(row["basepath"]) not in paths:
                continue
            total += 1
            metadata = row.get("metadata", {})
            if facet in {"region", "cell_type"}:
                labels = {
                    g[facet]
                    for g in metadata.get("cell_groups", [])
                    if not g["bad_unit"] and g[facet]
                }
            elif facet == "product":
                labels = {
                    key
                    for key, value in row.get("files", {}).items()
                    if value["present"]
                }
            elif facet == "recording_region":
                labels = set(metadata.get("recording_regions", []))
            elif facet == "status":
                labels = {row["status"]}
            elif facet in {"root", "event_type"}:
                labels = set(row.get("roots" if facet == "root" else "event_types", []))
            else:
                labels = set(
                    metadata.get(
                        "environments" if facet == "environment" else "animals", []
                    )
                )
            counts.update(labels)
            unknown += not bool(labels)
        frame = pd.DataFrame(
            [
                {"label": label, "session_count": count}
                for label, count in sorted(counts.items())
            ],
            columns=pd.Index(["label", "session_count"]),
        )
        return _bounded(
            {
                "status": "ok",
                "facet": facet,
                "sessions_in_scope": total,
                "sessions_without_label": unknown,
                "coverage": self.status(),
                **_page(frame, offset, limit),
            }
        )


class IndexBuilder:
    """Build an incremental metadata cache, with resumable directory discovery.

    Parameters
    ----------
    path : pathlib.Path
        Private local SQLite output. Never place it in a recording directory.
    max_source_bytes : int, optional
        Maximum size of each MAT source to load, default 128 MB. Larger sources
        remain discoverable with explicit incomplete-metadata status.
    exclude_dirs : list of str, optional
        Additional directory names to skip, matched case-insensitively. System
        recycling, snapshot, metadata-cache, and development folders are skipped.
    """

    def __init__(
        self,
        path: Path,
        max_source_bytes: int = 128_000_000,
        exclude_dirs: list[str] | None = None,
    ) -> None:
        if max_source_bytes < 1:
            raise ValueError("max_source_bytes must be positive")
        self.path = path.resolve()
        self.max_source_bytes = max_source_bytes
        self.exclude_dirs = EXCLUDED_DIRS | {
            name.casefold() for name in exclude_dirs or []
        }

    def _initialize(self) -> None:
        with self._connect() as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version not in {0, SCHEMA_VERSION}:
                raise ValueError("Unsupported metadata index schema")
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY, basepath TEXT NOT NULL,
                    fingerprint TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS memberships (
                    session_id TEXT NOT NULL, root TEXT NOT NULL,
                    generation TEXT NOT NULL,
                    PRIMARY KEY (session_id, root));
                CREATE TABLE IF NOT EXISTS roots (
                    root TEXT PRIMARY KEY, generation TEXT NOT NULL,
                    status TEXT NOT NULL, started_at TEXT NOT NULL, finished_at TEXT,
                    directories_seen INTEGER NOT NULL, sessions_seen INTEGER NOT NULL,
                    error_count INTEGER NOT NULL, errors TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS pending (
                    root TEXT NOT NULL, path TEXT NOT NULL, PRIMARY KEY (root, path));
                CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT);
                CREATE INDEX IF NOT EXISTS pending_roots ON pending(root);
                CREATE INDEX IF NOT EXISTS session_paths ON sessions(basepath);
                CREATE INDEX IF NOT EXISTS membership_roots
                    ON memberships(root, generation);
            """)
            connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

        if os.name != "nt":
            self.path.chmod(0o600)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @contextmanager
    def _writer_lock(self) -> Iterator[None]:
        # An OS lock survives commits and is released automatically on a crash.
        with open(str(self.path) + ".lock", "a+b") as handle:
            handle.seek(0)
            if not handle.read(1):
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                yield
            finally:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _begin(self, connection: sqlite3.Connection, root: str, restart: bool) -> str:
        row = connection.execute("SELECT * FROM roots WHERE root=?", (root,)).fetchone()
        if row is not None and row["status"] == "scanning" and not restart:
            return row["generation"]
        generation = uuid.uuid4().hex
        connection.execute("DELETE FROM pending WHERE root=?", (root,))
        connection.execute(
            "INSERT OR REPLACE INTO roots VALUES (?,?,'scanning',?,NULL,0,0,0,'[]')",
            (root, generation, _now()),
        )
        return generation

    def _error(self, connection: sqlite3.Connection, root: str, error: str) -> None:
        row = connection.execute(
            "SELECT errors FROM roots WHERE root=?", (root,)
        ).fetchone()
        errors = json.loads(row[0])
        if len(errors) < 5:
            errors.append(error[:1000])
        connection.execute(
            "UPDATE roots SET errors=?, error_count=error_count+1 WHERE root=?",
            (json.dumps(errors), root),
        )

    def _prune_pending(self, connection: sqlite3.Connection) -> None:
        # A resumed queue can contain millions of descendants of a newly
        # excluded cache. Remove them locally without visiting the share.
        prior = connection.execute(
            "SELECT value FROM settings WHERE key='discovery_policy'"
        ).fetchone()
        previous = (
            set(json.loads(prior[0]).get("excluded_directory_names", []))
            if prior
            else set()
        )
        added = self.exclude_dirs - previous
        if not added:
            return

        def excluded(root: str, path: str) -> bool:
            if root == "session-list":
                return False
            relative = path[len(root) :].replace("\\", "/").casefold()
            return bool(added.intersection(relative.split("/")))

        connection.create_function("excluded_pending", 2, excluded, deterministic=True)
        # Rebuild the queue in its original order rather than individually
        # deleting millions of rows and updating both indexes for every row.
        connection.execute(
            "CREATE TEMP TABLE retained_pending AS "
            "SELECT rowid AS queue_order, root, path FROM pending "
            "WHERE NOT excluded_pending(root, path)"
        )
        connection.execute("DELETE FROM pending")
        connection.execute(
            "INSERT INTO pending(rowid, root, path) "
            "SELECT queue_order, root, path FROM retained_pending ORDER BY queue_order"
        )
        connection.execute("DROP TABLE retained_pending")

    def _session(
        self, connection: sqlite3.Connection, path: Path, root: str, generation: str
    ) -> None:
        key = _identity(str(path))
        session_id = hashlib.sha256(key.encode()).hexdigest()[:20]
        prior = connection.execute(
            "SELECT * FROM sessions WHERE session_id=?", (session_id,)
        ).fetchone()
        try:
            observed = _observe(path)
            fingerprint = json.dumps(
                {"observed": observed, "max_source_bytes": self.max_source_bytes},
                sort_keys=True,
            )
            if (
                prior
                and prior["fingerprint"] == fingerprint
                and all(
                    value in {"missing_file", "source_exceeds_size_limit"}
                    for value in json.loads(prior["payload"])["errors"].values()
                )
            ):
                payload = json.loads(prior["payload"])
            else:
                payload = {
                    **observed,
                    **_extract(path, observed, self.max_source_bytes),
                }
                if _observe(path) != observed:
                    payload.update(
                        {
                            "metadata": {},
                            "status": "error",
                            "errors": {
                                "session": "Sources changed while indexing; retry."
                            },
                        }
                    )
            payload["observed_at"] = _now()
        except Exception as error:
            fingerprint = "unavailable"
            payload = {
                "status": "unavailable",
                "metadata": {},
                "files": {},
                "errors": {"session": f"{type(error).__name__}: {error}"[:1000]},
                "observed_at": _now(),
            }
        connection.execute(
            "INSERT OR REPLACE INTO sessions VALUES (?,?,?,?)",
            (
                session_id,
                str(path),
                fingerprint,
                json.dumps(_json_safe(payload), allow_nan=False),
            ),
        )
        connection.execute(
            "INSERT OR REPLACE INTO memberships VALUES (?,?,?)",
            (session_id, root, generation),
        )
        connection.execute(
            "UPDATE roots SET sessions_seen=sessions_seen+1 WHERE root=?", (root,)
        )

    def _finish(
        self, connection: sqlite3.Connection, root: str, generation: str
    ) -> None:
        # Prune only after a complete, error-free traversal. An offline subtree
        # must never be treated as evidence that its recordings were deleted.
        errors = connection.execute(
            "SELECT error_count FROM roots WHERE root=?", (root,)
        ).fetchone()[0]
        if errors == 0:
            connection.execute(
                "DELETE FROM memberships WHERE root=? AND generation<>?",
                (root, generation),
            )
            connection.execute(
                "DELETE FROM sessions WHERE session_id NOT IN "
                "(SELECT session_id FROM memberships)"
            )
        connection.execute(
            "UPDATE roots SET status=?, finished_at=? WHERE root=?",
            ("complete" if errors == 0 else "incomplete", _now(), root),
        )

    def build(
        self,
        roots: list[Path] | None = None,
        sessions: list[Path] | None = None,
        max_directories: int | None = None,
        max_seconds: float | None = None,
        restart: bool = False,
    ) -> dict[str, Any]:
        """Discover configured roots fairly, checkpointing each scanned directory.

        Parameters
        ----------
        roots : list of pathlib.Path, optional
            Approved recording roots. Links and junctions are not traversed.
        sessions : list of pathlib.Path, optional
            Session-list folders to index alongside the broader discovery roots.
        max_directories : int, optional
            Stop after this many directories in this invocation; resume next run.
        max_seconds : float, optional
            Stop between directories after this duration. An individual network
            operation or loader call can exceed the budget.
        restart : bool, optional
            Restart discovery from the roots, keeping cached session metadata.

        Returns
        -------
        dict
            Coverage status with durable pending directories and per-root errors.
        """
        if max_directories is not None and max_directories < 1:
            raise ValueError("max_directories must be positive")
        if max_seconds is not None and max_seconds <= 0:
            raise ValueError("max_seconds must be positive")
        roots = [Path(os.path.abspath(root)) for root in roots or []]
        if not roots and not sessions:
            raise ValueError("Supply discovery roots or session folders")
        for root in [*roots, *(sessions or [])]:
            root = Path(os.path.abspath(root))
            if self.path.is_relative_to(root):
                raise ValueError("The index cache must be outside recording roots")
        started = time.monotonic()
        scanned = 0
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self._writer_lock():
            self._initialize()
            with self._connect() as connection:
                self._prune_pending(connection)
                connection.execute(
                    "INSERT OR REPLACE INTO settings VALUES (?,?)",
                    (
                        "discovery_policy",
                        json.dumps(
                            {
                                "excluded_directory_names": sorted(self.exclude_dirs),
                                "links_followed": False,
                                "max_source_bytes": self.max_source_bytes,
                            }
                        ),
                    ),
                )
                seeds = {str(root): [root] for root in roots}
                if sessions:
                    seeds["session-list"] = list(
                        dict.fromkeys(Path(os.path.abspath(path)) for path in sessions)
                    )
                generations = {}
                # Finish the current multi-root pass before refreshing roots
                # already completed in that pass. Otherwise a small budget
                # could repeatedly restart the first root and starve the rest.
                resuming_pass = not restart and any(
                    connection.execute(
                        "SELECT 1 FROM roots WHERE root=? AND status='scanning'",
                        (root,),
                    ).fetchone()
                    for root in seeds
                )
                ordered_seeds = list(seeds.items())
                if resuming_pass:

                    def progress(item: tuple[str, list[Path]]) -> int:
                        row = connection.execute(
                            "SELECT directories_seen FROM roots WHERE root=?",
                            (item[0],),
                        ).fetchone()
                        return row[0] if row else 0

                    ordered_seeds.sort(key=progress)
                for root, paths in ordered_seeds:
                    prior = connection.execute(
                        "SELECT * FROM roots WHERE root=?", (root,)
                    ).fetchone()
                    if (
                        resuming_pass
                        and prior is not None
                        and prior["status"] != "scanning"
                    ):
                        continue
                    resuming = (
                        prior is not None
                        and prior["status"] == "scanning"
                        and not restart
                    )
                    generation = self._begin(connection, root, restart)
                    generations[root] = generation
                    if not resuming:
                        connection.executemany(
                            "INSERT OR IGNORE INTO pending VALUES (?,?)",
                            [(root, str(path)) for path in paths],
                        )
                connection.commit()
                while generations:
                    for root, generation in list(generations.items()):
                        item = connection.execute(
                            "SELECT path FROM pending WHERE root=? "
                            "ORDER BY rowid LIMIT 1",
                            (root,),
                        ).fetchone()
                        if item is None:
                            self._finish(connection, root, generation)
                            del generations[root]
                            connection.commit()
                            continue
                        if (
                            max_directories is not None and scanned >= max_directories
                        ) or (
                            max_seconds is not None
                            and time.monotonic() - started >= max_seconds
                        ):
                            return MetadataIndex(self.path).status()
                        path = Path(item[0])
                        if root == "session-list":
                            self._session(connection, path, root, generation)
                        else:
                            self._directory(connection, path, root, generation)
                        connection.execute(
                            "DELETE FROM pending WHERE root=? AND path=?",
                            (root, str(path)),
                        )
                        connection.execute(
                            "UPDATE roots SET directories_seen=directories_seen+1 "
                            "WHERE root=?",
                            (root,),
                        )
                        # Commit each directory so interruptions preserve both
                        # metadata and the unvisited subtree queue.
                        connection.commit()
                        scanned += 1
                        if scanned % 25 == 0:
                            print(
                                f"Discovery progress: {scanned} directories this run",
                                file=sys.stderr,
                                flush=True,
                            )
        return MetadataIndex(self.path).status()

    def _directory(
        self,
        connection: sqlite3.Connection,
        path: Path,
        root: str,
        generation: str,
    ) -> None:
        if str(path) != root and path.name.casefold() in self.exclude_dirs:
            return
        try:
            filenames = set()
            children = []
            with os.scandir(path) as entries:
                for entry in entries:
                    if entry.is_symlink():
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        info = entry.stat(follow_symlinks=False)
                        if getattr(info, "st_file_attributes", 0) & getattr(
                            stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400
                        ):
                            continue
                        if entry.name.casefold() not in self.exclude_dirs:
                            children.append((root, entry.path))
                    else:
                        filenames.add(os.path.normcase(entry.name))
            if any(
                os.path.normcase(path.name + suffix) in filenames
                for suffix in (SOURCES["epochs"], SOURCES["cells"])
            ):
                self._session(connection, path, root, generation)
            connection.executemany(
                "INSERT OR IGNORE INTO pending VALUES (?,?)", children
            )
        except OSError as error:
            self._error(connection, root, f"{path}: {type(error).__name__}: {error}")


def main() -> None:
    """Build a private index from a JSON configuration or explicit CLI roots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        help="Private JSON with index_db, roots, catalogs, path_maps",
    )
    parser.add_argument("--index-db", type=Path)
    parser.add_argument("--root", type=Path, action="append", default=[])
    parser.add_argument("--catalog", type=Path, action="append", default=[])
    parser.add_argument("--max-directories", type=int)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--max-source-bytes", type=int, default=128_000_000)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text()) if args.config else {}
    index_path = args.index_db or (
        Path(config["index_db"]) if "index_db" in config else None
    )
    if index_path is None:
        parser.error("Supply --index-db or index_db in --config")
    roots = [*args.root, *[Path(value) for value in config.get("roots", [])]]
    catalogs = [*args.catalog, *[Path(value) for value in config.get("catalogs", [])]]
    sessions = []
    if catalogs:
        from neuro_py.mcp.dataset import Dataset

        dataset = Dataset(
            catalogs=catalogs,
            path_maps=[tuple(pair) for pair in config.get("path_maps", [])],
        )
        sessions = [Path(entry["basepath"]) for entry in dataset._sessions.values()]
    result = IndexBuilder(
        index_path, args.max_source_bytes, exclude_dirs=config.get("exclude_dirs", [])
    ).build(roots, sessions, args.max_directories, args.max_seconds, args.restart)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
