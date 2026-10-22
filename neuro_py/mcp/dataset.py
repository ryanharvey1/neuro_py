"""Catalog registration and bounded reads using the neuro_py data loaders."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import ntpath
import os
from pathlib import Path, PureWindowsPath
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from neuro_py.mcp.index import MetadataIndex

MAX_ROWS = 200
MAX_RESPONSE_BYTES = 128_000
SEARCH_FIELDS = {
    "session_id",
    "basepath",
    "roots",
    "status",
    "metadata",
    "errors",
    "files",
    "event_types",
    "observed_at",
    "matching_cell_count",
    "in_session_list",
}
SOURCES = {
    "epochs": ".session.mat",
    "cells": ".cell_metrics.cellinfo.mat",
    "trials": ".animal.behavior.mat",
    "ripples": ".ripples.events.mat",
    "sleep_states": ".SleepState.states.mat",
}
DEFAULT_CELL_COLUMNS = (
    "UID",
    "brainRegion",
    "putativeCellType",
    "bad_unit",
    "firingRate",
    "deepSuperficial",
    "deepSuperficialDistance",
    "basename",
    "basepath",
    "sex",
    "species",
    "strain",
    "geneticLine",
)
CONVENTIONS = {
    "time": "Seconds relative to the session recording, as stored by neuro_py.",
    "cell_identity": "Use (session_id, UID); UID is session-local, not a row index.",
    "region_labels": "Preserve source labels; CA1sp, CA1so, etc. are not renamed.",
    "filters": "Case-insensitive literal substring matches, never regex.",
    "epochs": "Original session epochs; no automatic pre/task/post selection.",
    "trials": "Original neuro_py trial labels; no task-specific inference.",
    "missing_data": "Missing files, empty tables, and offline sessions differ.",
    "catalog": "Catalog metadata may be stale; it is not a recording-file audit.",
    "read_only": "No file writes, arbitrary code execution, or analysis jobs.",
}


def _identity(path: str) -> str:
    if PureWindowsPath(path).drive or "\\" in path:
        return ntpath.normcase(ntpath.normpath(path))
    return os.path.normpath(path)


def _recording_path(path: Path) -> Path:
    """Keep parent aliases, but use the target basename for a renamed session link."""
    resolved = path.resolve()
    if os.path.normcase(path.name) != os.path.normcase(resolved.name):
        return resolved
    return path


def _json_safe(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _bounded(result: dict[str, Any]) -> dict[str, Any]:
    result = _json_safe(result)
    if len(json.dumps(result, allow_nan=False).encode("utf-8")) > MAX_RESPONSE_BYTES:
        raise ValueError("Response exceeds 128 KB. Request fewer rows or columns.")
    return result


def _page(frame: pd.DataFrame, offset: int, limit: int) -> dict[str, Any]:
    if offset < 0 or not 1 <= limit <= MAX_ROWS:
        raise ValueError(f"offset must be >= 0 and limit must be 1..{MAX_ROWS}.")
    stop = offset + limit
    return {
        "total_rows": len(frame),
        "offset": offset,
        "limit": limit,
        "next_offset": stop if stop < len(frame) else None,
        "columns": list(frame.columns),
        "rows": frame.iloc[offset:stop].to_dict(orient="records"),
    }


def _contains(series: pd.Series, text: str) -> pd.Series:
    return series.astype("string").str.contains(text, case=False, regex=False, na=False)


def _search_fields(fields: list[str] | None) -> list[str] | None:
    if fields is None:
        return None
    if not fields or set(fields) - SEARCH_FIELDS:
        raise ValueError(
            f"fields must be a nonempty selection from {sorted(SEARCH_FIELDS)}"
        )
    return list(dict.fromkeys(fields))


def _search_values(value: Any) -> str:
    # Search source text rather than JSON-escaped Windows paths or Unicode.
    if isinstance(value, dict):
        return "\n".join(_search_values(item) for item in value.values())
    if isinstance(value, list):
        return "\n".join(_search_values(item) for item in value)
    return str(value)


class Dataset:
    """Register sessions at startup and expose only their known data products.

    Parameters
    ----------
    catalogs : list of pathlib.Path, optional
        CSV files containing a ``basepath`` column. Relative entries resolve
        against the CSV's parent directory.
    sessions : list of pathlib.Path, optional
        Explicitly registered session folders.
    roots : list of pathlib.Path, optional
        Folders recursively searched for ``*.session.mat`` at startup.
    path_maps : list of tuple of str, optional
        Old and new path prefixes. Longest matching prefix wins. Session IDs
        use the original catalog path, so remapping does not change them.
    index_db : pathlib.Path, optional
        Existing private metadata index. Enables discovery beyond the CSV list.
    """

    def __init__(
        self,
        catalogs: list[Path] | None = None,
        sessions: list[Path] | None = None,
        roots: list[Path] | None = None,
        path_maps: list[tuple[str, str]] | None = None,
        index_db: Path | None = None,
    ) -> None:
        self._sessions: dict[str, dict[str, Any]] = {}
        self._index: MetadataIndex | None = None
        self._maps = sorted(path_maps or [], key=lambda x: len(x[0]), reverse=True)
        for filename in catalogs or []:
            filename = Path(os.path.abspath(filename))
            frame = pd.read_csv(filename, dtype=str, keep_default_na=False)
            if "basepath" not in frame:
                raise ValueError(f"{filename}: expected a basepath column.")
            for row in frame.to_dict(orient="records"):
                path = row.pop("basepath").strip()
                if not path:
                    continue
                if not Path(path).is_absolute() and not PureWindowsPath(path).drive:
                    # Match the index's lexical paths without resolving symlinks.
                    path = os.path.abspath(filename.parent / path)
                metadata = {
                    k: v
                    for k, v in row.items()
                    if k and not k.startswith("Unnamed:") and v != ""
                }
                self._register(path, str(filename), metadata)
        for path in sessions or []:
            self._register(os.path.abspath(path), "explicit", {})
        for root in roots or []:
            root = Path(os.path.abspath(root))
            if not root.is_dir():
                raise ValueError(f"Not a directory: {root}")
            for source in sorted(root.rglob("*.session.mat")):
                if source.name == _recording_path(source.parent).name + ".session.mat":
                    self._register(str(source.parent), str(root), {})
        if index_db is not None:
            from neuro_py.mcp.index import MetadataIndex

            self._index = MetadataIndex(index_db)
        if not self._sessions and self._index is None:
            raise ValueError(
                "No sessions registered. Supply a catalog, session, or root."
            )

    def _mapped_path(self, original: str) -> Path:
        windows = bool(PureWindowsPath(original).drive or "\\" in original)
        path_type = PureWindowsPath if windows else Path
        for old, new in self._maps:
            try:
                relative = path_type(original).relative_to(path_type(old))
            except ValueError:
                continue
            return Path(new).joinpath(*relative.parts)
        return Path(original)

    def _register(self, original: str, source: str, metadata: dict[str, str]) -> None:
        key = _identity(original)
        session_id = hashlib.sha256(key.encode()).hexdigest()[:20]
        entry = self._sessions.setdefault(
            session_id,
            {
                "session_id": session_id,
                "catalog_basepath": original,
                "basepath": str(self._mapped_path(original)),
                "basename": PureWindowsPath(original).name,
                "registrations": [],
            },
        )
        registration = {"source": source, "metadata": metadata}
        if registration not in entry["registrations"]:
            entry["registrations"].append(registration)

    def _entry(self, session_id: str) -> dict[str, Any]:
        try:
            return self._sessions[session_id]
        except KeyError as error:
            if self._index is not None:
                entry = self._index.get_entry(session_id)
                if entry is not None:
                    return entry
            raise ValueError("Unknown session_id. Call list_sessions first.") from error

    def _scope_paths(self, scope: str) -> set[str] | None:
        if scope == "all":
            return None
        if scope == "session_list":
            return {_identity(entry["basepath"]) for entry in self._sessions.values()}
        raise ValueError("scope must be all or session_list")

    def index_status(self) -> dict[str, Any]:
        """Report discovery coverage without reading network shares.

        Returns
        -------
        dict
            Cached progress, or disabled status when no index was configured.
        """
        if self._index is None:
            return {
                "status": "disabled",
                "hint": "Build an index, then supply --index-db.",
            }
        result = self._index.status()
        result["session_list_count"] = len(self._sessions)
        return _bounded(result)

    def search_sessions(
        self,
        query: str = "",
        environments: list[str] | None = None,
        regions: list[str] | None = None,
        cell_type: str = "",
        products: list[str] | None = None,
        event_types: list[str] | None = None,
        include_bad_units: bool = False,
        scope: str = "all",
        offset: int = 0,
        limit: int = 20,
        recording_regions: list[str] | None = None,
        fields: list[str] | None = None,
        page_token: str | None = None,
    ) -> dict[str, Any]:
        """Find indexed sessions within the session list or across discovery roots.

        Parameters
        ----------
        query : str, optional
            Literal path, animal, environment, region, or cell-type substring.
        environments, regions, products, event_types : list of str, optional
            AND-combined metadata filters. Products are exact file-type keys.
        cell_type : str, optional
            Cell-type substring required within each requested region.
        recording_regions : list of str, optional
            Session-recorded region labels, independent of sorted cell counts.
        include_bad_units : bool, optional
            Include tagged bad cells for cell matching; default False.
        scope : {"all", "session_list"}, optional
            All indexed sessions or only paths from the registered session list.
        offset, limit : int, optional
            Page offset and row limit (1..200).
        fields : list of str, optional
            Top-level result fields; ["basepath"] returns compact session lists.
            Omit for full metadata. Filtering is independent of selected fields.
        page_token : str, optional
            Token from the first page, required when offset is nonzero. Restart
            at offset zero if the index, filters, or catalog have changed.

        Returns
        -------
        dict
            Matching sessions, usable session IDs, and cached discovery coverage.
        """
        if self._index is None:
            raise ValueError("No index configured. Build one and supply --index-db.")
        fields = _search_fields(fields)
        registered = {
            _identity(row["basepath"]): row["session_id"]
            for row in self._sessions.values()
        }
        index_fields = (
            list(
                dict.fromkeys(
                    [
                        "session_id",
                        "basepath",
                        *[field for field in fields if field != "in_session_list"],
                    ]
                )
            )
            if fields is not None
            else None
        )
        result = self._index.search(
            query,
            environments,
            regions,
            cell_type,
            products,
            event_types,
            include_bad_units,
            self._scope_paths(scope),
            offset,
            limit,
            recording_regions=recording_regions,
            fields=index_fields,
            page_token=page_token,
            query_context=json.dumps([registered, fields], sort_keys=True),
        )
        for row in result["rows"]:
            existing = registered.get(_identity(row["basepath"]))
            row["in_session_list"] = existing is not None
            if existing is not None:
                row["session_id"] = existing
        if fields is not None:
            result["rows"] = [
                {field: row.get(field) for field in fields} for row in result["rows"]
            ]
            result["columns"] = fields
        elif result["rows"]:
            result["columns"].append("in_session_list")
        result["scope"] = scope
        return _bounded(result)

    def summarize_index(
        self,
        facet: str = "environment",
        scope: str = "all",
        offset: int = 0,
        limit: int = 50,
        page_token: str | None = None,
    ) -> dict[str, Any]:
        """Summarize indexed metadata with explicit discovery coverage.

        Parameters
        ----------
        facet : str, optional
            environment, region, cell_type, animal, product, event_type, root,
            or status. Region/type facets exclude tagged bad units.
        scope : {"all", "session_list"}, optional
            All indexed sessions or only registered session-list paths.
        offset, limit : int, optional
            Page offset and row limit (1..200).
        page_token : str, optional
            First-page token, required for nonzero offsets. Restart at offset
            zero if the index or query has changed.

        Returns
        -------
        dict
            Session counts by source label, unknown counts, and scan coverage.
        """
        if self._index is None:
            raise ValueError("No index configured. Build one and supply --index-db.")
        result = self._index.summarize(
            facet,
            self._scope_paths(scope),
            offset,
            limit,
            page_token=page_token,
            query_context=scope,
        )
        result["scope"] = scope
        return result

    def describe(self) -> dict[str, Any]:
        """Return dataset conventions and runtime versions.

        Returns
        -------
        dict
            Registered counts, supported products, conventions, and versions.
        """
        versions = {}
        for name in ("neuro-analysis-py", "nelpy", "mcp"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        return _bounded(
            {
                "session_count": len(self._sessions),
                "metadata_index": self.index_status(),
                "products": SOURCES,
                "conventions": CONVENTIONS,
                "versions": versions,
                "max_rows": MAX_ROWS,
                "max_response_bytes": MAX_RESPONSE_BYTES,
                "limitations": [
                    "Metadata and intervals only; no raw spikes, LFP, or analysis.",
                    "MAT files are loaded on demand, potentially in full by neuro_py.",
                    "No region or task inference from folder names.",
                ],
            }
        )

    def list_sessions(
        self, query: str = "", offset: int = 0, limit: int = 50
    ) -> dict[str, Any]:
        """Search registered paths and catalog metadata without accessing drives.

        Parameters
        ----------
        query : str, optional
            Literal substring to find in paths or catalog metadata.
        offset, limit : int, optional
            Page offset and maximum number of returned sessions.

        Returns
        -------
        dict
            A bounded page containing stable session IDs and registrations.
        """
        rows = [
            entry
            for entry in self._sessions.values()
            if query.casefold() in _search_values(entry).casefold()
        ]
        frame = pd.DataFrame(
            rows,
            columns=pd.Index(
                [
                    "session_id",
                    "catalog_basepath",
                    "basepath",
                    "basename",
                    "registrations",
                ]
            ),
        )
        return _bounded({"status": "ok", **_page(frame, offset, limit)})

    def get_session(self, session_id: str) -> dict[str, Any]:
        """Inspect a registered session's expected files without loading MAT data.

        Parameters
        ----------
        session_id : str
            ID returned by ``list_sessions``.

        Returns
        -------
        dict
            Session metadata and file availability, sizes, and modification times.
        """
        entry = self._entry(session_id)
        available = Path(entry["basepath"]).is_dir()
        recording = _recording_path(Path(entry["basepath"]))
        files = {}
        for product in SOURCES:
            source = recording / (recording.name + SOURCES[product])
            stat = source.stat() if available and source.is_file() else None
            files[product] = {
                "path": str(source),
                "available": stat is not None,
                "size_bytes": stat.st_size if stat else None,
                "mtime_ns": stat.st_mtime_ns if stat else None,
            }
        return _bounded(
            {
                **entry,
                "basename": recording.name,
                "status": "ok" if available else "unavailable",
                "files": files,
            }
        )

    def read_table(
        self,
        session_id: str,
        product: str,
        offset: int = 0,
        limit: int = 50,
        region: str = "",
        cell_type: str = "",
        include_bad_units: bool = False,
        columns: list[str] | None = None,
        manual_events: bool = True,
        state: str = "NREMstate",
    ) -> dict[str, Any]:
        """Read a supported product with source metadata and bounded pagination.

        Parameters
        ----------
        session_id : str
            Registered session ID.
        product : str
            One of ``epochs``, ``cells``, ``trials``, ``ripples``, ``sleep_states``.
        offset, limit : int, optional
            Page offset and maximum rows; counts refer to the filtered table.
        region, cell_type : str, optional
            Literal cell-metric filters.
        include_bad_units : bool, optional
            Include cells marked ``bad_unit``. Default excludes them.
        columns : list of str, optional
            Selected cell metrics. UID is always retained.
        manual_events : bool, optional
            Apply neuro_py's manual ripple-event handling.
        state : str, optional
            Sleep-state interval key. Preserves stored intervals without merging.

        Returns
        -------
        dict
            Data page, provenance, query parameters, and explicit availability.
        """
        _page(pd.DataFrame(), offset, limit)  # Validate before any data loading.
        if product not in SOURCES:
            raise ValueError("Unknown data product.")
        if state not in {"WAKEstate", "NREMstate", "REMstate", "THETA", "nonTHETA"}:
            raise ValueError("Unsupported sleep-state key.")
        entry = self._entry(session_id)
        recording = _recording_path(Path(entry["basepath"]))
        source = recording / (recording.name + SOURCES[product])
        result = {
            "session_id": session_id,
            "basepath": entry["basepath"],
            "basename": recording.name,
            "product": product,
            "source": str(source),
            "parameters": {
                "offset": offset,
                "limit": limit,
                **(
                    {
                        "region": region,
                        "cell_type": cell_type,
                        "include_bad_units": include_bad_units,
                        "columns": columns,
                    }
                    if product == "cells"
                    else {}
                ),
                **({"manual_events": manual_events} if product == "ripples" else {}),
                **({"state": state} if product == "sleep_states" else {}),
            },
        }
        if not Path(entry["basepath"]).is_dir():
            return _bounded({**result, "status": "unavailable"})
        if not source.is_file():
            return _bounded({**result, "status": "missing_file"})
        stat = source.stat()
        result["source_size_bytes"] = stat.st_size
        result["source_mtime_ns"] = stat.st_mtime_ns
        from neuro_py.io import loading

        basepath = str(recording)
        if product == "cells":
            frame = loading.load_cell_metrics(basepath, only_metrics=True)
        elif product == "epochs":
            frame = loading.load_epoch(basepath)
        elif product == "trials":
            frame = loading.load_trials(basepath)
        elif product == "ripples":
            frame = loading.load_ripples_events(basepath, manual_events=manual_events)
        else:
            states = loading.load_SleepState_states(basepath)
            if states is None:
                raise ValueError(
                    "Sleep-state loader returned no data for existing file."
                )
            if state not in states:
                return _bounded({**result, "status": "missing_state"})
            intervals = np.asarray(states[state], dtype=float)
            if intervals.size == 0:
                intervals = np.empty((0, 2))
            else:
                intervals = intervals.reshape(-1, 2)
            frame = pd.DataFrame(intervals, columns=pd.Index(["start", "stop"]))
            frame["state"] = state
        if frame is None or not isinstance(frame, pd.DataFrame):
            raise ValueError("neuro_py did not return the expected table.")
        frame = frame.copy()
        # Preserve loader row identity, independent of filtered-page offsets.
        frame.insert(0, "source_row", frame.index)
        if product == "cells":
            for text, column in [
                (region, "brainRegion"),
                (cell_type, "putativeCellType"),
            ]:
                if text:
                    if column not in frame:
                        raise ValueError(
                            f"Cannot filter: source has no {column} column."
                        )
                    frame = frame.loc[_contains(frame[column], text)]
            if not include_bad_units:
                if "bad_unit" not in frame:
                    raise ValueError(
                        "Cannot exclude bad units: missing bad_unit field."
                    )
                frame = frame.loc[~frame["bad_unit"].fillna(False).astype(bool)]
            if "UID" not in frame:
                raise ValueError("Cell metrics have no UID column.")
            available_columns = list(frame.columns)
            selected = (
                columns
                if columns is not None
                else [name for name in DEFAULT_CELL_COLUMNS if name in frame]
            )
            unknown = set(selected) - set(frame.columns)
            if unknown:
                raise ValueError(f"Unknown columns: {sorted(unknown)}")
            frame = frame.loc[:, list(dict.fromkeys(["source_row", "UID", *selected]))]
            result["available_columns"] = available_columns
        return _bounded(
            {
                **result,
                "status": "empty" if frame.empty else "ok",
                "time_units": "seconds" if product != "cells" else None,
                **_page(frame, offset, limit),
            }
        )
