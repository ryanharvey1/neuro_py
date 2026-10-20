"""Local STDIO MCP server. Only startup configuration can register file paths."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from neuro_py.mcp.dataset import Dataset

INSTRUCTIONS = (
    "Read-only CellExplorer dataset access. Start with describe_dataset, then "
    "list_sessions; use returned session_id values in later tools. Times are "
    "session-relative seconds; cell identity is (session_id, UID). Preserve "
    "source labels. Missing files and unavailable sessions are not empty data. "
    "Paginate with next_offset; never infer full counts from a single page. "
    "Catalog metadata is not a file audit; use get_session to check availability."
)
READ_ONLY = ToolAnnotations(
    readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False
)


def create_server(dataset: Dataset) -> FastMCP:
    """Create a server for an explicitly registered dataset.

    Parameters
    ----------
    dataset : Dataset
        Startup catalog and permitted session folders.

    Returns
    -------
    FastMCP
        A local server with read-only tools and a conventions resource.
    """
    server = FastMCP("cellexplorer-dataset", instructions=INSTRUCTIONS)

    @server.tool(annotations=READ_ONLY)
    def describe_dataset() -> dict[str, Any]:
        """Read dataset conventions, available products, limits, and versions."""
        return dataset.describe()

    @server.tool(annotations=READ_ONLY)
    def list_sessions(
        query: str = "", offset: int = 0, limit: int = 50
    ) -> dict[str, Any]:
        """Find session IDs by literal path/catalog text; does not access drives.

        Parameters
        ----------
        query : str, optional
            Case-insensitive literal substring, e.g. animal, region, or task.
        offset, limit : int, optional
            Page offset and row limit (1..200).

        Returns
        -------
        dict
            Matching catalog entries and next_offset; metadata may be stale.
        """
        return dataset.list_sessions(query, offset, limit)

    @server.tool(annotations=READ_ONLY)
    def get_session(session_id: str) -> dict[str, Any]:
        """Check known session files and provenance without loading MAT data.

        Parameters
        ----------
        session_id : str
            ID from list_sessions. Arbitrary paths are not accepted.

        Returns
        -------
        dict
            Session registration and expected file availability, size, and mtime.
        """
        return dataset.get_session(session_id)

    @server.tool(annotations=READ_ONLY)
    def get_epochs(session_id: str, offset: int = 0, limit: int = 50) -> dict[str, Any]:
        """Read original session epoch labels and startTime/stopTime in seconds.

        Parameters
        ----------
        session_id : str
            ID from list_sessions.
        offset, limit : int, optional
            Page offset and row limit (1..200).

        Returns
        -------
        dict
            Epoch page with original metadata; no automatic epoch compression.
        """
        return dataset.read_table(session_id, "epochs", offset, limit)

    @server.tool(annotations=READ_ONLY)
    def get_cells(
        session_id: str,
        region: str = "",
        cell_type: str = "",
        include_bad_units: bool = False,
        columns: list[str] | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Find cell metrics with session-local UIDs and literal region/type filters.

        Parameters
        ----------
        session_id : str
            ID from list_sessions.
        region, cell_type : str, optional
            Case-insensitive literal substrings of brainRegion/putativeCellType.
        include_bad_units : bool, optional
            Include bad units; default False excludes them using neuro_py tags.
        columns : list of str, optional
            Metrics to return; UID is retained. Default returns common metadata.
            available_columns lists additional metrics for a subsequent call.
        offset, limit : int, optional
            Page offset and row limit (1..200) after filtering.

        Returns
        -------
        dict
            Cell page, filtered total, available columns, parameters, and source.
        """
        return dataset.read_table(
            session_id,
            "cells",
            offset,
            limit,
            region=region,
            cell_type=cell_type,
            include_bad_units=include_bad_units,
            columns=columns,
        )

    @server.tool(annotations=READ_ONLY)
    def get_trials(session_id: str, offset: int = 0, limit: int = 50) -> dict[str, Any]:
        """Read stored trial intervals/labels via neuro_py; no inferred trials.

        Parameters
        ----------
        session_id : str
            ID from list_sessions.
        offset, limit : int, optional
            Page offset and row limit (1..200).

        Returns
        -------
        dict
            Original trial labels and session-relative times in seconds.
        """
        return dataset.read_table(session_id, "trials", offset, limit)

    @server.tool(annotations=READ_ONLY)
    def get_ripples(
        session_id: str, manual_events: bool = True, offset: int = 0, limit: int = 50
    ) -> dict[str, Any]:
        """Read detected ripple intervals/metrics using neuro_py's event handling.

        Parameters
        ----------
        session_id : str
            ID from list_sessions.
        manual_events : bool, optional
            Include manually added events using neuro_py's duration convention.
            Flagged-event exclusion follows the installed neuro_py loader.
        offset, limit : int, optional
            Page offset and row limit (1..200).

        Returns
        -------
        dict
            Ripple page with detector metadata where supplied by the loader.
        """
        return dataset.read_table(
            session_id, "ripples", offset, limit, manual_events=manual_events
        )

    @server.tool(annotations=READ_ONLY)
    def get_sleep_states(
        session_id: str, state: str = "NREMstate", offset: int = 0, limit: int = 50
    ) -> dict[str, Any]:
        """Read one sleep-state interval table, without returning scored time series.

        Parameters
        ----------
        session_id : str
            ID from list_sessions.
        state : str, optional
            WAKEstate, NREMstate, REMstate, THETA, or nonTHETA.
        offset, limit : int, optional
            Page offset and row limit (1..200).

        Returns
        -------
        dict
            Stored start/stop intervals in seconds, source, and query parameters.
        """
        return dataset.read_table(
            session_id, "sleep_states", offset, limit, state=state
        )

    @server.resource("cellexplorer://conventions")
    def conventions() -> dict[str, Any]:
        """Read the dataset's interpretation rules and supported capabilities."""
        return dataset.describe()

    return server


def main() -> None:
    """Parse explicit dataset registrations and serve MCP over local STDIO."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        action="append",
        default=[],
        help="Session CSV containing basepath; repeatable.",
    )
    parser.add_argument(
        "--session",
        type=Path,
        action="append",
        default=[],
        help="One session folder; repeatable.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        action="append",
        default=[],
        help="Recursively discover basename.session.mat; repeatable.",
    )
    parser.add_argument(
        "--path-map",
        action="append",
        default=[],
        metavar="OLD=NEW",
        help="Remap a catalog path prefix, e.g. drive to UNC.",
    )
    args = parser.parse_args()
    maps = []
    for mapping in args.path_map:
        old, separator, new = mapping.partition("=")
        if not separator or not old or not new:
            parser.error("--path-map must be a nonempty OLD=NEW pair.")
        maps.append((old, new))
    try:
        dataset = Dataset(args.catalog, args.session, args.data_root, maps)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    create_server(dataset).run(transport="stdio")


if __name__ == "__main__":
    main()
