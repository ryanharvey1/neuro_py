"""Exercise initialization, schemas, resources, reads, and errors over real STDIO."""

import asyncio
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp", reason="Install the optional mcp extra for protocol tests")

from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402


def test_stdio_roundtrip(catalog, session, tmp_path):
    from neuro_py.mcp.index import IndexBuilder

    cache = tmp_path / "private" / "index.sqlite"
    IndexBuilder(cache).build(roots=[session.parent])

    async def exercise():
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join(
            filter(
                None,
                [
                    str(Path(__file__).resolve().parents[2]),
                    environment.get("PYTHONPATH"),
                ],
            )
        )
        parameters = StdioServerParameters(
            command=sys.executable,
            args=[
                "-m",
                "neuro_py.mcp.server",
                "--catalog",
                str(catalog),
                "--index-db",
                str(cache),
            ],
            env=environment,
        )
        async with stdio_client(parameters) as (read, write):
            async with ClientSession(read, write) as client:
                initialized = await client.initialize()
                assert "session_id" in initialized.instructions
                tools = (await client.list_tools()).tools
                assert len(tools) == 11
                assert all(t.annotations.readOnlyHint for t in tools)
                assert all(t.outputSchema for t in tools)
                conventions = await client.read_resource("cellexplorer://conventions")
                assert "cell_identity" in conventions.contents[0].text
                described = await client.call_tool("describe_dataset")
                assert described.structuredContent["session_count"] == 1
                status = await client.call_tool("get_index_status")
                assert status.structuredContent["discovery_complete"]
                indexed = await client.call_tool(
                    "search_sessions",
                    {
                        "environments": ["linear"],
                        "regions": ["CA1"],
                        "cell_type": "Pyr",
                        "scope": "session_list",
                    },
                )
                assert indexed.structuredContent["total_rows"] == 1
                compact = await client.call_tool(
                    "search_sessions",
                    {
                        "fields": ["basepath"],
                        "limit": 200,
                        "environments": ["linear"],
                        "regions": ["CA1"],
                    },
                )
                assert compact.structuredContent["rows"] == [{"basepath": str(session)}]
                compact_last = await client.call_tool(
                    "search_sessions",
                    {
                        "fields": ["basepath"],
                        "limit": 200,
                        "environments": ["linear"],
                        "regions": ["CA1"],
                        "offset": 1,
                        "page_token": compact.structuredContent["page_token"],
                    },
                )
                assert compact_last.structuredContent["rows"] == []
                stale = await client.call_tool(
                    "search_sessions",
                    {"fields": ["basepath"], "offset": 1, "page_token": "stale"},
                )
                assert stale.isError
                summary = await client.call_tool(
                    "summarize_dataset", {"facet": "region"}
                )
                assert summary.structuredContent["total_rows"] == 2
                found = await client.call_tool("list_sessions", {"query": "rat01"})
                session_id = found.structuredContent["rows"][0]["session_id"]
                assert indexed.structuredContent["rows"][0]["session_id"] == session_id
                for name in [
                    "get_session",
                    "get_epochs",
                    "get_cells",
                    "get_trials",
                    "get_ripples",
                    "get_sleep_states",
                ]:
                    result = await client.call_tool(name, {"session_id": session_id})
                    assert not result.isError, result
                    assert result.structuredContent["status"] == "ok"
                invalid = await client.call_tool(
                    "get_epochs", {"session_id": "../private"}
                )
                assert invalid.isError
                invalid = await client.call_tool("list_sessions", {"limit": 201})
                assert invalid.isError

    asyncio.run(exercise())
