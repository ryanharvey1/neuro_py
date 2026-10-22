# CellExplorer dataset MCP server

The optional `neuro_py.mcp` server lets an MCP client discover and read registered
CellExplorer sessions through the existing `neuro_py.io.loading` functions. It
runs as a local STDIO process using the same filesystem and network-share access
as the user who launches it. It requires no model API key.

This first version exposes session catalogs, file inventories, cell metrics,
epochs, trials, ripples, and sleep-state intervals. It supports independent
datasets and has no dependency on an analysis repository or lab-specific paths.

## Install

Use Python 3.10 or newer. From a checkout containing this feature:

```bash
python -m pip install -e ".[mcp]"
```

MCP is optional: importing `neuro_py` or `neuro_py.mcp` does not import the SDK.
Install the extra in the same environment as the Python command your client uses.

## Register a dataset

Choose one or combine these repeatable options:

```bash
# CSV with a required basepath column; other columns remain catalog metadata.
neuro-py-mcp --catalog /data/sessions.csv

# Explicit session folders.
neuro-py-mcp --session /data/rat01/day01 --session /data/rat02/day01

# Discover basename.session.mat files recursively, once at startup.
neuro-py-mcp --data-root /data/my_experiment
```

`python -m neuro_py.mcp.server` is equivalent to `neuro-py-mcp`. The command waits
for protocol messages on STDIO; connect it through an MCP client to call tools.
Prefer a CSV or explicit folders for large network shares. Root discovery walks
the selected subtree, so choose an experiment folder rather than an entire drive.

Only a `basepath` column is required. A standard session list works directly:

```csv
basepath
/data/rat01/day01
/data/rat02/day01
```

Additional columns such as animal, environment, or cohort are optional catalog
annotations. You do not need to add them: `get_epochs` and `get_cells` read the
recording's epoch and cell metadata directly from the session files. With a
basepath-only catalog, `list_sessions` searches the paths; it does not automatically
index the metadata inside every session.

For metadata searches across multiple shares and beyond a session list, enable
the optional [automatic metadata index](mcp-index.md).

Relative CSV paths resolve against the CSV directory. Repeated session rows are
deduplicated by normalized source path, while distinct metadata registrations
are preserved. Blank paths are ignored. Catalog files and registrations are read
at startup; restart the server after changing them. Session IDs are derived from
the original registered path and survive path remapping. They are not universal
identifiers across independently renamed or relocated datasets.

Catalog entries, explicit session folders, and startup discovery roots preserve
symbolic-link path spellings, matching the metadata index. Use the same alias
in the catalog and index roots so session-list searches and summaries match.
Different aliases to the same directory remain distinct session identities.
Session IDs for symlinked relative paths, explicit folders, or startup roots
that older versions dereferenced can change; refresh them with `list_sessions`
after restarting the server.

Windows mapped drives can be replaced with UNC paths without editing the CSV:

```powershell
neuro-py-mcp --catalog 'C:\lab\sessions.csv' `
  --path-map 'Z:\=\\fileserver\recordings'
```

Mappings match whole path components; the longest matching prefix wins. A mapping
can also replace an old Windows path with a Linux mount path. On another machine,
change only the launch configuration and dataset registration. UNC access uses
the launching user's existing Windows permissions; the server stores no network
credentials.

## Connect a client

Configure your MCP client to launch the Python environment containing this
checkout. For Codex, merge an entry such as this into your MCP configuration:

```toml
[mcp_servers.cellexplorer]
command = 'C:\path\to\environment\python.exe'
args = ['-m', 'neuro_py.mcp.server', '--catalog', 'C:\lab\sessions.csv']
startup_timeout_sec = 30
tool_timeout_sec = 120
```

On Linux or macOS, use the environment's absolute Python path and POSIX paths in
`args`. Other STDIO clients use the same command and argument list in their
configuration format. See the [generic configuration](examples/mcp/codex.toml)
and [sample catalog](examples/mcp/sessions.csv).

Codex supports local STDIO servers and project or user MCP configuration; see the
[official MCP configuration documentation](https://developers.openai.com/codex/mcp/).
A hosted client needs a separately implemented and authenticated remote transport.
This prototype provides local STDIO only.

## Let your AI set up and connect

Paste the following into a **local coding assistant with terminal and configuration
file access**, replacing the paths. It authorizes the assistant to install the
dependencies and configure the connection. A chat without those capabilities can
explain the setup, but cannot perform the local installation itself.

```text
Set up and connect neuro_py's CellExplorer MCP server in this AI client.

My neuro_py checkout: [absolute path to neuro_py]
My session CSV: [absolute path to sessions.csv; only basepath is required]
My existing metadata index: [absolute path to index.sqlite, or none]

Read docs/mcp.md from that checkout and perform the setup. Use a suitable existing
Python environment or create a dedicated virtual environment. Verify the checkout
contains neuro_py/mcp/server.py and install it with the optional MCP dependencies
using python -m pip install -e ".[mcp]". Use that environment's absolute Python path
to launch python -m neuro_py.mcp.server with my CSV and, if provided, my existing
index. Register it as a local STDIO server named cellexplorer in this client's MCP
configuration, preserving unrelated settings and server entries. Ask only for
essential information you cannot determine from the supplied paths and environment.

Verify installation and registration. If a client reload or restart is needed,
tell me exactly what to do. Once the tools are available, call describe_dataset,
list_sessions, and get_index_status. Distinguish configuration saved from an
actual successful tool connection; do not claim success until those calls work.

Then tell me how many sessions are in my list, preview up to five, and explain
whether indexed metadata is available and how complete its coverage is. Use my
session list as the default query scope. Keep data and index files private, and
start a new metadata crawl only when I request one and specify discovery roots.
```

For Codex, the assistant can register a server with the CLI, for example:

```powershell
codex mcp add cellexplorer -- 'C:\path\to\environment\python.exe' -m neuro_py.mcp.server --catalog 'C:\lab\sessions.csv'
```

Supply `--index-db` and its absolute path after the catalog argument when using
an existing index. Inspect an existing `cellexplorer` entry before updating it.
`codex mcp list` verifies registration. Actual calls from the AI client verify
connection and access to the tools. Follow the client's reload/restart procedure
when needed; registration alone does not make new tools available in every
already-open chat. See the [official MCP setup instructions](https://learn.chatgpt.com/docs/extend/mcp?surface=cli).

The ordinary ChatGPT website does not read local Codex configuration. This setup
prompt is for a client with local execution and STDIO MCP support. Connecting a
hosted client requires a separately supported remote integration.

## Start using the connected server

After setup, users can paste this shorter prompt into a new chat:

```text
Use the connected CellExplorer MCP to explore my recordings. Start with
describe_dataset, list_sessions, and get_index_status. Report my session-list
count and preview up to five sessions. If an index is available, summarize the
environments, recorded regions, and file types within my session list. Distinguish
the indexed subset from my full list, and report incomplete coverage or unknown
metadata explicitly. Use the MCP tools for data access and keep my session list
as the default scope unless I ask for discovery across all indexed shares.
```

For broader discovery, a follow-up can be:

```text
Search all indexed shares, including sessions outside my CSV. Summarize the
recorded environments, regions, and file types, and report scan coverage and
metadata gaps so I can tell whether the inventory is complete.
```

Users do not need to memorize tool names; these prompts give the AI a starting
workflow. Later questions can use ordinary language, such as "Find cheeseboard
sessions with CA1 pyramidal cells in my session list."

For complete basepath lists, ask the AI to use indexed
`search_sessions(fields=["basepath"], limit=200)` and follow every page. Indexed
searches and summaries return a `page_token` which must accompany later
`next_offset` requests with the same filters. A stale token means the index
changed: discard accumulated pages and restart at offset zero. See
[compact results and safe pagination](mcp-index.md#compact-basepath-lists-and-safe-pagination).

## Tools

| Tool | Purpose |
| --- | --- |
| `describe_dataset` | Read conventions, capabilities, versions, and limits. |
| `list_sessions` | Find IDs using literal path or catalog metadata text. |
| `get_session` | Check expected files, sizes, and modification times. |
| `get_epochs` | Read original session epochs and metadata. |
| `get_cells` | Read cell metrics, optionally filtering region and cell type. |
| `get_trials` | Read stored trial intervals and labels. |
| `get_ripples` | Read ripple intervals and metrics using loader event handling. |
| `get_sleep_states` | Read intervals for one sleep-state key. |
| `get_index_status` | Report optional metadata-index coverage and scan errors. |
| `search_sessions` | Find indexed sessions by task, region/type, and file availability. |
| `summarize_dataset` | Count indexed sessions by metadata category. |

All tools declare read-only annotations. The `cellexplorer://conventions` resource
provides the same interpretation rules as `describe_dataset`. Session-specific
tools accept only an ID returned by `list_sessions` or `search_sessions`; model calls cannot register
arbitrary files or execute Python.

A typical request is: "Find rat01 sessions, inspect the available files, then
show CA1 pyramidal cell metadata and the original task epochs for one session."
`list_sessions` searches supplied catalog text. A catalog with only paths supports
path searches. Use `search_sessions` for indexed metadata discovery, and
`get_cells` or `get_epochs` for recording-level details.

## Output and scientific conventions

- Times are in session-relative **seconds**, following the loaders. Cell identity
  is `(session_id, UID)`; UID is not a zero-based row number.
- Region, epoch, and trial labels stay as loaded. There is no region renaming,
  epoch compression, pre/task/post inference, or trial reclassification.
- `get_cells` excludes tagged bad units by default. Set `include_bad_units=true`
  to inspect them. Region and cell-type filters are case-insensitive literal
  substring matches, so `CA1` includes `CA1sp`, while `CA1.*` is literal text.
- The default cell table contains common metrics. `available_columns` lists all
  loader-provided columns; request additional metrics with `columns`. UID and
  `source_row` are always retained.
- `get_ripples` uses the installed loader's flagged-event exclusion. Its explicit
  `manual_events` option defaults to true, following the loader; manually added
  events use the loader's duration convention. `source_row` identifies the
  **loader output** row, which may differ from the original MATLAB event index.
- Sleep-state keys are `WAKEstate`, `NREMstate`, `REMstate`, `THETA`, and
  `nonTHETA`. Stored intervals are returned without merging or clipping.
- Tables include source path, size, modification time, and query parameters.
  Session context applies to every returned row. Missing numeric values become
  JSON `null`, including NaN and infinity. `describe_dataset` reports versions.
- `total_rows` is the full **filtered** table size. `offset` and `limit` page it;
  follow `next_offset` until null. Default limit is 50, maximum 200. A response
  over 128 KB fails with a request to reduce rows or columns.

`status` distinguishes `ok`, `empty` (zero matching rows), `missing_file`,
`missing_state`, and `unavailable` (session directory absent). Permission errors,
malformed files, and loader failures are tool errors, not empty results. Catalog
metadata can be stale; file existence is checked by `get_session` and data reads.
The inventory describes expected standard filenames, not all session contents.

## Limits and future work

Pagination bounds the response, not the loader's memory use: MAT files may load
in full, including when selecting cell metrics or sleep intervals. This version
does not expose raw spikes, position samples, continuous LFP, arbitrary event
types, or analysis execution. Add bounded time windows, loader-aware memory
limits, and explicit provenance before extending those capabilities. Metadata
indexing and cross-session metadata summaries are available through the optional
[index builder](mcp-index.md). Scientific analysis execution remains outside
these tools.

Register only trusted CellExplorer directories. The server reads files using the
launching user's account; it is not a multi-user authorization boundary. It has
no HTTP listener, upload service, or write tools.

## Validate a checkout

```bash
python -m pip install -e ".[mcp,dev]"
python -m pytest tests/mcp -q
```

Tests use small synthetic MAT files and the real `neuro_py` loaders, including
UID preservation, bad-unit tags, trial labels, ripple flags/manual additions,
missing versus empty data, path mapping, pagination, payload limits, and a real
MCP STDIO client/server round trip. Protocol tests skip when the optional SDK is
absent; catalog and loader tests still run. Base-package users need no MCP SDK.

On POSIX, `tests/mcp/test_paths.py` also checks that relative/absolute catalogs,
explicit folders, and startup roots agree with indexed symlink paths, including
session-list searches, summaries, IDs, and live epoch reads. Cache-containment
tests cover both root and explicit-session inputs and verify rejection before
creating the cache directory, database, or lock.
