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
tools accept only an ID returned by `list_sessions`; model calls cannot register
arbitrary files or execute Python.

A typical request is: "Find rat01 sessions, inspect the available files, then
show CA1 pyramidal cell metadata and the original task epochs for one session."
Search examines supplied catalog text; it does not infer animal, region, or task
from MAT data across the whole dataset. A catalog with only paths will support
path searches. Use `get_cells` or `get_epochs` for recording-level details.

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
