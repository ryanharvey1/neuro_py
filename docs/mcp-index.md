# Automatic metadata indexing

The metadata index adds discovery across recording roots to the usual
`basepath`-only session list. The builder reads session and cell metadata using
the existing `neuro_py` loaders, saves a private local SQLite cache, and records
its progress. MCP clients query that cache without crawling network shares.

Each person should build the index using their own filesystem permissions. The
index is private data: it contains source paths and unpublished metadata.

## Build an index

From a checkout containing this feature, use either CLI arguments:

```bash
python -m neuro_py.mcp.index \
  --index-db .neuro-py-mcp/index.sqlite \
  --root /data/experiment_one \
  --root /data/experiment_two \
  --catalog /data/my_sessions.csv
```

or a private JSON configuration:

```json
{
  "index_db": ".neuro-py-mcp/index.sqlite",
  "roots": ["/data/experiment_one", "/data/experiment_two"],
  "catalogs": ["/data/my_sessions.csv"],
  "path_maps": []
}
```

```bash
python -m neuro_py.mcp.index --config /path/to/private-index-config.json
```

The console entry point `neuro-py-mcp-index` runs the same command. Index building
uses the base package and does not require an MCP SDK or a model API key.
Roots and catalogs are independent: roots discover sessions outside the list,
and the catalog ensures that listed sessions are checked even when outside a
discovery root. Neither is required when the other is supplied. Relative config
paths resolve against the command's working directory.

On Windows, roots can be UNC shares. Use JSON escaping for backslashes:

```json
{
  "index_db": "C:/Users/yourname/.neuro-py-mcp/index.sqlite",
  "roots": ["\\\\fileserver\\recordings", "\\\\otherserver\\recordings"],
  "catalogs": ["C:/lab/sessions.csv"],
  "path_maps": [["Z:\\", "\\\\fileserver\\recordings"]]
}
```

Keep the index on a **local disk outside the recording roots**. It uses SQLite
WAL mode for a single scanner and concurrent readers; it is not intended to run
from an SMB/NFS share. The cache directory `.neuro-py-mcp/` is Git-ignored.
New cache directories/files use restrictive permissions on POSIX. On Windows,
choose a private folder with the appropriate inherited access permissions.
The builder checks resolved filesystem paths before creating the cache directory
or lock, so a symbolic-link alias cannot place the cache inside a recording root
or explicitly registered session folder.

## Resume and refresh

A broad share scan may take a long time. Budget a run when needed:

```bash
python -m neuro_py.mcp.index --config private-index-config.json --max-seconds 300
```

`--max-directories 1000` provides another stopping condition. The builder commits
each directory and its newly discovered children. Repeating the command resumes
the unfinished pass. It finishes pending roots before refreshing completed ones
and balances progress across roots. A crash can cause the current directory to
be retried; it does not require restarting the whole scan.

After a completed pass, the next invocation starts a refresh. Unchanged source
size/mtime fingerprints reuse the metadata cache. Source changes or size-limit
changes that alter a source's eligibility trigger re-extraction. Raising the
limit reuses already eligible unchanged metadata. `--restart` starts discovery
again while keeping unchanged metadata reusable; use it after changing the
configured inputs during
an unfinished pass. There is no automatic scheduler or background scan on MCP
startup. Budgets are checked between directories: one network operation or MAT
load can exceed the requested duration.

Traversal follows directories under the configured roots and skips symbolic
links, Windows junctions, development caches, and system recycling, snapshot,
and thumbnail metadata directories. The status response lists the exact excluded
directory names. Add names with an optional `exclude_dirs` list in the private
JSON configuration. It recognizes
directories containing a matching `basename.session.mat` or
`basename.cell_metrics.cellinfo.mat`. Both files are not required. Nested roots
and catalog overlap are deduplicated by normalized accessible path. Different
UNC aliases or differently named mounts are not resolved into one identity.
Catalog registration and discovery retain the same symbolic-link spelling;
relative CSV entries are made absolute without dereferencing links. Use the same
alias in the catalog and discovery roots for session-list queries. This also
applies to explicit session folders and server startup discovery roots.
An explicitly supplied root may contain symbolic-link components; links found
beneath that root are skipped during traversal.

Use the same path mappings in the index configuration and MCP launch command.
Map every drive letter in a Windows catalog to its corresponding scanned UNC
root; otherwise the drive-letter and UNC registrations can be counted as
different session folders.

Inaccessible subtrees produce explicit root errors. Previously discovered
sessions are retained when traversal has errors, since inaccessibility is not
evidence of deletion. After a complete error-free traversal, stale discoveries
under that root are removed. Roots omitted from later configurations remain in
the cache; use a separate cache when changing the set of accessible datasets.

## Connect the index to MCP

For AI-assisted installation and connection, copy the
[setup prompt](mcp.md#let-your-ai-set-up-and-connect) and supply your existing
index path. For everyday questions, use the
[first-use prompt](mcp.md#start-using-the-connected-server).

Keep the usual catalog and add the existing cache:

```bash
python -m neuro_py.mcp.server \
  --catalog /data/my_sessions.csv \
  --index-db /path/to/private/index.sqlite
```

The server can also run with just `--index-db`. Existing session-list tools keep
their behavior. The index adds these read-only tools:

| Tool | Purpose |
| --- | --- |
| `get_index_status` | Report root coverage, pending directories, errors, and metadata status counts. |
| `search_sessions` | Filter cached sessions by environments, regions, cell types, events, and files. |
| `summarize_dataset` | Count sessions by environment, region, animal, product, event type, root, or status. |

Use `scope="all"` for all indexed sessions, or `scope="session_list"` for the
paths in the currently registered catalogs. Search results include
`in_session_list` and IDs that work with the existing `get_*` tools. For paths in
both sources, the original catalog session IDs are preserved.
`scope` controls query selection; it is not an authorization boundary. Attaching
an index makes all of its indexed session registrations available to these tools.

Example questions:

- "What environments and recorded brain regions are represented across the shares?"
- "Find linear-track sessions containing both CA1 and PFC pyramidal cells."
- "Which sessions have ripple-event files and sleep scoring?"
- "Within my session list, find cheeseboard recordings with CA1 cells."
- "How much of the scan is complete, and which folders could not be read?"

For example, `search_sessions(environments=["linear"], regions=["CA1", "PFC"],
cell_type="Pyr", products=["ripples", "sleep_states"])` requires all those
conditions. Environment and region/type matches are literal case-insensitive
substrings: `CA1` includes labels such as `ca1so`, `ca1sp`, `ca1sr`, and
`ca1slm`. Original labels remain in the metadata. A substring can also match
other labels such as `ACA1`; inspect the region summary and returned labels
when anatomical specificity matters. Region and cell type must occur together
in the same cell group;
CA1 interneurons plus PFC pyramidal cells do not satisfy "CA1 pyramidal".
Tagged bad units are excluded by default. Set `include_bad_units=true` when
their presence matters. Original labels are retained; no region aliases are
inferred.

`recording_regions` instead filters labels from `session.brainRegions`, even
when there are no sorted cells. Use the `recording_region` summary facet for
those labels, and `region` for regions containing indexed non-bad cells. The
index also retains session animal names and available `sr`, `srLfp`, and
`nChannels` metadata. Sample rates retain the source convention of Hz.

## Compact basepath lists and safe pagination

For a reusable session list, request only the basepath field. All filters and
the total count retain their usual meaning:

```json
{
  "scope": "all",
  "environments": ["linear"],
  "regions": ["CA1"],
  "fields": ["basepath"],
  "limit": 200
}
```

Call `search_sessions` with these arguments. Each row contains only `basepath`;
counts, cached coverage, `next_offset`, and `page_token` remain at the response's
top level. To retain IDs and catalog membership, request
`fields=["session_id", "basepath", "in_session_list"]` instead. Omit `fields`
for full metadata. Projection happens before the response-size limit, so large
metadata do not prevent a compact path list from being returned.

Both `search_sessions` and `summarize_dataset` return a `page_token`. When
`next_offset` is not null, repeat the same query with `offset=next_offset` and
the returned `page_token`. Nonzero offsets require a token; changing filters,
scope, or selected fields requires starting a new query at offset zero.
Changing the page size is allowed.

Each indexed query reads results and coverage from one SQLite read snapshot.
If indexed session data change between pages, the next request returns a stale
token error. Discard all accumulated pages and restart at `offset=0` without a
token. An unchanged total count alone is not sufficient to detect a refresh.
This detects refreshes by other processes and works with existing read-only
indexes without rebuilding them. Catalog-only `list_sessions` and the live
session table tools retain their existing pagination behavior.

## Interpret results correctly

Discovery skips Phy's `.phy` cache directories as well as the system and
development directories listed in `discovery_policy`. Adding an exclusion
also removes its already-queued descendants when resuming an older crawl;
the builder does not need to visit those cache folders on the network.
Explicit session-list registrations are still processed.

Every query includes cached scan coverage. `discovery_complete=false` means the
results are not an exhaustive inventory of the configured roots. A complete
discovery can still have incomplete per-session metadata; inspect
`metadata_status_counts` and individual errors. Unknown metadata must not be
interpreted as proof that a region, task, or cell type was not recorded.

The index loads `.session.mat` and cell metrics. Other products are inventoried
by filename and size/mtime without loading signals or event payloads. Thus a
`behavior` file does not guarantee stored trials or position, a `spikes` file
count only describes separate spike files, and `lfp`/`eeg` presence does not
validate their contents. Raw spikes embedded in cell metrics may exist without
a separate spike file. Event types come from `basename.*.events.mat` filenames.

`--max-source-bytes` defaults to 128,000,000 bytes per metadata MAT source. Larger
files are recorded with explicit partial-metadata errors rather than loaded.
The limit bounds source-file size, not decoded memory use. Increase it only for
an environment with enough memory. No binary recording data are read by the
builder. `matching_cell_count` is derived from joint region/type groups; it is
null when cell metadata could not be indexed. Summaries count sessions, with
each session counted once for each applicable label.

Queries read cached observations, not current network permissions or file
contents. The cache is not encrypted, and copying an index built by a privileged
account can expose metadata to someone without access to the source recordings.
Keep lab configuration, indexes, WAL files, and logs private. Publishing the
software does not publish the data, and this local server has no HTTP listener
or per-user authorization service. Tool results are sent to the connected AI
client. Treat metadata strings as data rather than instructions.

Root status `scanning` describes an unfinished discovery pass. A budgeted command
can exit with that status and resume on its next invocation; it does not mean a
background scanner is necessarily running.
