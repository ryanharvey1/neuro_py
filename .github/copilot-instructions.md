# neuro_py Agent Instructions

neuro_py is a Python package for analysis of freely moving neuroelectrophysiology data, built on top of nelpy for core data objects.

These instructions are written to be useful for GitHub Copilot, Codex, and similar coding agents. Keep the guidance implementation-focused and repo-specific rather than tied to one tool's UI or workflow.

Repo docs:
- API: https://ryanharvey1.github.io/neuro_py/reference/
- Tutorials: https://ryanharvey1.github.io/neuro_py/tutorials/attractor_landscape/

## Agent Behavior

- Make the smallest change that fixes the problem at the root cause.
- Preserve existing public APIs unless the task explicitly requires a breaking change.
- Prefer local, targeted edits over broad refactors.
- If a change affects behavior, add or update a regression test in the matching `tests/<module>/` area.
- Do not widen scope because you notice unrelated issues.
- Keep scientific behavior stable unless the task explicitly calls for an algorithmic change.
- When behavior must change, describe the old vs new behavior clearly in code comments, tests, or the PR summary.

## Repo Conventions

- Use type hints on all function signatures.
- Use numpydoc for public docstrings.
- Keep imports in the usual order: stdlib, third-party, then project imports.
- Use `nelpy` objects for time series and ephys data when the API expects them.
- Use `EpochArray` indexing instead of manual boolean masking when restricting to epochs.
- When editing `neuro_py/io/loading.py`, also follow the loader-specific rules in `.github/instructions/io-loading.instructions.md`.
- Preserve existing return types and shape conventions unless the task explicitly requires a change.
- Prefer column names and outputs that match existing conventions such as `start`, `stop`, `peaks`, `center`, `duration`, `startTime`, and `stopTime`.
- Treat timestamps, sampling rates, and array orientation as part of the public behavior. Be explicit about whether data is `(n_samples, n_channels)` or `(n_channels, n_samples)`.
- Prefer warning-and-empty-result behavior over hard failure when a loader is missing an expected file, unless nearby loaders already establish a stricter contract.
- Avoid hidden unit changes. If a function works in seconds, samples, Hz, radians, or channel indices, keep that contract explicit.

## Scientific Guardrails

- Preserve numerical intent before chasing cleanup. Small-looking changes to indexing, smoothing, thresholds, interpolation, or event windows can change scientific results.
- Be careful with inclusive vs exclusive interval boundaries and off-by-one sample behavior.
- Preserve NaN handling when possible. Do not silently coerce missing data to zeros.
- Keep channel numbering and shank/group conventions aligned with the surrounding code and tests.
- When restricting data to epochs or supports, prefer the existing `nelpy`-based pathway used by nearby code.
- For behavior and LFP code, be cautious about timestamp alignment, sample counts, and whether derived arrays must stay the same length as the input signal.

## Performance

- Prefer vectorized NumPy or pandas operations over Python loops for sample-wise work.
- Avoid unnecessary materialization or copying of large arrays, memmaps, or lazy loader views.
- Preserve lazy-loading behavior unless the task explicitly requires eager loading.
- In loader code, avoid reading full binary files into memory when an existing memmap or view-based approach already works.

## Standard Imports

```python
import nelpy as nel
import nelpy.plotting as npl
import neuro_py as npy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

## Package Structure

```
neuro_py/
    behavior/       # Position, linearization, kinematics, trials, well traversals
    detectors/      # Event detectors (dentate spikes, UP/DOWN states)
    ensemble/       # Assemblies, reactivation, decoding, replay, geometry
    io/             # CellExplorer format I/O
    lfp/            # Spectral, CSD, theta cycles, preprocessing
    plotting/       # Figure helpers, event plots, replay plots, decorators
    process/        # PETH, correlations, intervals, batch analysis, utils
    raw/            # Raw data preprocessing, spike sorting utilities
    session/        # Epoch location, session metadata
    spikes/         # Spike train tools, firing rate, burst detection
    stats/          # Circular stats, regression, system identification
    tuning/         # Place fields, spatial tuning maps
    util/           # Array utilities
```

## Tooling

- Ruff checks NumPy 2.0 compatibility.
- Formatting should stay black-compatible.
- Pytest is the test runner.
- Write tests in pytest style; avoid `unittest.TestCase` and `unittest` assertion helpers.
- Python support starts at 3.10.
- CI runs `ruff check .` and `pytest` on Python 3.10, 3.11, 3.12, and 3.13.

Before finishing a change, run the narrowest relevant test subset first, then broader tests if needed. If you cannot run tests in the current environment, say so explicitly.

## Docstrings

Every public function should have a numpydoc docstring.

Good pattern:

```python
def load_spikes(basepath: str, putativeCellType: list | None = None) -> nel.SpikeTrainArray:
    """
    Load spike data from a CellExplorer session folder.

    Parameters
    ----------
    basepath : str
        Path to the session folder.
    putativeCellType : list, optional
        List of cell types to include. If None, all cells are loaded.

    Returns
    -------
    nel.SpikeTrainArray
        Spike train array with one unit per array entry.
    """
```

## Type Hints

- Use `|` or `Union` for optional types.
- Avoid `Any` unless there is no reasonable alternative.
- Keep signatures explicit enough that the intended return type is obvious.

## Test Placement

- Put tests in the matching module folder under `tests/`.
- Name tests after the behavior they cover, not the implementation detail.
- Add edge-case coverage when a bug fix changes control flow or input validation.
- Prefer realistic small fixtures over heavy mocks when shape, timestamps, or `nelpy` behavior matter.
- Assert behavior that users rely on: output types, array shapes, column names, warning behavior, and numerical sanity.

Common module-to-test mappings:

- `neuro_py/behavior/*` -> `tests/behavior/`
- `neuro_py/detectors/*` -> `tests/detectors/`
- `neuro_py/ensemble/*` -> `tests/ensemble/`
- `neuro_py/ensemble/decoding/*` -> `tests/ensemble/decoding/`
- `neuro_py/io/loading.py` -> `tests/test_io/test_loading.py`
- `neuro_py/lfp/*` -> `tests/lfp/`
- `neuro_py/plotting/*` -> `tests/plotting/`
- `neuro_py/process/*` -> `tests/process/`
- `neuro_py/raw/*` -> `tests/raw/`
- `neuro_py/session/*` -> `tests/session/`
- `neuro_py/spikes/*` -> `tests/spikes/`
- `neuro_py/stats/*` -> `tests/stats/`
- `neuro_py/tuning/*` -> `tests/tuning/`
- `neuro_py/util/*` -> `tests/util/`

Useful targeted test commands:

- `pytest tests/test_io/test_loading.py`
- `pytest tests/process`
- `pytest tests/lfp`
- `pytest tests/behavior`
- `pytest tests/ensemble`

## What Not to Do

- Do not hardcode session-specific filenames.
- Do not manually slice time series when nelpy indexing works.
- Do not introduce mutable default arguments.
- Do not skip tests for behavior changes.
- Do not expand a fix into unrelated cleanup.
- Do not silently change output orientation, units, or return types.
- Do not replace warnings with exceptions, or exceptions with warnings, unless nearby code and tests support the change.
- Do not densify large lazy arrays just to satisfy a small indexing or shape check.
- Do not “fix” failing scientific outputs by loosening assertions without understanding the numerical contract.

## Change Strategy

- For loaders and preprocessing, prioritize compatibility with messy real datasets: missing files, incomplete metadata, scalar-vs-vector MATLAB fields, NaNs, and mixed session layouts.
- For plotting utilities, preserve defaults and return values that downstream notebooks may rely on.
- For numerical routines, prefer a narrowly scoped regression test that captures the bug with a compact synthetic example.
- For public APIs, use deprecation-style changes only when necessary: keep old behavior working when feasible, and update tests/docstrings to explain the transition.

## Reference

- Issues: https://github.com/ryanharvey1/neuro_py/issues
- nelpy docs: https://nelpy.github.io/nelpy/
