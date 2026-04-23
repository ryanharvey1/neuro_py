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

## Repo Conventions

- Use type hints on all function signatures.
- Use numpydoc for public docstrings.
- Keep imports in the usual order: stdlib, third-party, then project imports.
- Use `nelpy` objects for time series and ephys data when the API expects them.
- Use `EpochArray` indexing instead of manual boolean masking when restricting to epochs.
- When editing `neuro_py/io/loading.py`, also follow the loader-specific rules in `.github/instructions/io-loading.instructions.md`.

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

## What Not to Do

- Do not hardcode session-specific filenames.
- Do not manually slice time series when nelpy indexing works.
- Do not introduce mutable default arguments.
- Do not skip tests for behavior changes.
- Do not expand a fix into unrelated cleanup.

## Reference

- Issues: https://github.com/ryanharvey1/neuro_py/issues
- nelpy docs: https://nelpy.github.io/nelpy/
