---
applyTo: "neuro_py/io/loading.py"
description: "Loader conventions for neuro_py io/loading changes"
---

## io/loading Conventions

When editing loaders in `io/loading.py`:

- All loaders take `basepath: str` as the first argument.
- Find files with `glob.glob`, not hardcoded filenames.
- Use `scipy.io.loadmat(filename, simplify_cells=True)` for `.mat` files.
- Warn, do not raise, when an expected file is missing.
- Return an appropriate empty object when data is absent.
- `LFPLoader` should preserve the parent `nel.AnalogSignalArray` interface.
- `loadLFP` is a low-level helper that returns raw `(data, timestamps)`.
- Keep missing-data handling aligned with the nearest loader in the same module.
- Preserve existing output orientation and laziness in `loadLFP`; be explicit about whether arrays are `(n_samples, n_channels)` or channel-selected 1-D outputs.
- Preserve Windows-safe memmap handling and explicit cleanup patterns when touching binary file loading.
- Support both concatenated session files and per-epoch fallback layouts when working on DAT/LFP loading paths.
- Prefer compact synthetic fixtures in `tests/test_io/test_loading.py` that assert types, shapes, timestamps, warnings, and fallback behavior.
- When parsing MATLAB structs, handle scalar fields, optional keys, and inconsistent field shapes defensively rather than assuming one exact layout.
