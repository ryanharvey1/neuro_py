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
