# neuro_py Agent Guide

Use this file as the Codex entry point for repository-specific instructions.

Primary guidance lives in [.github/copilot-instructions.md](.github/copilot-instructions.md). Those instructions are intentionally written to work for GitHub Copilot, Codex, and similar coding agents.

Additional scoped guidance:

- When editing `neuro_py/io/loading.py`, also follow `.github/instructions/io-loading.instructions.md`.

Working rules for Codex:

- Prefer the smallest targeted change that fixes the root cause.
- Preserve public APIs unless the task explicitly requires a breaking change.
- Add or update a regression test in the matching `tests/` area when behavior changes.
- Use type hints on function signatures and numpydoc docstrings for public functions.
- Run the narrowest relevant pytest target you can before finishing, and clearly report if you could not run tests.
- Preserve scientific behavior, units, shapes, and `nelpy` semantics unless the task explicitly requires a change.
- For loader work, follow `.github/instructions/io-loading.instructions.md` and keep missing-data behavior aligned with nearby loaders.
- Prefer vectorized changes and avoid unnecessarily materializing large arrays or lazy loader views.
