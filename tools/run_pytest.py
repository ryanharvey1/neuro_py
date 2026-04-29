"""Run pytest with local defaults that make editable ``nelpy`` test runs reliable."""

from __future__ import annotations

import os
import subprocess
import sys


def main() -> int:
    """Invoke pytest with repository-friendly environment defaults."""
    env = os.environ.copy()
    env.setdefault("NUMBA_DISABLE_JIT", "1")
    env.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    env.setdefault("MPLBACKEND", "Agg")

    command = [sys.executable, "-m", "pytest", *sys.argv[1:]]
    completed = subprocess.run(command, env=env, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
