"""Run pytest with local defaults that isolate ambient third-party plugins."""

from __future__ import annotations

import os
import subprocess
import sys


def main() -> int:
    """Invoke pytest with repository-friendly environment defaults."""
    env = os.environ.copy()
    env.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    env.setdefault("PYTHONFAULTHANDLER", "1")

    # GitHub Actions can report a very high CPU count, which encourages native
    # libraries and joblib workers to oversubscribe the runner during the full
    # test suite. Keep CI parallelism bounded without affecting normal local
    # development runs.
    if env.get("CI"):
        env.setdefault("LOKY_MAX_CPU_COUNT", "4")
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        env.setdefault("NUMEXPR_NUM_THREADS", "1")
        env.setdefault("VECLIB_MAXIMUM_THREADS", "1")

    command = [sys.executable, "-m", "pytest", *sys.argv[1:]]
    completed = subprocess.run(command, env=env, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
