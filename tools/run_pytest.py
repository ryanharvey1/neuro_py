"""Run pytest with local defaults that isolate ambient third-party plugins."""

from __future__ import annotations

import os
import subprocess
import sys


def _run_pytest(args: list[str], env: dict[str, str]) -> int:
    """Run pytest with the provided arguments and environment."""
    command = [sys.executable, "-m", "pytest", *args]
    completed = subprocess.run(command, env=env, check=False)
    return completed.returncode


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

    user_args = sys.argv[1:]

    # On GitHub Actions, `tests/process/test_batch_analysis.py` can segfault
    # late in the full suite after many C-extension-heavy tests have already
    # run. Executing it in a fresh subprocess keeps the rest of the suite and
    # the batch-analysis coverage intact without changing production behavior.
    if env.get("CI") and not user_args:
        first_phase = _run_pytest(["--ignore=tests/process/test_batch_analysis.py"], env)
        if first_phase != 0:
            return first_phase
        return _run_pytest(["tests/process/test_batch_analysis.py"], env)

    return _run_pytest(user_args, env)


if __name__ == "__main__":
    raise SystemExit(main())
