"""Run pytest with local defaults that isolate ambient third-party plugins."""

from __future__ import annotations

import os
import subprocess
import sys


def main() -> int:
    """Invoke pytest with repository-friendly environment defaults."""
    env = os.environ.copy()
    env.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")

    command = [sys.executable, "-m", "pytest", *sys.argv[1:]]
    completed = subprocess.run(command, env=env, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
