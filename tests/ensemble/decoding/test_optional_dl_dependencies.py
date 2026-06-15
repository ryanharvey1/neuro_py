import builtins
import importlib
import sys
from unittest.mock import patch

import pytest


DL_MODULES = {"torch", "lightning", "tensorboard"}
_ORIGINAL_IMPORT = builtins.__import__


def _clear_neuro_py_modules() -> None:
    for module in list(sys.modules):
        if module.startswith("neuro_py"):
            sys.modules.pop(module)


def _mocked_import(name, globals=None, locals=None, fromlist=(), level=0):
    root_name = name.split(".")[0]
    if root_name in DL_MODULES:
        raise ImportError(f"{root_name} intentionally unavailable during test")
    return _ORIGINAL_IMPORT(name, globals, locals, fromlist, level)


def test_decoding_imports_without_dl_dependencies() -> None:
    _clear_neuro_py_modules()

    with patch("builtins.__import__", side_effect=_mocked_import):
        neuro_py = importlib.import_module("neuro_py")
        ensemble = importlib.import_module("neuro_py.ensemble")
        decoding = importlib.import_module("neuro_py.ensemble.decoding")
        bayesian = importlib.import_module("neuro_py.ensemble.decoding.bayesian")

    assert neuro_py is not None
    assert ensemble is not None
    assert decoding is not None
    assert bayesian.decode is not None


def test_dl_entrypoint_requires_dl_extra() -> None:
    _clear_neuro_py_modules()

    with patch("builtins.__import__", side_effect=_mocked_import):
        decoding = importlib.import_module("neuro_py.ensemble.decoding")
        with pytest.raises(ImportError, match=r"\[dl\]"):
            decoding.MLP
