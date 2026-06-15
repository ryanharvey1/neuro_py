from __future__ import annotations

import importlib
from typing import Any

from ...util._dependencies import _check_dependency
from .bayesian import decode
from .pipeline import (
    format_trial_segs_nsv,
    get_spikes_with_history,
    normalize_format_trial_segs,
    normalize_labels,
    shuffle_nsv_intrialsegs,
    zscore_trial_segs,
)
from .preprocess import partition_indices, partition_sets, split_data

__all__ = [
    "MLP",
    "LSTM",
    "M2MLSTM",
    "NDT",
    "seed_worker",
    "get_spikes_with_history",
    "format_trial_segs_nsv",
    "zscore_trial_segs",
    "normalize_format_trial_segs",
    "minibatchify",
    "normalize_labels",
    "create_model",
    "preprocess_data",
    "evaluate_model",
    "shuffle_nsv_intrialsegs",
    "train_model",
    "predict_models_folds",
    "split_data",
    "partition_indices",
    "partition_sets",
    "decode",
]

_DL_EXPORTS = {
    "MLP": (".mlp", "MLP"),
    "LSTM": (".lstm", "LSTM"),
    "M2MLSTM": (".m2mlstm", "M2MLSTM"),
    "NDT": (".transformer", "NDT"),
    "seed_worker": (".pipeline", "seed_worker"),
    "minibatchify": (".pipeline", "minibatchify"),
    "create_model": (".pipeline", "create_model"),
    "preprocess_data": (".pipeline", "preprocess_data"),
    "evaluate_model": (".pipeline", "evaluate_model"),
    "train_model": (".pipeline", "train_model"),
    "predict_models_folds": (".pipeline", "predict_models_folds"),
}
_TENSORBOARD_EXPORTS = {"train_model"}


def _load_dl_export(name: str) -> Any:
    _check_dependency("torch", "dl")
    _check_dependency("lightning", "dl")
    module_name, attr_name = _DL_EXPORTS[name]
    if name in _TENSORBOARD_EXPORTS:
        _check_dependency("tensorboard", "dl")

    module = importlib.import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __getattr__(name: str) -> Any:
    if name in _DL_EXPORTS:
        return _load_dl_export(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
