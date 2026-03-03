__all__ = (
    "find_terminal_masked_indices",
    "replace_border_zeros_with_nan",
    "_check_dependency",
    "is_nested",
    "circular_interp",
    "shrink",
    "zscore_columns",
)

from ._dependencies import _check_dependency
from .array import (
    circular_interp,
    find_terminal_masked_indices,
    is_nested,
    replace_border_zeros_with_nan,
    shrink,
    zscore_columns,
)
