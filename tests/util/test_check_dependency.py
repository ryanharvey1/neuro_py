import pytest

# Assuming _check_dependency is in a file named dependencies.py in the utils submodule
from neuro_py.util._dependencies import _check_dependency


def test_check_dependency_success():
    """Test that _check_dependency does not raise an error for an installed module."""
    # Example of a commonly installed module, 'math'
    try:
        _check_dependency("math", "test")
    except ImportError:
        pytest.fail("Unexpected ImportError for a module that should be installed.")


def test_check_dependency_failure():
    """Test that _check_dependency raises an ImportError for a non-existent module."""
    non_existent_module = "some_non_existent_module"
    extra_name = "test"
    with pytest.raises(ImportError, match=f"{non_existent_module} is not installed."):
        _check_dependency(non_existent_module, extra_name)


def test_check_dependency_preserves_nested_import_errors(monkeypatch):
    """Nested import failures inside a dependency should surface the original error."""

    def _raise_nested_module_not_found(name, *args, **kwargs):
        if name == "some_installed_module":
            raise ModuleNotFoundError("missing nested import", name="nested_dependency")
        return __import__(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _raise_nested_module_not_found)

    with pytest.raises(ModuleNotFoundError, match="missing nested import"):
        _check_dependency("some_installed_module", "test")
