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
