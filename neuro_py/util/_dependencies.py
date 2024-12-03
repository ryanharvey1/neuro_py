def _check_dependency(module_name: str, extra: str) -> None:
    """
    Check if a module is installed, and raise an ImportError with a helpful message if not.

    Parameters
    ----------
    module_name : str
        The name of the module to check.
    extra : str
        The name of the extra requirement group (e.g., 'csd') to suggest in the error message.
    """
    try:
        __import__(module_name)
    except ImportError:
        raise ImportError(
            f"{module_name} is not installed. Please install it to use this function. "
            f"Run: pip install -e .[{extra}]"
        )