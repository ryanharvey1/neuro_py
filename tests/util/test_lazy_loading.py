import sys


def test_lazy_loading():
    # clear all neuro_py submodules
    for module in list(sys.modules.keys()):
        if module.startswith("neuro_py"):
            sys.modules.pop(module)

    import neuro_py

    npy_submodules = set(
        x.split(".")[1]
        for x in sys.modules.keys()
        if (
            x.startswith("neuro_py.")
            and "__" not in x
            and not x.split(".")[1].startswith("_")
            and sys.modules[x] is not None
        )
    )

    # ensure that no submodules have been imported yet
    assert npy_submodules == set(), f"Submodules already loaded: {npy_submodules}"

    # ensure that only the accessed submodule has been imported
    neuro_py.behavior
    npy_submodules = set(
        x.split(".")[1]
        for x in sys.modules.keys()
        if (
            x.startswith("neuro_py.")
            and "__" not in x
            and not x.split(".")[1].startswith("_")
            and sys.modules[x] is not None
        )
    )
    assert npy_submodules == {"behavior"}, (
        f"Submodules already loaded: {npy_submodules}"
    )
