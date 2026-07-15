import pytest

from neuro_py.raw import spike_sorting


def test_spike_sorting_module_imports_without_ipython_display_import():
    assert callable(spike_sorting.spike_sorting_progress)
    assert callable(spike_sorting.phy_log_to_epocharray)


def test_spike_sorting_progress_reports_missing_ipython(monkeypatch):
    def raise_missing_ipython(_: str):
        raise ModuleNotFoundError("No module named 'IPython'")

    monkeypatch.setattr(spike_sorting, "import_module", raise_missing_ipython)

    with pytest.raises(ImportError, match="requires IPython"):
        spike_sorting.spike_sorting_progress("cluster_info.tsv")
