from neuro_py.raw import spike_sorting


def test_spike_sorting_module_imports_without_ipython_display_import():
    assert callable(spike_sorting.spike_sorting_progress)
    assert callable(spike_sorting.phy_log_to_epocharray)
