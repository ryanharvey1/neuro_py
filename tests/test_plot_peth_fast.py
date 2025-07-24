import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from neuro_py.plotting.events import plot_peth_fast
import pandas as pd

def test_plot_peth_fast_runs():
    np.random.seed(42)
    peth = np.random.randn(50, 10)  # 50 time points, 10 trials
    ts = np.linspace(-1, 1, 50)
    # Test with default estimator (mean)
    ax = plot_peth_fast(peth, ts=ts)
    assert isinstance(ax, plt.Axes)
    # Test with median estimator
    ax = plot_peth_fast(peth, ts=ts, estimator=np.nanmedian)
    assert isinstance(ax, plt.Axes)
    # Test with custom estimator (25th percentile)
    ax = plot_peth_fast(peth, ts=ts, estimator=lambda x, axis: np.nanpercentile(x, 25, axis=axis))
    assert isinstance(ax, plt.Axes)
    # Test with DataFrame input
    df = pd.DataFrame(peth, index=ts)
    ax = plot_peth_fast(df)
    assert isinstance(ax, plt.Axes) 