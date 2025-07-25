import numpy as np
import pandas as pd

from neuro_py.stats import circ_stats as ncirc
from neuro_py.stats import regression as nreg
from neuro_py.stats import stats as nstats


def test_reindex_df():
    df = pd.DataFrame({"a": [1, 2, 3], "w": [2, 1, 3]})
    expanded = nstats.reindex_df(df, "w")
    assert isinstance(expanded, pd.DataFrame)
    assert len(expanded) == sum(df["w"])
    assert set(expanded.columns) == set(df.columns)


def test_regress_out():
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([2, 4, 6, 8, 10])
    out = nstats.regress_out(a, b)
    assert out.shape == a.shape
    # Output should be finite and constant (since a and b are perfectly collinear)
    assert np.all(np.isfinite(out))
    assert np.allclose(out, out[0])


def test_ideal_data():
    X, Y = nreg.ideal_data(100, 5, 3, 2)
    assert X.shape == (100, 5)
    assert Y.shape == (100, 3)


def test_ReducedRankRegressor():
    X, Y = nreg.ideal_data(100, 5, 3, 2)
    model = nreg.ReducedRankRegressor(X, Y, rank=2)
    Y_pred = model.predict(X)
    assert Y_pred.shape == Y.shape
    score = model.score(X, Y)
    assert isinstance(score, float)


def test_MultivariateRegressor():
    X, Y = nreg.ideal_data(100, 5, 3, 2)
    model = nreg.MultivariateRegressor(X, Y)
    Y_pred = model.predict(X)
    assert Y_pred.shape == Y.shape
    score = model.score(X, Y)
    assert isinstance(score, float)


def test_kernelReducedRankRegressor():
    X, Y = nreg.ideal_data(30, 4, 2, 2)
    model = nreg.kernelReducedRankRegressor(rank=1, reg=0.1)
    model.fit(X, Y)
    Y_pred = model.predict(X)
    assert Y_pred.shape == Y.shape
    mse = model.mse(X, Y)
    assert isinstance(mse, float)
    score = model.score(X, Y)
    assert isinstance(score, float)


def test_resultant_vector_length():
    alpha = np.linspace(0, 2 * np.pi, 100)
    r = ncirc.resultant_vector_length(alpha)
    assert isinstance(r, float) or isinstance(r, np.floating)


def test_mean():
    # Use a concentrated sample for valid CI
    alpha = np.random.vonmises(mu=0, kappa=4, size=100)
    mu = ncirc.mean(alpha)
    assert np.isscalar(mu)
    mu, ci = ncirc.mean(alpha, ci=0.95)
    assert isinstance(ci, tuple) or hasattr(ci, "lower")


def test_center():
    alpha = np.linspace(0, 2 * np.pi, 100)
    centered = ncirc.center(alpha)
    assert centered.shape == alpha.shape


def test_rayleigh():
    alpha = np.linspace(0, 2 * np.pi, 100)
    pval, z = ncirc.rayleigh(alpha)
    assert isinstance(pval, float) or isinstance(pval, np.floating)
    assert isinstance(z, float) or isinstance(z, np.floating)
