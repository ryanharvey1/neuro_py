from neuro_py.ensemble import assembly
import numpy as np
from scipy import linalg


def test_assembly():
    # make a random matrix with rank 5
    n_samples, n_features, rank = 1000, 25, 5
    rng = np.random.RandomState(42)
    U, _, _ = linalg.svd(rng.randn(n_features, n_features))
    X = np.dot(rng.randn(n_samples, rank), U[:, :rank].T).T

    # test runPatterns ICA, MP (marcenkopastur)
    patterns, significance, zactmat = assembly.runPatterns(
        X, method="ica", nullhyp="mp"
    )
    assert patterns.shape[0] == rank
    assert X.shape == zactmat.shape

    # test runPatterns ICA, bin shuffle
    patterns, significance, zactmat = assembly.runPatterns(
        X, method="ica", nullhyp="bin", nshu=100
    )
    assert patterns.shape[0] == rank
    assert X.shape == zactmat.shape

    # test runPatterns ICA, circ shuffle
    patterns, significance, zactmat = assembly.runPatterns(
        X, method="ica", nullhyp="circ", nshu=100
    )
    assert patterns.shape[0] == rank
    assert X.shape == zactmat.shape

    # test runPatterns PCA, MP (marcenkopastur)
    patterns, significance, zactmat = assembly.runPatterns(
        X, method="pca", nullhyp="mp"
    )
    assert patterns.shape[0] == rank
    assert X.shape == zactmat.shape

    # test runPatterns PCA, bin shuffle
    patterns, significance, zactmat = assembly.runPatterns(
        X, method="pca", nullhyp="bin", nshu=100
    )
    assert patterns.shape[0] == rank
    assert X.shape == zactmat.shape

    # test runPatterns PCA, circ shuffle
    patterns, significance, zactmat = assembly.runPatterns(
        X, method="pca", nullhyp="circ", nshu=100
    )
    assert patterns.shape[0] == rank
    assert X.shape == zactmat.shape

    # test computeAssemblyActivity
    patterns, significance, zactmat = assembly.runPatterns(
        X, method="ica", nullhyp="mp"
    )
    assemblyAct = assembly.computeAssemblyActivity(patterns, zactmat, zerodiag=True)
    assert assemblyAct.shape == (rank, X.shape[1])

    # test no active cells found
    n_samples, n_features, rank = 1000, 25, 0
    rng = np.random.RandomState(42)
    U, _, _ = linalg.svd(rng.randn(n_features, n_features))
    X = np.dot(rng.randn(n_samples, rank), U[:, :rank].T).T
    patterns, significance, zactmat = assembly.runPatterns(X,method="ica",nullhyp="mp")
    assert patterns is None
    assert zactmat is None
    assert significance is None

    # test no patterns found
    X = np.zeros([n_samples, n_features],int).T
    np.fill_diagonal(X, 1)
    patterns, significance, zactmat = assembly.runPatterns(X,method="ica",nullhyp="mp")
    assert patterns is None
    assert zactmat is None