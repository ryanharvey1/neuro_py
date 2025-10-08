import numpy as np
import pytest
from scipy import linalg

from neuro_py.ensemble import assembly


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
    with pytest.warns(Warning) as record:
        patterns, significance, zactmat = assembly.runPatterns(
            X, method="ica", nullhyp="mp"
        )
    assert len(record) > 0  # Ensure that at least one warning was raised
    assert "no active neurons" in str(record[0].message)  # Verify the warning message
    assert patterns is None
    assert zactmat is None
    assert significance is None

    # test no patterns found
    X = np.zeros([n_samples, n_features], int).T
    np.fill_diagonal(X, 1)
    with pytest.warns(Warning) as record:
        patterns, significance, zactmat = assembly.runPatterns(
            X, method="ica", nullhyp="mp"
        )
    assert len(record) > 0  # Ensure that at least one warning was raised
    assert "no assembly detected" in str(
        record[0].message
    )  # Verify the warning message
    assert patterns is None
    assert zactmat is None


def test_cross_structural_correlation_matrix():
    """Test the _compute_cross_structural_correlation function."""
    # Create simple test data
    np.random.seed(42)
    zactmat = np.random.randn(5, 100)  # 5 neurons, 100 time bins
    cross_structural = np.array([0, 0, 1, 1, 1])  # 2 neurons in group 0, 3 in group 1

    # Compute cross-structural correlation matrix
    corr_matrix = assembly._compute_cross_structural_correlation(
        zactmat, cross_structural
    )

    # Test matrix properties
    assert corr_matrix.shape == (5, 5)
    assert np.allclose(np.diag(corr_matrix), 1.0)  # Diagonal should be 1

    # Test within-group correlations are zero
    assert corr_matrix[0, 1] == 0  # Both in group 0
    assert corr_matrix[1, 0] == 0  # Both in group 0
    assert corr_matrix[2, 3] == 0  # Both in group 1
    assert corr_matrix[3, 4] == 0  # Both in group 1
    assert corr_matrix[4, 2] == 0  # Both in group 1

    # Test cross-group correlations are preserved (not zero)
    cross_group_corrs = [
        corr_matrix[0, 2],
        corr_matrix[0, 3],
        corr_matrix[0, 4],
        corr_matrix[1, 2],
        corr_matrix[1, 3],
        corr_matrix[1, 4],
    ]
    # At least some cross-group correlations should be non-zero
    assert any(abs(c) > 0.01 for c in cross_group_corrs)


def test_cross_structural_assemblies_synthetic():
    """Test cross-structural assembly detection with synthetic data."""
    np.random.seed(42)

    # Create synthetic data with cross-structural assemblies
    n_neurons_group1 = 15
    n_neurons_group2 = 10
    n_bins = 1000
    base_rate = 1.0

    # Create activity matrix
    actmat = np.random.poisson(base_rate, (n_neurons_group1 + n_neurons_group2, n_bins))

    # Add within-group assembly (should be detected by standard but not cross-structural)
    within_group_neurons = np.arange(5, 10)  # all from group 1
    within_group_times = np.sort(np.random.choice(n_bins, 200, replace=False))

    for t in within_group_times:
        actmat[within_group_neurons, t] += np.random.poisson(
            4, len(within_group_neurons)
        )

    # Add cross-structural assembly
    cross_neurons_g1 = np.arange(12, 15)  # from group 1
    cross_neurons_g2 = np.arange(
        n_neurons_group1 + 3, n_neurons_group1 + 6
    )  # from group 2
    cross_times = np.sort(np.random.choice(n_bins, 180, replace=False))

    for t in cross_times:
        actmat[cross_neurons_g1, t] += np.random.poisson(4, len(cross_neurons_g1))
        actmat[cross_neurons_g2, t] += np.random.poisson(4, len(cross_neurons_g2))

    # Create group labels
    cross_structural = np.array(
        ["Group1"] * n_neurons_group1 + ["Group2"] * n_neurons_group2
    )

    # Test standard detection
    patterns_std, _, _ = assembly.runPatterns(actmat, method="ica", nullhyp="mp")

    # Test cross-structural detection
    patterns_cross, _, _ = assembly.runPatterns(
        actmat, method="ica", nullhyp="mp", cross_structural=cross_structural
    )

    # Cross-structural should detect fewer assemblies (filtering out within-group)
    if patterns_std is not None and patterns_cross is not None:
        assert patterns_cross.shape[0] <= patterns_std.shape[0]

        # Check that detected assemblies are actually cross-structural
        for pattern in patterns_cross:
            group1_weights = np.abs(pattern[:n_neurons_group1])
            group2_weights = np.abs(pattern[n_neurons_group1:])
            group1_active = np.sum(group1_weights > 0.1)
            group2_active = np.sum(group2_weights > 0.1)
            # Should have active neurons in both groups
            assert group1_active > 0 and group2_active > 0


def test_cross_structural_pca_vs_ica():
    """Test that both PCA and ICA methods work with cross-structural detection."""
    np.random.seed(42)

    # Create simple synthetic data
    n_neurons = 20
    n_bins = 500
    actmat = np.random.poisson(2, (n_neurons, n_bins))

    # Add cross-structural assembly
    group1_neurons = np.arange(5, 8)
    group2_neurons = np.arange(15, 18)
    assembly_times = np.sort(np.random.choice(n_bins, 100, replace=False))

    for t in assembly_times:
        actmat[group1_neurons, t] += np.random.poisson(3, len(group1_neurons))
        actmat[group2_neurons, t] += np.random.poisson(3, len(group2_neurons))

    cross_structural = np.array(["A"] * 10 + ["B"] * 10)

    # Test both methods
    patterns_pca, _, _ = assembly.runPatterns(
        actmat, method="pca", nullhyp="mp", cross_structural=cross_structural
    )
    patterns_ica, _, _ = assembly.runPatterns(
        actmat, method="ica", nullhyp="mp", cross_structural=cross_structural
    )

    # Both should detect assemblies
    assert patterns_pca is not None or patterns_ica is not None

    # If both detect assemblies, they should have the same number of patterns
    if patterns_pca is not None and patterns_ica is not None:
        assert patterns_pca.shape[0] == patterns_ica.shape[0]


def test_cross_structural_validation():
    """Test parameter validation for cross-structural detection."""
    np.random.seed(42)
    actmat = np.random.poisson(1, (10, 100))

    # Test wrong length cross_structural vector
    cross_structural_wrong = np.array(["A"] * 5)  # Should be length 10

    with pytest.raises(ValueError, match="cross_structural length"):
        assembly.runPatterns(actmat, cross_structural=cross_structural_wrong)

    # Test correct length
    cross_structural_correct = np.array(["A"] * 5 + ["B"] * 5)
    patterns, _, _ = assembly.runPatterns(
        actmat, cross_structural=cross_structural_correct
    )
    # Should not raise an error


def test_cross_structural_silent_neurons():
    """Test cross-structural detection with silent neurons."""
    np.random.seed(42)

    # Create data with some silent neurons
    n_neurons = 15
    n_bins = 500
    actmat = np.random.poisson(1, (n_neurons, n_bins))

    # Make some neurons silent
    actmat[3, :] = 0  # Silent neuron in group 1
    actmat[12, :] = 0  # Silent neuron in group 2

    cross_structural = np.array(["A"] * 8 + ["B"] * 7)

    # Should handle silent neurons gracefully
    patterns, _, _ = assembly.runPatterns(
        actmat, method="ica", nullhyp="mp", cross_structural=cross_structural
    )

    # Check that patterns have correct shape (accounting for original neuron count)
    if patterns is not None:
        assert patterns.shape[1] == n_neurons


def test_cross_structural_no_cross_components():
    """Test behavior when no cross-structural components are found."""
    np.random.seed(42)

    # Create data with only within-group correlations
    n_neurons = 10
    n_bins = 200
    actmat = np.random.poisson(1, (n_neurons, n_bins))

    # Add strong within-group assembly but no cross-group
    group1_neurons = np.arange(0, 5)
    assembly_times = np.sort(np.random.choice(n_bins, 50, replace=False))

    for t in assembly_times:
        actmat[group1_neurons, t] += np.random.poisson(5, len(group1_neurons))

    cross_structural = np.array(["A"] * 5 + ["B"] * 5)

    # Should handle case with no cross-structural assemblies
    patterns, _, _ = assembly.runPatterns(
        actmat, method="ica", nullhyp="mp", cross_structural=cross_structural
    )

    # Assert: Either no patterns detected, or all detected patterns are not cross-structural
    if patterns is None:
        assert True  # No cross-structural assemblies detected, as expected
    else:
        # All detected patterns should be within-group only (not cross-structural)
        for pattern in patterns:
            group1_weights = np.abs(pattern[:5])
            group2_weights = np.abs(pattern[5:])
            group1_active = np.sum(group1_weights > 0.1)
            group2_active = np.sum(group2_weights > 0.1)
            # Assert that at least one group is inactive (not cross-structural)
            assert group1_active == 0 or group2_active == 0
