import numpy as np
import pytest
from scipy import linalg

from neuro_py.ensemble import assembly

NEURON_ACTIVITY_THRESHOLD = (
    0.1  # Threshold for determining active neurons in assemblies
)


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


def test_cross_structural_covariance_matrix():
    """Test the _compute_cross_structural_covariance function."""
    # Create simple test data
    np.random.seed(42)
    zactmat = np.random.randn(5, 100)  # 5 neurons, 100 time bins
    cross_structural = np.array([0, 0, 1, 1, 1])  # 2 neurons in group 0, 3 in group 1

    # Compute cross-structural covariance matrix
    cov_matrix = assembly._compute_cross_structural_covariance(
        zactmat, cross_structural
    )

    # Test matrix properties
    assert cov_matrix.shape == (5, 5)
    assert np.allclose(np.diag(cov_matrix), 0.0)  # Diagonal should be 0
    assert np.allclose(cov_matrix, cov_matrix.T)  # Should be symmetric

    # Test within-group covariances are zero
    assert cov_matrix[0, 1] == 0  # Both in group 0
    assert cov_matrix[1, 0] == 0  # Both in group 0
    assert cov_matrix[2, 3] == 0  # Both in group 1
    assert cov_matrix[3, 4] == 0  # Both in group 1
    assert cov_matrix[4, 2] == 0  # Both in group 1

    # Test cross-group covariances are preserved (not zero)
    cross_group_covs = [
        cov_matrix[0, 2],
        cov_matrix[0, 3],
        cov_matrix[0, 4],
        cov_matrix[1, 2],
        cov_matrix[1, 3],
        cov_matrix[1, 4],
    ]
    # At least some cross-group covariances should be non-zero
    assert any(abs(c) > 0.01 for c in cross_group_covs)


def test_cross_structural_group_normalization():
    """Test group-size normalization for cross-structural analysis."""
    zactmat = np.ones((5, 4), dtype=float)
    cross_structural = np.array(["A", "A", "B", "B", "B"])

    normalized = assembly._normalize_by_group(zactmat, cross_structural)

    assert np.allclose(normalized[:2], 1 / np.sqrt(2))
    assert np.allclose(normalized[2:], 1 / np.sqrt(3))


def test_filter_cross_group_patterns():
    """Test filtering keeps only patterns active in at least two groups."""
    patterns = np.array(
        [
            [1.0, 0.0, 0.5, 0.0],  # active in both groups
            [1.0, 0.5, 0.0, 0.0],  # active only in first group
            [0.0, 0.0, 0.4, 0.2],  # active only in second group
        ]
    )
    cross_structural = np.array(["G1", "G1", "G2", "G2"])

    filtered = assembly._filter_cross_group_patterns(patterns, cross_structural)
    assert filtered.shape[0] == 1
    assert np.allclose(filtered[0], patterns[0])


def test_filter_cross_group_patterns_relative_threshold_removes_tiny_weights():
    """Relative threshold should ignore tiny numerical weights in a group."""
    patterns = np.array(
        [
            [1.0, 0.8, 1e-10, 0.0],  # tiny noise in second group
            [1.0, 0.8, 0.2, 0.1],  # meaningful weights in both groups
        ]
    )
    cross_structural = np.array(["G1", "G1", "G2", "G2"])

    filtered = assembly._filter_cross_group_patterns(
        patterns,
        cross_structural,
        threshold=1e-3,
        threshold_mode="relative",
    )

    assert filtered.shape[0] == 1
    assert np.allclose(filtered[0], patterns[1])


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
            group1_active = np.sum(group1_weights > NEURON_ACTIVITY_THRESHOLD)
            group2_active = np.sum(group2_weights > NEURON_ACTIVITY_THRESHOLD)
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

    # If both detect assemblies, they should have the same number of patterns
    if patterns_pca is not None and patterns_ica is not None:
        assert patterns_pca.shape[0] == patterns_ica.shape[0]

    # Otherwise, stricter cross-structural filtering may remove weak candidates
    assert patterns_pca is None or patterns_pca.shape[1] == n_neurons
    assert patterns_ica is None or patterns_ica.shape[1] == n_neurons


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


def test_cross_structural_mp_fallbacks_to_bin():
    """Cross-structural mode should replace MP null with bin shuffling."""
    np.random.seed(42)
    actmat = np.random.poisson(1, (12, 120))
    cross_structural = np.array(["A"] * 6 + ["B"] * 6)

    _, significance, _ = assembly.runPatterns(
        actmat, method="ica", nullhyp="mp", nshu=50, cross_structural=cross_structural
    )

    assert significance is not None
    assert significance.nullhyp == "bin"


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

    # No genuine cross-structural assemblies should be detected
    assert patterns is None


def test_cross_svd_detects_cross_area_assembly():
    """cross_svd should detect strong cross-area coupling."""
    np.random.seed(42)
    n_group1 = 8
    n_group2 = 8
    n_bins = 800

    actmat = np.random.poisson(1, (n_group1 + n_group2, n_bins))
    assembly_times = np.sort(np.random.choice(n_bins, 220, replace=False))

    for t in assembly_times:
        actmat[2:5, t] += np.random.poisson(8, 3)
        actmat[n_group1 + 3 : n_group1 + 6, t] += np.random.poisson(8, 3)

    groups = np.array(["CA1"] * n_group1 + ["PFC"] * n_group2)
    patterns, significance, zactmat = assembly.runPatterns(
        actmat,
        method="cross_svd",
        cross_structural=groups,
        nshu=100,
        percentile=95,
    )

    assert significance is not None
    assert significance.nullhyp == "bin"
    assert patterns is not None
    assert patterns.shape[0] >= 1
    assert patterns.shape[1] == n_group1 + n_group2
    assert zactmat.shape == actmat.shape


def test_cross_svd_requires_two_groups():
    """cross_svd should fail when cross_structural has more than two groups."""
    np.random.seed(42)
    actmat = np.random.poisson(1, (9, 120))
    groups = np.array(["A"] * 3 + ["B"] * 3 + ["C"] * 3)

    with pytest.raises(ValueError, match="exactly two groups"):
        assembly.runPatterns(actmat, method="cross_svd", cross_structural=groups)


def test_compute_cross_area_activity_shape():
    """computeCrossAreaActivity returns expected shape."""
    np.random.seed(42)
    patterns = np.array(
        [
            [0.7, 0.0, 0.7, 0.0],
            [0.0, 0.7, 0.0, 0.7],
        ]
    )
    zactmat = np.random.randn(4, 50)
    groups = np.array(["CA1", "CA1", "PFC", "PFC"])

    activity = assembly.computeCrossAreaActivity(patterns, zactmat, groups)
    assert activity is not None
    assert activity.shape == (2, 50)
