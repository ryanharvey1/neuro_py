import numpy as np

from neuro_py.ensemble import similarity_index


def test_similarityindex():
    # Test 1: Basic functionality
    patterns = [np.random.normal(0, 0.1, 100) for _ in range(10)]
    si, combos, pvalues = similarity_index(patterns)

    # Check that all similarity indices are between 0 and 1
    assert all(0 <= i <= 1 for i in si), "Similarity indices should be between 0 and 1."

    # Check that the first combination starts with (0, 1)
    assert combos[0, 0] == 0 and combos[0, 1] == 1, (
        "First combination should be (0, 1)."
    )

    # Check that all p-values are between 0 and 1
    assert all(0 <= i <= 1 for i in pvalues), "P-values should be between 0 and 1."

    # Check that the number of combinations matches the number of p-values and similarity indices
    assert combos.shape[0] == pvalues.shape[0] == si.shape[0], (
        "Output shapes should match."
    )

    # Test 2: Cross-group comparisons
    groups = np.hstack(
        [np.ones(5), np.ones(5) + 1]
    )  # First 5 patterns in group 1, next 5 in group 2
    _, combos_cross, _ = similarity_index(patterns, groups=groups)

    # Check that all combinations are cross-group
    assert all(groups[i] != groups[j] for i, j in combos_cross), (
        "All combinations should be cross-group."
    )

    # Check that the number of cross-group combinations is correct
    n_cross = 5 * 5  # 5 patterns in group 1 vs. 5 patterns in group 2
    assert combos_cross.shape[0] == n_cross, (
        f"Expected {n_cross} cross-group combinations."
    )

    # Test 3: P-value adjustment
    _, _, pvalues_adj = similarity_index(patterns, adjust_pvalue=True)

    # Check that p-values are adjusted (should be larger than or equal to original p-values)
    _, _, pvalues_no_adj = similarity_index(patterns, adjust_pvalue=False)
    assert np.mean(pvalues_adj) >= np.mean(pvalues_no_adj), (
        "Adjusted p-values should be larger than or equal to original p-values."
    )

    # Test 4: Edge case - fewer than 2 patterns
    try:
        similarity_index([np.random.normal(0, 0.1, 100)])
    except ValueError as e:
        assert str(e) == "At least 2 patterns are required to compute similarity.", (
            "Should raise ValueError for fewer than 2 patterns."
        )

    # Test 5: Edge case - all patterns identical
    identical_patterns = [np.ones(100) for _ in range(10)]
    si_identical, _, _ = similarity_index(identical_patterns)

    # Check that all similarity indices are 1 (since patterns are identical)
    assert np.allclose(si_identical, 1), (
        "Similarity indices should be 1 for identical patterns."
    )

    # Test 6: Consistency between parallel and non-parallel runs
    si_parallel, combos_parallel, pvalues_parallel = similarity_index(
        patterns, parallel=True
    )
    si_non_parallel, combos_non_parallel, pvalues_non_parallel = similarity_index(
        patterns, parallel=False
    )

    # Check that results are consistent
    assert np.allclose(si_parallel, si_non_parallel), (
        "Similarity indices should match between parallel and non-parallel runs."
    )
    assert np.allclose(combos_parallel, combos_non_parallel), (
        "Combinations should match between parallel and non-parallel runs."
    )
    assert np.allclose(pvalues_parallel, pvalues_non_parallel), (
        "P-values should match between parallel and non-parallel runs."
    )
