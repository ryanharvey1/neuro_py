import numpy as np
import pytest

from neuro_py.stats.stats import get_significant_events


def test_get_significant_events():
    # Test case 1:
    shuffled_scores = np.random.rand(1000, 10)
    scores = np.random.rand(1, 10)[0]
    scores[2] = 4
    q = 95
    tail = "both"
    sig_event_idx, pvalues, stddev = get_significant_events(
        scores, shuffled_scores, q=q, tail=tail
    )
    assert 2 in sig_event_idx
    assert pvalues[2] < 0.05
    assert stddev[2] > 8


@pytest.mark.parametrize("tail", ["both", "right", "left"])
def test_nonfinite_shuffles_do_not_inflate_significance(tail):
    shuffles = np.full((100, 1), np.nan)
    shuffles[0, 0] = 0 if tail == "left" else 2
    shuffles[1:3, 0] = [-np.inf, np.inf]
    indices, pvalues, _ = get_significant_events([1.0], shuffles, tail=tail)
    np.testing.assert_array_equal(pvalues, [1])
    assert indices.size == 0


@pytest.mark.parametrize("tail", ["both", "right", "left"])
def test_each_event_uses_its_own_finite_null_distribution(tail):
    scores = np.array([2.5, -2.5, 0.3])
    shuffles = np.array(
        [
            [0, -1, 1],
            [1, np.nan, np.inf],
            [2, -3, 0],
            [3, -2, np.nan],
            [np.nan, np.nan, -1],
        ],
        dtype=float,
    )
    original = shuffles.copy()
    indices, pvalues, zscores = get_significant_events(
        scores, shuffles, q=80, tail=tail
    )
    expected_indices = []
    for event in range(3):
        finite = shuffles[np.isfinite(shuffles[:, event]), event, None]
        wanted_indices, wanted_p, wanted_z = get_significant_events(
            [scores[event]], finite, q=80, tail=tail
        )
        if wanted_indices.size:
            expected_indices.append(event)
        assert pvalues[event] == pytest.approx(wanted_p[0])
        assert zscores[event] == pytest.approx(wanted_z[0])
    np.testing.assert_array_equal(indices, expected_indices)
    np.testing.assert_array_equal(shuffles, original)


def test_ties_count_as_exceedances_and_pvalues_use_finite_sample_correction():
    _, pvalues, _ = get_significant_events(
        [2.0, 1.0], np.array([[0, 0], [1, np.nan], [2, 1], [3, np.nan]]), tail="right"
    )
    np.testing.assert_allclose(pvalues, [3 / 5, 2 / 3])


@pytest.mark.parametrize("tail", ["both", "right", "left"])
def test_untestable_events_are_not_significant(tail):
    scores = np.array([np.nan, np.inf, -np.inf, 5])
    shuffles = np.zeros((10, 4))
    shuffles[:, 3] = np.nan
    indices, pvalues, zscores = get_significant_events(scores, shuffles, tail=tail)
    assert indices.size == 0
    np.testing.assert_array_equal(pvalues, np.ones(4))
    assert np.isnan(zscores).all()


def test_empty_null_distribution_has_no_evidence_of_significance():
    indices, pvalues, zscores = get_significant_events([1, 2], np.empty((0, 2)))
    assert indices.size == 0
    np.testing.assert_array_equal(pvalues, [1, 1])
    assert np.isnan(zscores).all()


def test_transposed_and_single_event_inputs_are_preserved():
    shuffles = np.arange(15).reshape(5, 3)
    normal = get_significant_events([1, 2, 3], shuffles)
    transposed = get_significant_events([1, 2, 3], shuffles.T)
    for actual, expected in zip(transposed, normal):
        np.testing.assert_allclose(actual, expected)
    one = get_significant_events(np.array(1.0), np.arange(5))
    np.testing.assert_allclose(one[1], [5 / 6])


@pytest.mark.parametrize("q", [-1, 101, np.nan])
def test_invalid_percentiles_are_rejected(q):
    with pytest.raises(ValueError, match="q must"):
        get_significant_events([1], np.ones((5, 1)), q=q)


def test_incompatible_event_dimensions_are_rejected():
    with pytest.raises(ValueError, match="one column per"):
        get_significant_events([1, 2], np.ones((4, 3)))
