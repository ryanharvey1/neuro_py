import numpy as np
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
