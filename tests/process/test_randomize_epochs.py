import nelpy as nel
import numpy as np
import pytest

from neuro_py.process.intervals import randomize_epochs


def _mock_uniform_factory(values):
    """
    Build a mock for np.random.uniform that returns deterministic values.
    - If size is None: return scalar values[0]
    - If size is provided: return np.full(size, values[0]) when values is scalar-like,
      or np.array(values) if len(values) == size
    """

    def _mock_uniform(low, high, size=None):
        if size is None:
            # single shift value
            if isinstance(values, (list, tuple, np.ndarray)):
                return float(values[0])
            return float(values)
        # size provided
        if isinstance(values, (list, tuple, np.ndarray)):
            arr = np.asarray(values, dtype=float)
            if arr.size == 1:
                return np.full(size, arr.item())
            return arr
        return np.full(size, float(values))

    return _mock_uniform


def test_single_interval_no_wrap(monkeypatch):
    # shift by +3 within [0,10] support
    monkeypatch.setattr(np.random, "uniform", _mock_uniform_factory(3.0))

    epoch = nel.EpochArray([(2.0, 4.0), (6.0, 7.0)])
    out = randomize_epochs(epoch, randomize_each=False, start_stop=[0.0, 10.0])

    expected = np.array([[5.0, 7.0], [9.0, 10.0]])
    assert np.allclose(out.data, expected)
    # durations preserved
    assert np.allclose(out.lengths, epoch.lengths)
    # all within support
    assert out.starts.min() >= 0.0 and out.stops.max() <= 10.0


def test_single_interval_with_wrap(monkeypatch):
    # shift by +3 creates wrap for interval near the end
    monkeypatch.setattr(np.random, "uniform", _mock_uniform_factory(3.0))

    epoch = nel.EpochArray([(8.0, 9.0)])
    out = randomize_epochs(epoch, randomize_each=False, start_stop=[0.0, 10.0])

    expected = np.array([[1.0, 2.0]])
    assert np.allclose(out.data, expected)
    assert np.allclose(out.lengths, epoch.lengths)


def test_multi_interval_split_across_gap(monkeypatch):
    # Support: two disjoint intervals with total length 10
    support = np.array([[0.0, 5.0], [10.0, 15.0]])

    # Choose an input within the second segment such that after shift 0.5
    # it crosses the end of linearized support and splits across segments
    # Linearized second segment maps to [5,10). interval: lin ~ [8.5, 9.7]
    # After +0.5 => [9.0, 10.2] -> split into [9.0, 10.0] and [0.0, 0.2]
    # Map back => abs: [14.0, 15.0] and [0.0, 0.2]
    monkeypatch.setattr(np.random, "uniform", _mock_uniform_factory(0.5))

    epoch = nel.EpochArray([(13.5, 14.7)])
    out = randomize_epochs(epoch, randomize_each=False, start_stop=support)

    # Expect two intervals, sorted by start
    expected = np.array([[0.0, 0.2], [14.0, 15.0]])
    assert np.allclose(out.data, expected)
    assert np.allclose(out.lengths.sum(), epoch.lengths.sum())
    # Ensure output lies within the multi-interval support
    for s, e in out.data:
        assert (0.0 <= s <= 5.0 and 0.0 <= e <= 5.0) or (
            10.0 <= s <= 15.0 and 10.0 <= e <= 15.0
        )


def test_randomize_each_independent_shifts(monkeypatch):
    # Provide two distinct shifts per interval in single support
    monkeypatch.setattr(np.random, "uniform", _mock_uniform_factory([1.0, 2.0]))

    epoch = nel.EpochArray([(1.0, 2.0), (3.0, 4.0)])
    out = randomize_epochs(epoch, randomize_each=True, start_stop=[0.0, 10.0])

    # First interval +1 => [2,3], second +2 => [5,6]
    # Order already sorted
    expected = np.array([[2.0, 3.0], [5.0, 6.0]])
    assert np.allclose(out.data, expected)
    assert np.allclose(out.lengths, epoch.lengths)


def test_multi_interval_three_segments(monkeypatch):
    # Support with three disjoint segments
    # Linear segments: [0,2], [2,4], [4,7]
    support = np.array([[0.0, 2.0], [5.0, 7.0], [10.0, 13.0]])

    # Place epoch in the second segment; duration 1.0
    # Map to linear: [2.5, 3.5]. Shift +1.0 => [3.5, 4.5]
    # This crosses from second segment into third, so it should split
    monkeypatch.setattr(np.random, "uniform", _mock_uniform_factory(1.0))

    epoch = nel.EpochArray([(5.5, 6.5)])
    out = randomize_epochs(epoch, randomize_each=False, start_stop=support)

    expected = np.array([[6.5, 7.0], [10.0, 10.5]])
    assert np.allclose(out.data, expected)
    assert np.allclose(out.lengths.sum(), epoch.lengths.sum())
    # Ensure all output intervals lie within the support
    for s, e in out.data:
        assert (
            (0.0 <= s <= 2.0 and 0.0 <= e <= 2.0)
            or (5.0 <= s <= 7.0 and 5.0 <= e <= 7.0)
            or (10.0 <= s <= 13.0 and 10.0 <= e <= 13.0)
        )


def test_uses_epoch_domain_when_no_start_stop(monkeypatch):
    # Domain with two segments; verify we use epoch.domain when start_stop is None
    domain_support = np.array([[0.0, 5.0], [10.0, 12.0]])
    domain = nel.EpochArray(domain_support)

    # Epoch intervals within the domain
    epoch = nel.EpochArray([(1.0, 2.0), (10.5, 11.0)], domain=domain)

    # Deterministic shift +1.0 across the linearized domain [0,7)
    monkeypatch.setattr(np.random, "uniform", _mock_uniform_factory(1.0))

    out = randomize_epochs(epoch, randomize_each=False)

    expected = np.array([[2.0, 3.0], [11.5, 12.0]])
    assert np.allclose(out.data, expected)
    # durations preserved per-interval
    assert np.allclose(out.lengths, epoch.lengths)
    # domain should equal the domain support used
    assert np.allclose(out.domain.data, domain_support)


def test_avoid_split_per_interval(monkeypatch):
    # Two segments; choose per-interval shifts that keep each interval within its segment
    support = np.array([[0.0, 5.0], [10.0, 15.0]])
    epoch = nel.EpochArray([(1.0, 2.0), (12.0, 13.0)])

    # For avoid_split=True with randomize_each=True, we sample per-interval within valid ranges.
    # Mock uniform to return 1.0 for both intervals (valid for each segment)
    monkeypatch.setattr(np.random, "uniform", _mock_uniform_factory([1.0, 1.0]))

    out = randomize_epochs(
        epoch, randomize_each=True, start_stop=support, avoid_split=True
    )

    # First interval shift +1 => [2,3] in segment 1; second => [13,14] in segment 2
    expected = np.array([[2.0, 3.0], [13.0, 14.0]])
    assert np.allclose(out.data, expected)
    # Interval count preserved
    assert out.n_intervals == epoch.n_intervals
    # All within their respective segments
    for s, e in out.data:
        assert (0.0 <= s <= 5.0 and 0.0 <= e <= 5.0) or (
            10.0 <= s <= 15.0 and 10.0 <= e <= 15.0
        )


def test_avoid_split_common_shift_feasible(monkeypatch):
    # Three segments; choose a common shift that keeps all intervals within their segments
    support = np.array([[0.0, 2.0], [5.0, 7.0], [10.0, 13.0]])
    epoch = nel.EpochArray([(0.5, 1.0), (5.5, 6.0), (11.0, 11.5)])

    # Analyze valid ranges roughly: for each small interval, a shift of +0.2 works for all.
    monkeypatch.setattr(np.random, "uniform", _mock_uniform_factory(0.2))

    out = randomize_epochs(
        epoch, randomize_each=False, start_stop=support, avoid_split=True
    )

    expected = np.array([[0.7, 1.2], [5.7, 6.2], [11.2, 11.7]])
    assert np.allclose(out.data, expected)
    assert out.n_intervals == epoch.n_intervals


def test_avoid_split_common_shift_infeasible(monkeypatch):
    # Construct intervals whose permissible shift ranges do not intersect
    support = np.array([[0.0, 2.0], [5.0, 7.0]])
    # Craft intervals so permissible ranges don't intersect:
    # First segment interval near the end -> allowed shift range ends at 0.0
    # Second segment interval long enough from left -> allowed shift range starts > 0.0
    # Ranges: first high = 0.0, second low = 0.2 => empty intersection
    epoch = nel.EpochArray([(1.95, 2.0), (5.0, 6.8)])

    # Any common shift likely fails; expect ValueError
    monkeypatch.setattr(np.random, "uniform", _mock_uniform_factory(0.0))

    with pytest.raises(ValueError):
        _ = randomize_epochs(
            epoch, randomize_each=False, start_stop=support, avoid_split=True
        )
