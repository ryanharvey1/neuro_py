import nelpy as nel
import numpy as np
import pytest

from neuro_py.process.intervals import (
    find_intersecting_intervals,
    find_intersection_intervals_strict,
    get_overlapping_intervals,
    in_intervals_interval,
    overlap_intersect,
    randomize_epochs,
    split_epoch_by_width,
    split_epoch_equal_parts,
)


class TestFindIntersectingIntervals:
    """Tests for find_intersecting_intervals function."""

    def test_basic_intersection(self):
        """Test basic intersection detection."""
        set1 = nel.EpochArray([(1, 3), (5, 7), (9, 10)])
        set2 = nel.EpochArray([(2, 4), (6, 8)])
        
        # Test return_indices=True (boolean array)
        result = find_intersecting_intervals(set1, set2, return_indices=True)
        expected = np.array([True, True, False])
        assert np.array_equal(result, expected)
        
        # Test return_indices=False (intersection amounts)
        result = find_intersecting_intervals(set1, set2, return_indices=False)
        expected = np.array([1.0, 1.0, 0.0])
        assert np.allclose(result, expected)

    def test_complete_overlap(self):
        """Test when intervals completely overlap."""
        set1 = nel.EpochArray([(2, 4), (6, 8)])
        set2 = nel.EpochArray([(1, 5), (5, 9)])
        
        result = find_intersecting_intervals(set1, set2, return_indices=True)
        expected = np.array([True, True])
        assert np.array_equal(result, expected)
        
        result = find_intersecting_intervals(set1, set2, return_indices=False)
        expected = np.array([2.0, 2.0])
        assert np.allclose(result, expected)

    def test_no_intersection(self):
        """Test when there is no intersection."""
        set1 = nel.EpochArray([(1, 2), (3, 4)])
        set2 = nel.EpochArray([(5, 6), (7, 8)])
        
        result = find_intersecting_intervals(set1, set2, return_indices=True)
        expected = np.array([False, False])
        assert np.array_equal(result, expected)
        
        result = find_intersecting_intervals(set1, set2, return_indices=False)
        expected = np.array([0.0, 0.0])
        assert np.allclose(result, expected)

    def test_partial_overlap(self):
        """Test partial overlap cases."""
        set1 = nel.EpochArray([(1, 5), (6, 10)])
        set2 = nel.EpochArray([(3, 7)])
        
        result = find_intersecting_intervals(set1, set2, return_indices=False)
        expected = np.array([2.0, 1.0])
        assert np.allclose(result, expected)


class TestFindIntersectionIntervalsStrict:
    """Tests for find_intersection_intervals_strict function."""

    def test_completely_contained(self):
        """Test intervals completely contained within another set."""
        set1 = nel.EpochArray([(1, 3), (5, 7), (9, 10)])
        set2 = nel.EpochArray([(0, 4), (6, 8)])
        
        result = find_intersection_intervals_strict(set1, set2)
        
        # Only (1, 3) is completely within (0, 4)
        assert result.n_intervals == 1
        assert np.allclose(result.data[0], [1, 3])

    def test_partial_overlap_excluded(self):
        """Test that partially overlapping intervals are excluded."""
        set1 = nel.EpochArray([(1, 5), (6, 8)])
        set2 = nel.EpochArray([(2, 4), (7, 10)])
        
        result = find_intersection_intervals_strict(set1, set2)
        
        # Neither interval is completely contained
        assert result.n_intervals == 0

    def test_all_contained(self):
        """Test when all intervals are completely contained."""
        set1 = nel.EpochArray([(2, 3), (5, 6), (8, 9)])
        set2 = nel.EpochArray([(0, 10)])
        
        result = find_intersection_intervals_strict(set1, set2)
        
        assert result.n_intervals == 3
        assert np.allclose(result.data, set1.data)

    def test_domain_inheritance(self):
        """Test that domain is inherited from set1."""
        set1 = nel.EpochArray([(2, 3), (5, 6)], domain=nel.EpochArray([0, 10]))
        set2 = nel.EpochArray([(0, 10)])
        
        result = find_intersection_intervals_strict(set1, set2)
        
        assert result.domain.start == set1.domain.start
        assert result.domain.stop == set1.domain.stop

    def test_empty_result(self):
        """Test when no intervals are completely contained."""
        set1 = nel.EpochArray([(1, 5), (7, 12)])
        set2 = nel.EpochArray([(3, 8)])
        
        result = find_intersection_intervals_strict(set1, set2)
        
        assert result.n_intervals == 0

    def test_theta_cycles_in_running(self):
        """Test realistic use case: theta cycles completely within running periods."""
        # Simulate theta cycles
        theta_cycles = nel.EpochArray([
            (10.0, 10.125),  # Completely in run
            (10.5, 10.625),  # Completely in run
            (14.85, 14.95),  # Completely in run (within bounds)
            (14.95, 15.05),  # Partially overlaps run end
            (15.2, 15.35),   # Outside run
        ])
        
        # Simulate running periods
        running = nel.EpochArray([
            (10.0, 15.0),
            (20.0, 25.0)
        ])
        
        result = find_intersection_intervals_strict(theta_cycles, running)
        
        # First three cycles are completely within running
        assert result.n_intervals == 3
        assert np.allclose(result.data[0], [10.0, 10.125])
        assert np.allclose(result.data[1], [10.5, 10.625])
        assert np.allclose(result.data[2], [14.85, 14.95])


class TestOverlapIntersect:
    """Tests for overlap_intersect function."""

    def test_basic_overlap(self):
        """Test basic overlap detection."""
        epoch = nel.EpochArray([(1, 3), (5, 7), (9, 11)])
        interval = nel.EpochArray([(2, 6), (10, 12)])
        
        result, indices = overlap_intersect(epoch, interval, return_indices=True)
        
        # All three epochs have overlap with at least one interval
        assert result.n_intervals == 3
        assert len(indices) == 3
        assert indices[0] == 0  # (1,3) overlaps with (2,6)
        assert indices[1] == 0  # (5,7) overlaps with (2,6)
        assert indices[2] == 1  # (9,11) overlaps with (10,12)

    def test_no_overlap(self):
        """Test when there is no overlap."""
        epoch = nel.EpochArray([(1, 2), (3, 4)])
        interval = nel.EpochArray([(5, 6)])
        
        result, indices = overlap_intersect(epoch, interval, return_indices=True)
        
        assert result.n_intervals == 0
        assert indices == []

    def test_without_indices(self):
        """Test without returning indices."""
        epoch = nel.EpochArray([(1, 3), (5, 7)])
        interval = nel.EpochArray([(2, 6)])
        
        result = overlap_intersect(epoch, interval, return_indices=False)
        
        assert result.n_intervals == 2

    def test_domain_preservation(self):
        """Test that domain is preserved."""
        epoch = nel.EpochArray([(1, 3), (5, 7)], domain=nel.EpochArray([0, 10]))
        interval = nel.EpochArray([(2, 6)])
        
        result = overlap_intersect(epoch, interval, return_indices=False)
        
        assert result.domain.start == epoch.domain.start
        assert result.domain.stop == epoch.domain.stop


class TestRandomizeEpochs:
    """Tests for randomize_epochs function."""

    def test_randomize_each(self):
        """Test randomizing each epoch independently."""
        epoch = nel.EpochArray([(1, 2), (3, 4), (5, 6)])
        
        result = randomize_epochs(epoch, randomize_each=True)
        
        # Should have same number of intervals
        assert result.n_intervals == epoch.n_intervals
        
        # Should preserve interval lengths
        assert np.allclose(result.lengths, epoch.lengths)
        
        # Should be within domain bounds
        assert result.start >= epoch.start
        assert result.stop <= epoch.stop

    def test_randomize_all_same(self):
        """Test randomizing all epochs by the same amount."""
        epoch = nel.EpochArray([(10, 11), (13, 14), (16, 17)])
        
        result = randomize_epochs(epoch, randomize_each=False)
        
        # Should have same number of intervals
        assert result.n_intervals == epoch.n_intervals
        
        # Should preserve lengths
        assert np.allclose(result.lengths, epoch.lengths)
        
        # Check relative ordering is maintained (or equal after wrapping)
        assert result.data[1, 0] >= result.data[0, 1]  # Gap or adjacent between 1st and 2nd
        assert result.data[2, 0] >= result.data[1, 1]  # Gap or adjacent between 2nd and 3rd
        
        # Intervals should not overlap
        assert np.all(result.data[:, 1] >= result.data[:, 0])

    def test_custom_start_stop(self):
        """Test with custom start/stop boundaries."""
        epoch = nel.EpochArray([(1, 2), (3, 4)])
        start_stop = np.array([0, 10])
        
        result = randomize_epochs(epoch, randomize_each=True, start_stop=start_stop)
        
        assert result.start >= start_stop[0]
        assert result.stop <= start_stop[1]

    def test_wrapping(self):
        """Test that intervals wrap around correctly."""
        epoch = nel.EpochArray([(1, 2)])
        
        # Run multiple times to check wrapping behavior
        for _ in range(10):
            result = randomize_epochs(epoch, randomize_each=True)
            assert np.all(result.data[:, 1] > result.data[:, 0])


class TestSplitEpochByWidth:
    """Tests for split_epoch_by_width function."""

    def test_single_interval(self):
        """Test splitting a single interval."""
        intervals = [(0, 1)]
        bin_width = 0.1
        
        result = split_epoch_by_width(intervals, bin_width)
        
        assert result.shape[0] == 10  # 1 second / 0.1 = 10 bins
        assert np.allclose(result[0], [0, 0.1])
        assert np.allclose(result[-1], [0.9, 1.0])

    def test_multiple_intervals(self):
        """Test splitting multiple intervals."""
        intervals = [(0, 1), (2, 3)]
        bin_width = 0.5
        
        result = split_epoch_by_width(intervals, bin_width)
        
        assert result.shape[0] == 4  # 2 bins per interval
        assert np.allclose(result[0], [0, 0.5])
        assert np.allclose(result[2], [2, 2.5])

    def test_default_bin_width(self):
        """Test with default bin width."""
        intervals = [(0, 0.01)]
        
        result = split_epoch_by_width(intervals)
        
        assert result.shape[0] == 10  # 0.01 / 0.001 = 10


class TestSplitEpochEqualParts:
    """Tests for split_epoch_equal_parts function."""

    def test_single_interval(self):
        """Test splitting a single interval into equal parts."""
        intervals = np.array([[0, 10]])
        n_parts = 5
        
        result = split_epoch_equal_parts(intervals, n_parts, return_epoch_array=False)
        
        assert result.shape == (5, 2)
        assert np.allclose(result[0], [0, 2])
        assert np.allclose(result[-1], [8, 10])

    def test_multiple_intervals(self):
        """Test splitting multiple intervals."""
        intervals = np.array([[0, 10], [20, 30]])
        n_parts = 2
        
        result = split_epoch_equal_parts(intervals, n_parts, return_epoch_array=False)
        
        assert result.shape == (4, 2)
        assert np.allclose(result[0], [0, 5])
        assert np.allclose(result[1], [5, 10])
        assert np.allclose(result[2], [20, 25])
        assert np.allclose(result[3], [25, 30])

    def test_return_epoch_array(self):
        """Test returning an EpochArray."""
        intervals = np.array([[0, 10]])
        n_parts = 3
        
        result = split_epoch_equal_parts(intervals, n_parts, return_epoch_array=True)
        
        assert isinstance(result, nel.EpochArray)
        assert result.n_intervals == 3

    def test_unequal_interval_lengths(self):
        """Test with intervals of different lengths."""
        intervals = np.array([[0, 10], [20, 25]])
        n_parts = 2
        
        result = split_epoch_equal_parts(intervals, n_parts, return_epoch_array=False)
        
        # First interval: [0, 5], [5, 10]
        assert np.allclose(result[0], [0, 5])
        assert np.allclose(result[1], [5, 10])
        
        # Second interval: [20, 22.5], [22.5, 25]
        assert np.allclose(result[2], [20, 22.5])
        assert np.allclose(result[3], [22.5, 25])


class TestInIntervalsInterval:
    """Tests for in_intervals_interval function."""

    def test_basic_assignment(self):
        """Test basic interval assignment."""
        timestamps = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        intervals = np.array([[2, 4], [5, 7]])
        
        result = in_intervals_interval(timestamps, intervals)
        
        expected = np.array([np.nan, 0, 0, 0, 1, 1, 1, np.nan])
        assert np.allclose(result, expected, equal_nan=True)

    def test_all_outside(self):
        """Test when all timestamps are outside intervals."""
        timestamps = np.array([1, 2, 3])
        intervals = np.array([[5, 10]])
        
        result = in_intervals_interval(timestamps, intervals)
        
        assert np.all(np.isnan(result))

    def test_all_inside(self):
        """Test when all timestamps are inside one interval."""
        timestamps = np.array([5, 6, 7, 8])
        intervals = np.array([[4, 10]])
        
        result = in_intervals_interval(timestamps, intervals)
        
        expected = np.array([0, 0, 0, 0])
        assert np.array_equal(result, expected)

    def test_boundary_conditions(self):
        """Test timestamps at interval boundaries."""
        timestamps = np.array([2, 4, 5, 7])
        intervals = np.array([[2, 4], [5, 7]])
        
        result = in_intervals_interval(timestamps, intervals)
        
        expected = np.array([0, 0, 1, 1])
        assert np.array_equal(result, expected)


class TestGetOverlappingIntervals:
    """Tests for get_overlapping_intervals function."""

    def test_basic_overlapping(self):
        """Test basic overlapping intervals generation."""
        start, stop = 0, 10
        interval_width = 2
        slideby = 1
        
        result = get_overlapping_intervals(start, stop, interval_width, slideby)
        
        expected = np.array([
            [0, 2], [1, 3], [2, 4], [3, 5], [4, 6],
            [5, 7], [6, 8], [7, 9]
        ])
        assert np.array_equal(result, expected)

    def test_no_overlap(self):
        """Test when slideby equals interval_width (no overlap)."""
        start, stop = 0, 10
        interval_width = 2
        slideby = 2
        
        result = get_overlapping_intervals(start, stop, interval_width, slideby)
        
        expected = np.array([[0, 2], [2, 4], [4, 6], [6, 8]])
        assert np.array_equal(result, expected)

    def test_high_overlap(self):
        """Test with high overlap (small slideby)."""
        start, stop = 0, 5
        interval_width = 2
        slideby = 0.5
        
        result = get_overlapping_intervals(start, stop, interval_width, slideby)
        
        assert result.shape[0] == 6  # (5 - 2) / 0.5 = 6
        assert np.allclose(result[0], [0, 2])
        assert np.allclose(result[1], [0.5, 2.5])

    def test_float_values(self):
        """Test with float start/stop values."""
        start, stop = 1.5, 5.5
        interval_width = 1.0
        slideby = 0.5
        
        result = get_overlapping_intervals(start, stop, interval_width, slideby)
        
        assert result.shape[1] == 2
        assert np.allclose(result[0], [1.5, 2.5])
        assert result[-1, 1] <= stop
