import numpy as np
import pytest
from nelpy import EpochArray, SpikeTrainArray

from neuro_py.process.peri_event import get_rank_order


# Mock spike train data
def generate_mock_spike_train(n_cells, n_spikes, duration):
    """
    Generate mock spike train data for testing.
    """
    spike_times = [
        np.sort(np.random.uniform(0, duration, size=n_spikes)) for _ in range(n_cells)
    ]
    return SpikeTrainArray(spike_times, fs=20_000)


# Mock epoch array data
def generate_mock_epochs(n_epochs, duration, epoch_length):
    """
    Generate mock epochs for testing.
    """
    starts = np.arange(0, n_epochs * duration, duration)
    stops = starts + epoch_length
    return EpochArray(np.vstack((starts, stops)).T)


def test_get_rank_order_first_spike_cells():
    st = generate_mock_spike_train(n_cells=10, n_spikes=20, duration=10)
    epochs = generate_mock_epochs(n_epochs=5, duration=2, epoch_length=1)

    median_rank, rank_order = get_rank_order(
        st, epochs, method="first_spike", ref="cells", padding=0.1
    )

    assert median_rank.shape == (10,)
    assert rank_order.shape == (10, 5)
    # Replace NaNs with 0 for easier comparison
    rank_order[np.isnan(rank_order)] = 0
    assert np.all((0 <= rank_order) & (rank_order <= 1))  # Normalized rank orders


def test_get_rank_order_first_spike_epoch():
    st = generate_mock_spike_train(n_cells=10, n_spikes=20, duration=10)
    epochs = generate_mock_epochs(n_epochs=5, duration=2, epoch_length=1)

    median_rank, rank_order = get_rank_order(
        st, epochs, method="first_spike", ref="epoch", padding=0.1
    )

    assert median_rank.shape == (10,)
    assert rank_order.shape == (10, 5)
    # Replace NaNs with 0 for easier comparison
    rank_order[np.isnan(rank_order)] = 0
    assert np.all((0 <= rank_order) & (rank_order <= 1))  # Normalized rank orders


def test_get_rank_order_peak_fr_cells():
    st = generate_mock_spike_train(n_cells=10, n_spikes=50, duration=10)
    epochs = generate_mock_epochs(n_epochs=5, duration=2, epoch_length=1)

    median_rank, rank_order = get_rank_order(
        st, epochs, method="peak_fr", ref="cells", dt=0.001, sigma=0.01, padding=0.1
    )

    assert median_rank.shape == (10,)
    assert rank_order.shape == (10, 5)
    # Replace NaNs with 0 for easier comparison
    rank_order[np.isnan(rank_order)] = 0
    assert np.all((0 <= rank_order) & (rank_order <= 1))  # Normalized rank orders


def test_get_rank_order_peak_fr_epoch():
    st = generate_mock_spike_train(n_cells=10, n_spikes=50, duration=10)
    epochs = generate_mock_epochs(n_epochs=5, duration=2, epoch_length=1)

    median_rank, rank_order = get_rank_order(
        st, epochs, method="peak_fr", ref="epoch", dt=0.001, sigma=0.01, padding=0.1
    )

    assert median_rank.shape == (10,)
    assert rank_order.shape == (10, 5)
    # Replace NaNs with 0 for easier comparison
    rank_order[np.isnan(rank_order)] = 0
    assert np.all((0 <= rank_order) & (rank_order <= 1))  # Normalized rank orders


def test_get_rank_order_empty_spike_train():
    st = SpikeTrainArray([[] for _ in range(10)])  # 10 cells, no spikes
    epochs = generate_mock_epochs(n_epochs=5, duration=2, epoch_length=1)

    median_rank, rank_order = get_rank_order(
        st, epochs, method="first_spike", ref="cells"
    )

    assert np.all(np.isnan(median_rank))
    assert np.all(np.isnan(rank_order))


def test_get_rank_order_no_spikes_in_epochs():
    st = generate_mock_spike_train(n_cells=10, n_spikes=20, duration=10)
    epochs = EpochArray([[20, 22]])  # No spikes fall within this range

    median_rank, rank_order = get_rank_order(st, epochs, method="peak_fr", ref="epoch")

    assert np.all(np.isnan(median_rank))
    assert np.all(np.isnan(rank_order))


def test_get_rank_order_invalid_method():
    st = generate_mock_spike_train(n_cells=10, n_spikes=20, duration=10)
    epochs = generate_mock_epochs(n_epochs=5, duration=2, epoch_length=1)

    with pytest.raises(Exception, match="method random_method not implemented"):
        get_rank_order(st, epochs, method="random_method", ref="cells")


def test_get_rank_order_invalid_reference():
    st = generate_mock_spike_train(n_cells=10, n_spikes=20, duration=10)
    epochs = generate_mock_epochs(n_epochs=5, duration=2, epoch_length=1)

    with pytest.raises(Exception, match="ref random_ref not implemented"):
        get_rank_order(st, epochs, method="first_spike", ref="random_ref")


def test_get_rank_order_explicit_spike_times():
    # Explicit spike times for predictable rank order
    # Cell 0 spikes first, Cell 1 second, etc.
    spike_times = [
        np.array([0.1]),  # Cell 0 spikes first
        np.array([0.2]),  # Cell 1 spikes second
        np.array([0.3]),  # Cell 2 spikes third
        np.array([0.4]),  # Cell 3 spikes fourth
        np.array([0.5]),  # Cell 4 spikes fifth
    ]
    st = SpikeTrainArray(spike_times)

    # One epoch covering all spike times
    epochs = EpochArray(np.array([[0, 1]]))  # From 0 to 1 second

    # Expected rank order: cells are ranked based on their spike times
    expected_rank_order_cells = np.array(
        [
            [0.0],  # Cell 0 has rank 0 (earliest spike)
            [0.2],  # Cell 1
            [0.4],  # Cell 2
            [0.6],  # Cell 3
            [0.8],  # Cell 4 has rank 1 (latest spike)
        ]
    )

    # Test the 'first_spike' method with 'cells' reference
    median_rank, rank_order = get_rank_order(
        st, epochs, method="first_spike", ref="cells", padding=0, dt=0.001
    )

    # Check shapes
    assert median_rank.shape == (5,)
    assert rank_order.shape == (5, 1)

    # Check rank order values
    np.testing.assert_almost_equal(rank_order, expected_rank_order_cells, decimal=3)
    np.testing.assert_almost_equal(
        median_rank, np.nanmedian(expected_rank_order_cells, axis=1), decimal=3
    )

    # Test the 'first_spike' method with 'epoch' reference
    # For 'epoch' reference, rank should be normalized to the epoch (0-1)
    expected_rank_order_epoch = np.array(
        [
            [0.1],  # Cell 0 spikes at 0.1 in a 0-1 interval
            [0.2],  # Cell 1
            [0.3],  # Cell 2
            [0.4],  # Cell 3
            [0.5],  # Cell 4
        ]
    )

    median_rank_epoch, rank_order_epoch = get_rank_order(
        st, epochs, method="first_spike", ref="epoch", padding=0, dt=0.001
    )

    # Check shapes
    assert median_rank_epoch.shape == (5,)
    assert rank_order_epoch.shape == (5, 1)

    # Check rank order values
    np.testing.assert_almost_equal(
        rank_order_epoch, expected_rank_order_epoch, decimal=3
    )
    np.testing.assert_almost_equal(
        median_rank_epoch, np.nanmedian(expected_rank_order_epoch, axis=1), decimal=3
    )
