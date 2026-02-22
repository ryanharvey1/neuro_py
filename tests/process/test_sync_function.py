import numpy as np

from neuro_py.process.peri_event import sync


def test_sync_unsorted_inputs_maps_indices_to_original_order():
    samples = np.array(
        [
            [2.0, 20.0],
            [0.9, 9.0],
            [1.1, 11.0],
            [3.0, 30.0],
            [2.1, 21.0],
        ]
    )
    sync_times = np.array([2.0, 1.0])

    synchronized, Ie, Is = sync(samples, sync_times, durations=(-0.15, 0.15), fast=False)

    expected_sync = np.array(
        [
            [-0.1, 9.0],
            [0.1, 11.0],
            [0.0, 20.0],
            [0.1, 21.0],
        ]
    )
    expected_Ie = np.array([1, 1, 0, 0])
    expected_Is = np.array([1, 2, 0, 4])

    np.testing.assert_allclose(synchronized, expected_sync, atol=1e-12)
    np.testing.assert_array_equal(Ie, expected_Ie)
    np.testing.assert_array_equal(Is, expected_Is)


def test_sync_fast_mode_with_sorted_inputs():
    samples = np.array([[0.9, 9.0], [1.1, 11.0], [2.0, 20.0], [2.1, 21.0], [3.0, 30.0]])
    sync_times = np.array([1.0, 2.0])

    synchronized, Ie, Is = sync(samples, sync_times, durations=(-0.15, 0.15), fast=True)

    np.testing.assert_allclose(
        synchronized,
        np.array([[-0.1, 9.0], [0.1, 11.0], [0.0, 20.0], [0.1, 21.0]]),
        atol=1e-12,
    )
    np.testing.assert_array_equal(Ie, np.array([0, 0, 1, 1]))
    np.testing.assert_array_equal(Is, np.array([0, 1, 2, 3]))


def test_sync_returns_empty_when_no_samples_in_windows():
    samples = np.array([[10.0, 1.0], [20.0, 2.0]])
    sync_times = np.array([0.0, 1.0])

    synchronized, Ie, Is = sync(samples, sync_times, durations=(-0.1, 0.1))

    assert synchronized.shape == (0, 2)
    assert Ie.size == 0
    assert Is.size == 0
