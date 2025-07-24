from neuro_py.process import peri_event
import numpy as np

def test_count_events():
    events = np.array([1, 2, 3, 4, 5])
    time_ref = np.array([0, 1, 2, 3])
    time_range = (1, 3)
    expected = np.array([1, 1, 1, 1])
    assert np.allclose(peri_event.count_events(events, time_ref, time_range), expected)

    events = np.array([1, 6, 7, 10])
    time_ref = np.array([0, 2, 4, 6])
    time_range = (1, 3)
    expected = np.array([0, 0, 1, 0])
    assert np.allclose(peri_event.count_events(events, time_ref, time_range), expected)

    events = np.array([1, 2, 3, 6])
    time_ref = np.array([0, 1, 2, 3])
    time_range = (0, 1)
    expected = np.array([0, 0, 0, 0])
    assert np.allclose(peri_event.count_events(events, time_ref, time_range), expected)

    events = np.array([1, 2, 3, 6])
    time_ref = np.array([0, 1, 2, 3])
    time_range = (-2, 2)
    expected = np.array([1, 2, 3, 2])
    assert np.allclose(peri_event.count_events(events, time_ref, time_range), expected)
