import numpy as np
from neuro_py.process.peri_event import event_triggered_average_fast


def test_event_triggered_average_fast():
    # Test 1: events are at the right time

    # make 5 random signals
    x = np.random.rand(5, 100_000)
    # create 100_000 timestamps at 1000 Hz
    ts = np.linspace(0, 100, 100_000)
    # 5 random timestamps from 0 to 10
    ts_rand = np.random.choice(ts, 5)
    # set the 5 signals to 5 at the random timestamps
    x[:, np.isin(ts, ts_rand)] = 5
    # make psth of the 5 signals at the random timestamps
    psth = event_triggered_average_fast(
        x,
        ts_rand,
        sampling_rate=1000,
        window=[-0.5, 0.5],
        return_average=True,
        return_pandas=True,
    )
    # assert that the psth peak is at the middle of the window
    assert np.allclose(psth.index[np.argmax(psth.values, axis=0)], 0, atol=1e-3)
