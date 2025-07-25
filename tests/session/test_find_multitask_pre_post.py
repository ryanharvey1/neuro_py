import numpy as np
import pandas as pd

from neuro_py.session import locate_epochs


def test_find_multitask_pre_post():
    # Example usage
    df1 = pd.DataFrame(
        {
            "name": [
                "pre_task_sleep",
                "open_field",
                "injection",
                "post_task_sleep",
                "test",
                "post_task_sleep",
            ],
            "environment": [
                "sleep",
                "open_field",
                "home_cage",
                "sleep",
                "open_field",
                "sleep",
            ],
        }
    )

    epoch_list = locate_epochs.find_multitask_pre_post(
        df1["environment"], task_tag="open_field"
    )
    assert np.allclose(epoch_list, [[0, 1, 3], [3, 4, 5]])

    # Example usage
    df2 = pd.DataFrame(
        {
            "name": ["box", "box", "box", "sleep", "box", "sleep", "tmaze", "sleep"],
            "environment": [
                "box",
                "box",
                "box",
                "sleep",
                "box",
                "sleep",
                "tmaze",
                "sleep",
            ],
        }
    )

    epoch_list = locate_epochs.find_multitask_pre_post(
        df2["environment"], task_tag="box|tmaze"
    )
    assert np.allclose(epoch_list, [[3, 4, 5], [5, 6, 7]])
