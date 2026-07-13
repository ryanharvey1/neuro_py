import numpy as np
import pandas as pd

from neuro_py.ensemble.decoding.preprocess import partition_sets


def test_partition_sets_preserves_dataframe_rows() -> None:
    indices = [(np.array([0]), np.array([1]), np.array([2]))]
    neural = pd.DataFrame({"neural": [1.0, 2.0, 3.0]})
    behavior = pd.DataFrame({"behavior": [4.0, 5.0, 6.0]})

    train, train_behavior, validation, validation_behavior, test, test_behavior = (
        partition_sets(indices, neural, behavior)[0]
    )

    assert train["neural"].tolist() == [1.0]
    assert train_behavior["behavior"].tolist() == [4.0]
    assert validation["neural"].tolist() == [2.0]
    assert validation_behavior["behavior"].tolist() == [5.0]
    assert test["neural"].tolist() == [3.0]
    assert test_behavior["behavior"].tolist() == [6.0]
