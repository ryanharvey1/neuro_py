"""Small CellExplorer files that exercise the real neuro_py loaders."""

import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat


@pytest.fixture
def session(tmp_path):
    path = tmp_path / "rat01" / "day01"
    path.mkdir(parents=True)

    def save(suffix, data):
        savemat(path / (path.name + suffix), data, long_field_names=True)

    save(
        ".session.mat",
        {
            "session": {
                "animal": {"name": "rat01"},
                "brainRegions": {"CA1": {"channels": [1, 2]}, "PFC": {"channels": [3]}},
                "extracellular": {"sr": 30000.0, "srLfp": 1250.0, "nChannels": 3},
                "epochs": np.array(
                    [
                        {
                            "name": "pre",
                            "startTime": 0.0,
                            "stopTime": 1.0,
                            "environment": "sleep",
                        },
                        {
                            "name": "maze",
                            "startTime": 1.0,
                            "stopTime": 3.0,
                            "environment": "linear",
                        },
                        {
                            "name": "post",
                            "startTime": 3.0,
                            "stopTime": 4.0,
                            "environment": "sleep",
                        },
                    ],
                    dtype=object,
                ),
            }
        },
    )
    save(
        ".cell_metrics.cellinfo.mat",
        {
            "cell_metrics": {
                "UID": [7, 42, 99],
                "brainRegion": np.array(["CA1sp", "CA1so", "PFC"], dtype=object),
                "putativeCellType": np.array(["Pyr", "Pyr", "Int"], dtype=object),
                "firingRate": [1.0, 2.0, np.nan],
                "tags": {"Bad": [42]},
                "general": {"basename": "day01", "animal": {"species": "rat"}},
            }
        },
    )
    save(
        ".animal.behavior.mat",
        {
            "behavior": {
                "trials": {
                    "starts": [1.1, 2.1],
                    "stops": [1.9, 2.9],
                    "stateName": np.array(["out", "in"], dtype=object),
                },
            }
        },
    )
    save(
        ".ripples.events.mat",
        {
            "ripples": {
                "timestamps": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
                "peaks": [0.15, 0.35, 0.55],
                "flagged": [2],
                "added": [0.8],
                "detectorinfo": {"detectorname": "synthetic"},
            }
        },
    )
    save(
        ".SleepState.states.mat",
        {
            "SleepState": {
                "idx": {
                    "statenames": np.array(["WAKE", "NREM", "REM"], dtype=object),
                    "states": [1, 2, 3],
                    "timestamps": [0.0, 1.0, 2.0],
                },
                "ints": {
                    "NREMstate": [[0.0, 1.0], [3.0, 4.0]],
                    "WAKEstate": [[1.0, 3.0]],
                    "REMstate": np.empty((0, 2)),
                },
            }
        },
    )
    return path


@pytest.fixture
def catalog(session):
    filename = session.parent.parent / "sessions.csv"
    pd.DataFrame(
        [
            {"basepath": "rat01/day01", "animal": "rat01", "env": "linear"},
            {"basepath": "rat01/day01", "animal": "rat01", "env": "sleep"},
        ]
    ).to_csv(filename, index=False)
    return filename
