import glob
import os
import pickle
import tempfile

import pandas as pd

from neuro_py.process import batch_analysis


def test_batchanalysis():
    def test_analysis(basepath, a="Real", b="Python", c="Is", d="Great", e="!"):
        results = {
            "basepath": os.path.exists(basepath),
            "exists?": basepath,
            "a": a,
            "b": b,
            "c": c,
            "d": d,
            "e": e,
        }
        return results

    df = pd.DataFrame()
    basepaths = [
        os.sep + os.path.join("test_data", f"test_data_{i}") for i in range(1, 4)
    ]
    df["basepath"] = basepaths

    # test serial
    with tempfile.TemporaryDirectory() as save_path:
        batch_analysis.run(
            df,
            save_path,
            test_analysis,
            parallel=False,
            overwrite=True,
            a="fake",
            b="julksdjflm",
        )

        sessions = glob.glob(save_path + os.sep + "*.pkl")
        assert len(sessions) == 3
        for session in sessions:
            with open(session, "rb") as f:
                results = pickle.load(f)

            assert results["a"] == "fake"
            assert results["b"] == "julksdjflm"
            assert results["c"] == "Is"
            assert results["d"] == "Great"
            assert results["e"] == "!"

    # test parallel
    with tempfile.TemporaryDirectory() as save_path:

        batch_analysis.run(
            df,
            save_path,
            test_analysis,
            parallel=True,
            overwrite=True,
            a="fake",
            b="julksdjflm",
        )

        sessions = glob.glob(save_path + os.sep + "*.pkl")
        assert len(sessions) == 3
        for session in sessions:
            with open(session, "rb") as f:
                results = pickle.load(f)

            assert results["a"] == "fake"
            assert results["b"] == "julksdjflm"
            assert results["c"] == "Is"
            assert results["d"] == "Great"
            assert results["e"] == "!"

    # test load_results
    def test_analysis(basepath):
        results = pd.DataFrame()
        results["basepath"] = [basepath]
        return results

    with tempfile.TemporaryDirectory() as save_path:
        batch_analysis.run(
            df,
            save_path,
            test_analysis,
            parallel=False,
            overwrite=True,
        )

        df = batch_analysis.load_results(save_path)
        assert df.shape[0] == 3
        for basepath in basepaths:
            assert basepath in df["basepath"].values

    # test file encode/decode
    with tempfile.TemporaryDirectory() as save_path:
        file = os.path.join("C:", os.sep, "test_data", "test_data_1")
        file = os.path.normpath(file)
        encoded_file = batch_analysis.encode_file_path(file, save_path)
        decoded_file = batch_analysis.decode_file_path(encoded_file)
        assert decoded_file == file
        assert encoded_file == os.path.normpath(
            os.path.join(save_path, "C---___test_data___test_data_1.pkl")
        )
