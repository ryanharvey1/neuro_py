import glob
import os
import pickle
import pandas as pd
from neuro_py.process import batch_analysis
import tempfile


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
    df["basepath"] = [
        r"\test_data\test_data_1",
        r"\test_data\test_data_2",
        r"\test_data\test_data_3",
    ]

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
        assert r"\test_data\test_data_1" in df["basepath"].values
        assert r"\test_data\test_data_2" in df["basepath"].values
        assert r"\test_data\test_data_3" in df["basepath"].values

    # test file encode/decode
    with tempfile.TemporaryDirectory() as save_path:
        file = r"C:\test_data\test_data_1"
        encoded_file = batch_analysis.encode_file_path(file, save_path)
        decoded_file = batch_analysis.decode_file_path(encoded_file)
        assert decoded_file == file
        assert encoded_file == save_path + os.sep + "C---___test_data___test_data_1.pkl"
