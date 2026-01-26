import glob
import os
import pickle
import tempfile
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd
import pytest

from neuro_py.process import batch_analysis
from neuro_py.process.batch_analysis import (
    _is_homogeneous_array_compatible,
    _load_dataframe_from_hdf5,
    _load_from_hdf5,
    _load_inhomogeneous_data_hdf5,
    _save_dataframe_to_hdf5,
    _save_inhomogeneous_data_hdf5,
    _save_to_hdf5,
    decode_file_path,
    encode_file_path,
    load_results,
    load_specific_data,
    main_loop,
    run,
)


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
        # extract drive letter
        cwd = os.getcwd()
        drive = os.path.splitdrive(cwd)[0]

        if drive:
            drive_letter = drive[0]
        else:
            drive_letter = ""  # No drive letter on Linux systems

        # create a test file path
        file = os.path.join(drive, os.sep, "test_data", "test_data_1")
        file = os.path.normpath(file)
        # encode and decode the file path
        encoded_file = batch_analysis.encode_file_path(file, save_path)
        decoded_file = batch_analysis.decode_file_path(encoded_file)
        # check that the decoded file path is the same as the original file path
        assert decoded_file == file
        # check that the encoded file path is the same as the expected file path
        if drive_letter:
            expected_encoded_file = os.path.join(
                save_path, drive_letter + "---___test_data___test_data_1.pkl"
            )
        else:
            expected_encoded_file = os.path.join(
                save_path, "___test_data___test_data_1.pkl"
            )

        assert encoded_file == os.path.normpath(expected_encoded_file)


class TestFilePathEncoding:
    """Test file path encoding and decoding functions."""

    def test_encode_file_path_pickle(self):
        """Test encoding file path for pickle format."""
        with tempfile.TemporaryDirectory() as save_path:
            # extract drive letter
            cwd = os.getcwd()
            drive = os.path.splitdrive(cwd)[0]

            if drive:
                drive_letter = drive[0]
            else:
                drive_letter = ""  # No drive letter on Linux systems

            # create a test file path
            basepath = os.path.join(drive, os.sep, "Data", "Session1")
            basepath = os.path.normpath(basepath)

            # encode the file path
            result = encode_file_path(basepath, save_path, "pickle")

            # check that the encoded file path is the same as the expected file path
            if drive_letter:
                expected = os.path.join(
                    save_path, drive_letter + "---___Data___Session1.pkl"
                )
            else:
                expected = os.path.join(save_path, "___Data___Session1.pkl")

            assert result == os.path.normpath(expected)

    def test_encode_file_path_hdf5(self):
        """Test encoding file path for HDF5 format."""
        with tempfile.TemporaryDirectory() as save_path:
            # extract drive letter
            cwd = os.getcwd()
            drive = os.path.splitdrive(cwd)[0]

            if drive:
                drive_letter = drive[0]
            else:
                drive_letter = ""  # No drive letter on Linux systems

            # create a test file path
            basepath = os.path.join(drive, os.sep, "Data", "Session1")
            basepath = os.path.normpath(basepath)

            # encode the file path
            result = encode_file_path(basepath, save_path, "hdf5")

            # check that the encoded file path is the same as the expected file path
            if drive_letter:
                expected = os.path.join(
                    save_path, drive_letter + "---___Data___Session1.h5"
                )
            else:
                expected = os.path.join(save_path, "___Data___Session1.h5")

            assert result == os.path.normpath(expected)

    def test_decode_file_path_pickle(self):
        """Test decoding file path from pickle format."""
        with tempfile.TemporaryDirectory() as save_path:
            # extract drive letter
            cwd = os.getcwd()
            drive = os.path.splitdrive(cwd)[0]

            if drive:
                drive_letter = drive[0]
            else:
                drive_letter = ""  # No drive letter on Linux systems

            # create a test file path
            original_path = os.path.join(drive, os.sep, "Data", "Session1")
            original_path = os.path.normpath(original_path)

            # create the encoded file path
            if drive_letter:
                encoded_filename = drive_letter + "---___Data___Session1.pkl"
            else:
                encoded_filename = "___Data___Session1.pkl"

            save_file = os.path.join(save_path, encoded_filename)

            # decode the file path
            result = decode_file_path(save_file)

            # check that the decoded file path matches the original
            assert result == original_path

    def test_decode_file_path_hdf5(self):
        """Test decoding file path from HDF5 format."""
        with tempfile.TemporaryDirectory() as save_path:
            # extract drive letter
            cwd = os.getcwd()
            drive = os.path.splitdrive(cwd)[0]

            if drive:
                drive_letter = drive[0]
            else:
                drive_letter = ""  # No drive letter on Linux systems

            # create a test file path
            original_path = os.path.join(drive, os.sep, "Data", "Session1")
            original_path = os.path.normpath(original_path)

            # create the encoded file path
            if drive_letter:
                encoded_filename = drive_letter + "---___Data___Session1.h5"
            else:
                encoded_filename = "___Data___Session1.h5"

            save_file = os.path.join(save_path, encoded_filename)

            # decode the file path
            result = decode_file_path(save_file)

            # check that the decoded file path matches the original
            assert result == original_path

    def test_round_trip_encoding_decoding(self):
        """Test that encoding and then decoding returns the original path."""
        with tempfile.TemporaryDirectory() as save_path:
            # extract drive letter
            cwd = os.getcwd()
            drive = os.path.splitdrive(cwd)[0]

            if drive:
                drive_letter = drive[0]
            else:
                drive_letter = ""  # No drive letter on Linux systems

            # create a test file path
            original_path = os.path.join(drive, os.sep, "test_data", "test_data_1")
            original_path = os.path.normpath(original_path)

            # encode and decode the file path
            encoded_file = encode_file_path(original_path, save_path)
            decoded_file = decode_file_path(encoded_file)

            # check that the decoded file path is the same as the original file path
            assert decoded_file == original_path

            # check that the encoded file path is the same as the expected file path
            if drive_letter:
                expected_encoded_file = os.path.join(
                    save_path, drive_letter + "---___test_data___test_data_1.pkl"
                )
            else:
                expected_encoded_file = os.path.join(
                    save_path, "___test_data___test_data_1.pkl"
                )

            assert encoded_file == os.path.normpath(expected_encoded_file)


class TestHDF5DataFrameOperations:
    """Test HDF5 DataFrame save/load operations."""

    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame for testing."""
        return pd.DataFrame(
            {
                "int_col": [1, 2, 3, 4, 5],
                "float_col": [1.1, 2.2, 3.3, 4.4, 5.5],
                "string_col": ["a", "b", "c", "d", "e"],
                "bool_col": [True, False, True, False, True],
            }
        )

    def test_save_load_dataframe_hdf5(self, sample_dataframe, tmp_path):
        """Test saving and loading DataFrame to/from HDF5."""
        filepath = tmp_path / "test.h5"

        with h5py.File(filepath, "w") as f:
            _save_dataframe_to_hdf5(sample_dataframe, f, "test_df")

        with h5py.File(filepath, "r") as f:
            loaded_df = _load_dataframe_from_hdf5(f["test_df"])

        # Dtypes (including StringDtype na_value) should be preserved through HDF5 round-trip
        pd.testing.assert_frame_equal(
            sample_dataframe,
            loaded_df,
            check_dtype=True,
            check_names=True,
        )

    def test_save_load_dataframe_with_custom_index(self, tmp_path):
        """Test DataFrame with custom index."""
        df = pd.DataFrame({"values": [10, 20, 30]}, index=["a", "b", "c"])

        filepath = tmp_path / "test.h5"

        with h5py.File(filepath, "w") as f:
            _save_dataframe_to_hdf5(df, f, "test_df")

        with h5py.File(filepath, "r") as f:
            loaded_df = _load_dataframe_from_hdf5(f["test_df"])

        pd.testing.assert_frame_equal(df, loaded_df)


class TestHDF5StringDtype:
    """Lock in behavior for pandas StringDtype columns."""

    def test_save_load_stringdtype_python(self, tmp_path):
        """Round-trip pandas StringDtype (python storage) via HDF5."""
        df = pd.DataFrame({"s": pd.Series(["a", "b", None], dtype="string")})

        filepath = tmp_path / "string_python.h5"
        with h5py.File(filepath, "w") as f:
            _save_dataframe_to_hdf5(df, f, "df")

        with h5py.File(filepath, "r") as f:
            loaded_df = _load_dataframe_from_hdf5(f["df"])

        # Dtype restored as pandas StringDtype
        assert isinstance(loaded_df["s"].dtype, pd.StringDtype)
        # Missing values are serialized as empty strings in current implementation
        assert loaded_df["s"].tolist() == ["a", "b", ""]

    def test_save_load_stringdtype_pyarrow(self, tmp_path):
        """Round-trip pandas StringDtype (pyarrow storage) if available."""
        # Try to construct a pyarrow-backed StringDtype; skip if unavailable
        try:
            dtype_pa = pd.StringDtype(storage="pyarrow")
        except Exception:
            import pytest

            pytest.skip("pyarrow-backed StringDtype unavailable in this env")

        df = pd.DataFrame({"s": pd.Series(["x", "y"], dtype=dtype_pa)})

        filepath = tmp_path / "string_pyarrow.h5"
        with h5py.File(filepath, "w") as f:
            _save_dataframe_to_hdf5(df, f, "df")

        with h5py.File(filepath, "r") as f:
            loaded_df = _load_dataframe_from_hdf5(f["df"])

        # Dtype restored as pandas StringDtype (storage may default to python)
        assert isinstance(loaded_df["s"].dtype, pd.StringDtype)
        assert loaded_df["s"].tolist() == ["x", "y"]


class TestHDF5MixedDataOperations:
    """Test HDF5 operations with mixed data types."""

    @pytest.fixture
    def sample_mixed_data(self):
        """Create sample mixed data for testing."""
        return {
            "dataframe": pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]}),
            "numpy_array": np.array([[1, 2, 3], [4, 5, 6]]),
            "scalar_int": 42,
            "scalar_float": 3.14,
            "scalar_str": "hello",
            "list_data": [1, 2, 3, 4, 5],
        }

    def test_save_load_mixed_data_hdf5(self, sample_mixed_data, tmp_path):
        """Test saving and loading mixed data types."""
        filepath = tmp_path / "test.h5"

        _save_to_hdf5(sample_mixed_data, filepath)
        loaded_data = _load_from_hdf5(filepath)

        # Dtypes should be preserved exactly through HDF5 round-trip
        pd.testing.assert_frame_equal(
            sample_mixed_data["dataframe"],
            loaded_data["dataframe"],
            check_dtype=True,
        )

        # Check numpy array
        np.testing.assert_array_equal(
            sample_mixed_data["numpy_array"], loaded_data["numpy_array"]
        )

        # Check scalars
        assert loaded_data["scalar_int"] == sample_mixed_data["scalar_int"]
        assert loaded_data["scalar_float"] == sample_mixed_data["scalar_float"]
        assert loaded_data["scalar_str"] == sample_mixed_data["scalar_str"]

        # Check list (converted to numpy array)
        np.testing.assert_array_equal(
            np.array(sample_mixed_data["list_data"]), loaded_data["list_data"]
        )

    def test_save_load_single_dataframe_hdf5(self, tmp_path):
        """Test saving and loading a single DataFrame."""
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        filepath = tmp_path / "test.h5"

        _save_to_hdf5(df, filepath)
        loaded_df = _load_from_hdf5(filepath)

        pd.testing.assert_frame_equal(df, loaded_df)


class TestMainLoop:
    """Test main_loop function."""

    def test_main_loop_pickle(self, tmp_path):
        """Test main_loop with pickle format."""

        def dummy_func(basepath):
            return pd.DataFrame({"path": [basepath], "value": [1]})

        basepath = "test_session"
        save_path = str(tmp_path)

        main_loop(basepath, save_path, dummy_func, format_type="pickle")

        # Check file was created
        expected_file = encode_file_path(basepath, save_path, "pickle")
        assert os.path.exists(expected_file)

        # Check contents
        import pickle

        with open(expected_file, "rb") as f:
            result = pickle.load(f)

        expected_df = pd.DataFrame({"path": [basepath], "value": [1]})
        # Pickle preserves dtypes naturally
        pd.testing.assert_frame_equal(result, expected_df, check_dtype=True)

    def test_main_loop_hdf5(self, tmp_path):
        """Test main_loop with HDF5 format."""

        def dummy_func(basepath):
            return pd.DataFrame({"path": [basepath], "value": [1]})

        basepath = "test_session"
        save_path = str(tmp_path)

        main_loop(basepath, save_path, dummy_func, format_type="hdf5")

        # Check file was created
        expected_file = encode_file_path(basepath, save_path, "hdf5")
        assert os.path.exists(expected_file)

        # Check contents
        result = _load_from_hdf5(expected_file)
        expected_df = pd.DataFrame({"path": [basepath], "value": [1]})
        # HDF5 now preserves dtypes exactly (including StringDtype na_value)
        pd.testing.assert_frame_equal(result, expected_df, check_dtype=True)

    def test_main_loop_skip_existing(self, tmp_path):
        """Test main_loop skips existing files when overwrite=False."""

        def dummy_func(basepath):
            return pd.DataFrame({"path": [basepath], "value": [1]})

        basepath = "test_session"
        save_path = str(tmp_path)

        # Create file first
        main_loop(basepath, save_path, dummy_func, format_type="pickle")

        # Modify function to return different data
        def modified_func(basepath):
            return pd.DataFrame({"path": [basepath], "value": [999]})

        # Run again with overwrite=False (default)
        main_loop(basepath, save_path, modified_func, format_type="pickle")

        # Check original data is preserved
        expected_file = encode_file_path(basepath, save_path, "pickle")
        import pickle

        with open(expected_file, "rb") as f:
            result = pickle.load(f)

        assert result["value"].iloc[0] == 1  # Original value

    def test_main_loop_overwrite(self, tmp_path):
        """Test main_loop overwrites existing files when overwrite=True."""

        def dummy_func(basepath):
            return pd.DataFrame({"path": [basepath], "value": [1]})

        basepath = "test_session"
        save_path = str(tmp_path)

        # Create file first
        main_loop(basepath, save_path, dummy_func, format_type="pickle")

        # Modify function to return different data
        def modified_func(basepath):
            return pd.DataFrame({"path": [basepath], "value": [999]})

        # Run again with overwrite=True
        main_loop(
            basepath, save_path, modified_func, format_type="pickle", overwrite=True
        )

        # Check data was overwritten
        expected_file = encode_file_path(basepath, save_path, "pickle")
        import pickle

        with open(expected_file, "rb") as f:
            result = pickle.load(f)

        assert result["value"].iloc[0] == 999  # New value

    def test_main_loop_skip_if_error(self, tmp_path):
        """Test main_loop handles errors when skip_if_error=True."""

        def error_func(basepath):
            raise ValueError("Test error")

        basepath = "test_session"
        save_path = str(tmp_path)

        # Should not raise error
        main_loop(basepath, save_path, error_func, skip_if_error=True)

        # Check no file was created
        expected_file = encode_file_path(basepath, save_path, "pickle")
        assert not os.path.exists(expected_file)


class TestRun:
    """Test run function."""

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame with basepath column."""
        return pd.DataFrame(
            {"basepath": ["session1", "session2", "session3"], "other_col": [1, 2, 3]}
        )

    def test_run_pickle(self, sample_df, tmp_path):
        """Test run function with pickle format."""

        def dummy_func(basepath):
            return pd.DataFrame({"session": [basepath], "processed": [True]})

        save_path = str(tmp_path)

        run(sample_df, save_path, dummy_func, parallel=False, format_type="pickle")

        # Check files were created
        for basepath in sample_df["basepath"]:
            expected_file = encode_file_path(basepath, save_path, "pickle")
            assert os.path.exists(expected_file)

    def test_run_hdf5(self, sample_df, tmp_path):
        """Test run function with HDF5 format."""

        def dummy_func(basepath):
            return pd.DataFrame({"session": [basepath], "processed": [True]})

        save_path = str(tmp_path)

        run(sample_df, save_path, dummy_func, parallel=False, format_type="hdf5")

        # Check files were created
        for basepath in sample_df["basepath"]:
            expected_file = encode_file_path(basepath, save_path, "hdf5")
            assert os.path.exists(expected_file)

    @patch("neuro_py.process.batch_analysis.Parallel")
    def test_run_parallel(self, mock_parallel, sample_df, tmp_path):
        """Test run function with parallel processing."""

        def dummy_func(basepath):
            return pd.DataFrame({"session": [basepath], "processed": [True]})

        save_path = str(tmp_path)

        run(sample_df, save_path, dummy_func, parallel=True, format_type="pickle")

        # Check Parallel was called
        mock_parallel.assert_called_once()


class TestLoadResults:
    """Test load_results function."""

    def test_load_results_pickle(self, tmp_path):
        """Test loading results from pickle files."""
        save_path = str(tmp_path)

        # Create test data
        test_data = [
            pd.DataFrame({"session": ["session1"], "value": [1]}),
            pd.DataFrame({"session": ["session2"], "value": [2]}),
            pd.DataFrame({"session": ["session3"], "value": [3]}),
        ]

        # Save test files
        import pickle

        for i, data in enumerate(test_data):
            filepath = tmp_path / f"test_{i}.pkl"
            with open(filepath, "wb") as f:
                pickle.dump(data, f)

        # Load results
        results = load_results(save_path, format_type="pickle")

        # Check results - don't assume order since file loading order is not guaranteed
        assert len(results) == 3
        assert set(results["session"].tolist()) == {"session1", "session2", "session3"}
        assert set(results["value"].tolist()) == {1, 2, 3}

    def test_load_results_hdf5(self, tmp_path):
        """Test loading results from HDF5 files."""
        save_path = str(tmp_path)

        # Create test data
        test_data = [
            pd.DataFrame({"session": ["session1"], "value": [1]}),
            pd.DataFrame({"session": ["session2"], "value": [2]}),
            pd.DataFrame({"session": ["session3"], "value": [3]}),
        ]

        # Save test files
        for i, data in enumerate(test_data):
            filepath = tmp_path / f"test_{i}.h5"
            _save_to_hdf5(data, filepath)

        # Load results
        results = load_results(save_path, format_type="hdf5")

        # Check results - don't assume order since file loading order is not guaranteed
        assert len(results) == 3
        assert set(results["session"].tolist()) == {"session1", "session2", "session3"}
        assert set(results["value"].tolist()) == {1, 2, 3}

    def test_load_results_mixed_formats(self, tmp_path):
        """Test loading results from mixed file formats."""
        save_path = str(tmp_path)

        # Create test data
        df1 = pd.DataFrame({"session": ["session1"], "value": [1]})
        df2 = pd.DataFrame({"session": ["session2"], "value": [2]})

        # Save one as pickle, one as HDF5
        import pickle

        with open(tmp_path / "test1.pkl", "wb") as f:
            pickle.dump(df1, f)

        _save_to_hdf5(df2, tmp_path / "test2.h5")

        # Load results (auto-detect format)
        results = load_results(save_path)

        # Check results
        assert len(results) == 2
        assert set(results["session"].tolist()) == {"session1", "session2"}
        assert set(results["value"].tolist()) == {1, 2}

    def test_load_results_add_filename(self, tmp_path):
        """Test adding filename column to results."""
        save_path = str(tmp_path)

        # Create test data
        df = pd.DataFrame({"session": ["session1"], "value": [1]})

        import pickle

        with open(tmp_path / "test.pkl", "wb") as f:
            pickle.dump(df, f)

        # Load results with filename
        results = load_results(save_path, add_save_file_name=True)

        # Check filename was added
        assert "save_file_name" in results.columns
        assert results["save_file_name"].iloc[0] == "test.pkl"

    def test_load_results_nonexistent_folder(self):
        """Test error handling for nonexistent folder."""
        with pytest.raises(ValueError, match="folder .* does not exist"):
            load_results("/nonexistent/folder")


class TestLoadSpecificData:
    """Test load_specific_data function."""

    def test_load_specific_data_hdf5_single_key(self, tmp_path):
        """Test loading specific key from HDF5 file."""
        # Create test data
        data = {
            "dataframe": pd.DataFrame({"col1": [1, 2, 3]}),
            "array": np.array([4, 5, 6]),
        }

        filepath = tmp_path / "test.h5"
        _save_to_hdf5(data, filepath)

        # Load specific key
        df_only = load_specific_data(filepath, key="dataframe")

        pd.testing.assert_frame_equal(data["dataframe"], df_only)

    def test_load_specific_data_hdf5_all_data(self, tmp_path):
        """Test loading all data from HDF5 file."""
        # Create test data
        data = {
            "dataframe": pd.DataFrame({"col1": [1, 2, 3]}),
            "array": np.array([4, 5, 6]),
        }

        filepath = tmp_path / "test.h5"
        _save_to_hdf5(data, filepath)

        # Load all data
        loaded_data = load_specific_data(filepath)

        pd.testing.assert_frame_equal(data["dataframe"], loaded_data["dataframe"])
        np.testing.assert_array_equal(data["array"], loaded_data["array"])

    def test_load_specific_data_pickle(self, tmp_path):
        """Test loading from pickle file."""
        # Create test data
        data = {"key1": pd.DataFrame({"col1": [1, 2, 3]})}

        filepath = tmp_path / "test.pkl"
        import pickle

        with open(filepath, "wb") as f:
            pickle.dump(data, f)

        # Load specific key
        result = load_specific_data(filepath, key="key1")

        pd.testing.assert_frame_equal(data["key1"], result)

    def test_load_specific_data_key_not_found(self, tmp_path):
        """Test error handling for missing key."""
        # Create test data
        data = {"dataframe": pd.DataFrame({"col1": [1, 2, 3]})}

        filepath = tmp_path / "test.h5"
        _save_to_hdf5(data, filepath)

        # Try to load non-existent key
        assert load_specific_data(filepath, key="nonexistent") is None


# Integration tests
class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_full_pipeline_pickle(self, tmp_path):
        """Test complete pipeline with pickle format."""
        # Create test DataFrame
        df = pd.DataFrame(
            {"basepath": ["session1", "session2"], "metadata": ["info1", "info2"]}
        )

        def analysis_func(basepath):
            return pd.DataFrame(
                {
                    "session": [basepath],
                    "result": [np.random.rand()],
                    "processed": [True],
                }
            )

        save_path = str(tmp_path)

        # Run pipeline
        run(df, save_path, analysis_func, parallel=False, format_type="pickle")

        # Load results
        results = load_results(save_path, format_type="pickle")

        # Check results
        assert len(results) == 2
        assert set(results["session"].tolist()) == {"session1", "session2"}
        assert all(results["processed"])

    def test_full_pipeline_hdf5(self, tmp_path):
        """Test complete pipeline with HDF5 format."""
        # Create test DataFrame
        df = pd.DataFrame(
            {"basepath": ["session1", "session2"], "metadata": ["info1", "info2"]}
        )

        def analysis_func(basepath):
            return {
                "results": pd.DataFrame(
                    {
                        "session": [basepath],
                        "result": [np.random.rand()],
                        "processed": [
                            True
                        ],  # This will become np.True_ when saved to HDF5
                    }
                ),
                "large_array": np.random.rand(100, 100),
            }

        save_path = str(tmp_path)

        # Run pipeline
        run(df, save_path, analysis_func, parallel=False, format_type="hdf5")

        # Test selective loading
        session_files = [f for f in os.listdir(tmp_path) if f.endswith(".h5")]
        assert len(session_files) == 2

        # Load only results (not the large array)
        first_file = os.path.join(tmp_path, session_files[0])
        results_only = load_specific_data(first_file, key="results")

        assert isinstance(results_only, pd.DataFrame)
        assert "session" in results_only.columns
        assert "result" in results_only.columns
        # Updated assertion to handle NumPy boolean
        assert bool(results_only["processed"].iloc[0]) is True


class TestHomogeneousArrayCompatibility:
    """Test _is_homogeneous_array_compatible function."""

    def test_homogeneous_lists(self):
        """Test homogeneous lists return True."""
        assert _is_homogeneous_array_compatible([1, 2, 3, 4])
        assert _is_homogeneous_array_compatible([1.0, 2.0, 3.0])
        assert _is_homogeneous_array_compatible(["a", "b", "c"])
        assert _is_homogeneous_array_compatible([[1, 2], [3, 4]])

    def test_inhomogeneous_lists(self):
        """Test inhomogeneous lists return False."""
        assert not _is_homogeneous_array_compatible(
            [[1, 2], [3, 4, 5]]
        )  # Different lengths
        assert not _is_homogeneous_array_compatible(
            [np.array([1, 2]), np.array([3, 4, 5])]
        )  # Different shapes
        assert _is_homogeneous_array_compatible([1, "string", 3.0])  # Mixed types

    def test_numpy_arrays(self):
        """Test numpy arrays return True."""
        assert _is_homogeneous_array_compatible(np.array([1, 2, 3]))
        assert _is_homogeneous_array_compatible(np.array([[1, 2], [3, 4]]))

    def test_edge_cases(self):
        """Test edge cases."""
        assert _is_homogeneous_array_compatible([])  # Empty list
        assert _is_homogeneous_array_compatible([1])  # Single element
        assert _is_homogeneous_array_compatible(42)  # Scalar


class TestInhomogeneousDataHDF5:
    """Test inhomogeneous data handling in HDF5."""

    def test_save_load_list_of_arrays(self, tmp_path):
        """Test saving and loading list of numpy arrays."""
        data = [np.array([1, 2, 3]), np.array([4, 5, 6, 7]), np.array([8, 9])]

        filepath = tmp_path / "test.h5"

        with h5py.File(filepath, "w") as f:
            _save_inhomogeneous_data_hdf5(f, "test_data", data)

        with h5py.File(filepath, "r") as f:
            loaded_data = _load_inhomogeneous_data_hdf5(f, "test_data")

        assert len(loaded_data) == 3
        np.testing.assert_array_equal(loaded_data[0], data[0])
        np.testing.assert_array_equal(loaded_data[1], data[1])
        np.testing.assert_array_equal(loaded_data[2], data[2])

    def test_save_load_nested_list_of_arrays(self, tmp_path):
        """Test saving and loading nested list of numpy arrays."""
        data = [
            [np.array([1, 2]), np.array([3, 4])],
            [np.array([5, 6, 7]), np.array([8, 9, 10, 11])],
        ]

        filepath = tmp_path / "test.h5"

        with h5py.File(filepath, "w") as f:
            _save_inhomogeneous_data_hdf5(f, "test_data", data)

        with h5py.File(filepath, "r") as f:
            loaded_data = _load_inhomogeneous_data_hdf5(f, "test_data")

        assert len(loaded_data) == 2
        assert len(loaded_data[0]) == 2
        assert len(loaded_data[1]) == 2

        np.testing.assert_array_equal(loaded_data[0][0], data[0][0])
        np.testing.assert_array_equal(loaded_data[0][1], data[0][1])
        np.testing.assert_array_equal(loaded_data[1][0], data[1][0])
        np.testing.assert_array_equal(loaded_data[1][1], data[1][1])

    def test_save_load_pickled_object(self, tmp_path):
        """Test saving and loading complex objects via pickle."""
        data = {"complex": {"nested": [1, 2, {"inner": "value"}]}}

        filepath = tmp_path / "test.h5"

        with h5py.File(filepath, "w") as f:
            _save_inhomogeneous_data_hdf5(f, "test_data", data)

        with h5py.File(filepath, "r") as f:
            loaded_data = _load_inhomogeneous_data_hdf5(f, "test_data")

        assert loaded_data == data

    def test_load_nonexistent_key(self, tmp_path):
        """Test error handling for nonexistent key."""
        filepath = tmp_path / "test.h5"

        with h5py.File(filepath, "w") as f:
            f.create_dataset("dummy", data=[1, 2, 3])

        with h5py.File(filepath, "r") as f:
            with pytest.raises(KeyError):
                _load_inhomogeneous_data_hdf5(f, "nonexistent")


class TestDataFrameHDF5EdgeCases:
    """Test edge cases for DataFrame HDF5 operations."""

    def test_dataframe_with_numeric_columns(self, tmp_path):
        """Test DataFrame with numeric column names."""
        df = pd.DataFrame({0: [1, 2, 3], 1: [4, 5, 6], 2.5: [7, 8, 9]})

        filepath = tmp_path / "test.h5"

        with h5py.File(filepath, "w") as f:
            _save_dataframe_to_hdf5(df, f, "test_df")

        with h5py.File(filepath, "r") as f:
            loaded_df = _load_dataframe_from_hdf5(f["test_df"])

        pd.testing.assert_frame_equal(df, loaded_df)

    def test_dataframe_with_boolean_columns(self, tmp_path):
        """Test DataFrame with boolean column names."""
        df = pd.DataFrame({True: [1, 2, 3], False: [4, 5, 6]})

        filepath = tmp_path / "test.h5"

        with h5py.File(filepath, "w") as f:
            _save_dataframe_to_hdf5(df, f, "test_df")

        with h5py.File(filepath, "r") as f:
            loaded_df = _load_dataframe_from_hdf5(f["test_df"])

        pd.testing.assert_frame_equal(df, loaded_df)

    def test_dataframe_with_mixed_column_types(self, tmp_path):
        """Test DataFrame with mixed column name types."""
        df = pd.DataFrame({"str_col": [1, 2, 3], 42: [4, 5, 6], 3.14: [7, 8, 9]})

        filepath = tmp_path / "test.h5"

        with h5py.File(filepath, "w") as f:
            _save_dataframe_to_hdf5(df, f, "test_df")

        with h5py.File(filepath, "r") as f:
            loaded_df = _load_dataframe_from_hdf5(f["test_df"])

        pd.testing.assert_frame_equal(df, loaded_df)

    def test_dataframe_with_datetime_index(self, tmp_path):
        """Test DataFrame with datetime index."""
        dates = pd.date_range("2023-01-01", periods=3, freq="D")
        df = pd.DataFrame({"values": [1, 2, 3]}, index=dates)

        filepath = tmp_path / "test.h5"

        with h5py.File(filepath, "w") as f:
            _save_dataframe_to_hdf5(df, f, "test_df")

        with h5py.File(filepath, "r") as f:
            loaded_df = _load_dataframe_from_hdf5(f["test_df"])

        # Note: datetime index might not be preserved exactly due to HDF5 limitations
        # but values should be the same
        pd.testing.assert_frame_equal(
            df.reset_index(drop=True), loaded_df.reset_index(drop=True)
        )

    def test_dataframe_with_empty_dataframe(self, tmp_path):
        """Test empty DataFrame."""
        df = pd.DataFrame()

        filepath = tmp_path / "test.h5"

        with h5py.File(filepath, "w") as f:
            _save_dataframe_to_hdf5(df, f, "test_df")

        with h5py.File(filepath, "r") as f:
            loaded_df = _load_dataframe_from_hdf5(f["test_df"])

        assert loaded_df.empty

    def test_dataframe_missing_columns_attr(self, tmp_path):
        """Test loading DataFrame without columns attribute."""
        # Create HDF5 file manually without columns attribute
        filepath = tmp_path / "test.h5"

        with h5py.File(filepath, "w") as f:
            group = f.create_group("test_df")
            group.create_dataset("col1", data=[1, 2, 3])
            group.create_dataset("col2", data=[4, 5, 6])
            # Intentionally not setting columns attribute

        with h5py.File(filepath, "r") as f:
            loaded_df = _load_dataframe_from_hdf5(f["test_df"])

        assert len(loaded_df.columns) == 2
        assert set(loaded_df.columns) == {"col1", "col2"}


class TestMixedDataHDF5EdgeCases:
    """Test edge cases for mixed data HDF5 operations."""

    def test_save_load_complex_mixed_data(self, tmp_path):
        """Test saving and loading complex mixed data structures."""
        data = {
            "dataframe": pd.DataFrame({"col1": [1, 2, 3]}),
            "homogeneous_list": [1, 2, 3, 4],
            "inhomogeneous_arrays": [np.array([1, 2]), np.array([3, 4, 5])],
            "nested_structure": [
                [np.array([1, 2]), np.array([3, 4])],
                [np.array([5, 6, 7])],
            ],
            "scalar_bool": True,
            "scalar_none": None,  # This will be stored as pickled data
            "tuple_data": (1, 2, 3),
        }

        filepath = tmp_path / "test.h5"

        _save_to_hdf5(data, filepath)
        loaded_data = _load_from_hdf5(filepath)

        # Check DataFrame
        pd.testing.assert_frame_equal(data["dataframe"], loaded_data["dataframe"])

        # Check homogeneous list (converted to numpy array)
        np.testing.assert_array_equal(
            np.array(data["homogeneous_list"]), loaded_data["homogeneous_list"]
        )

        # Check inhomogeneous arrays
        assert len(loaded_data["inhomogeneous_arrays"]) == 2
        np.testing.assert_array_equal(
            loaded_data["inhomogeneous_arrays"][0], data["inhomogeneous_arrays"][0]
        )
        np.testing.assert_array_equal(
            loaded_data["inhomogeneous_arrays"][1], data["inhomogeneous_arrays"][1]
        )

        # Check nested structure
        assert len(loaded_data["nested_structure"]) == 2
        assert len(loaded_data["nested_structure"][0]) == 2
        assert len(loaded_data["nested_structure"][1]) == 1
        np.testing.assert_array_equal(
            loaded_data["nested_structure"][0][0], data["nested_structure"][0][0]
        )
        np.testing.assert_array_equal(
            loaded_data["nested_structure"][0][1], data["nested_structure"][0][1]
        )
        np.testing.assert_array_equal(
            loaded_data["nested_structure"][1][0], data["nested_structure"][1][0]
        )

        # Check scalar bool
        assert loaded_data["scalar_bool"] == data["scalar_bool"]

        # Check None value (stored as pickled data)
        assert loaded_data["scalar_none"] is None

        # Check tuple (might be converted to list or array)
        if isinstance(loaded_data["tuple_data"], np.ndarray):
            np.testing.assert_array_equal(
                loaded_data["tuple_data"], np.array(data["tuple_data"])
            )
        else:
            assert list(loaded_data["tuple_data"]) == list(data["tuple_data"])

        # Verify all original keys are present
        assert set(data.keys()) == set(loaded_data.keys())

    def test_save_data_with_problematic_types(self, tmp_path):
        """Test saving data with types that might cause issues."""

        class CustomObject:
            def __init__(self, value):
                self.value = value

            def __str__(self):
                return f"CustomObject({self.value})"

        data = {
            "custom_object": CustomObject(42),
            "function": lambda x: x * 2,  # Functions can't be pickled easily
        }

        filepath = tmp_path / "test.h5"

        # This should handle the error gracefully
        with patch("builtins.print") as mock_print:
            _save_to_hdf5(data, filepath)

            # Check that warnings were printed for problematic types
            mock_print.assert_called()


class TestMainLoopEdgeCases:
    """Test edge cases for main_loop function."""

    def test_main_loop_with_kwargs(self, tmp_path):
        """Test main_loop passing kwargs to function."""

        def func_with_kwargs(basepath, multiplier=1, prefix=""):
            return pd.DataFrame(
                {
                    "basepath": [basepath],
                    "value": [10 * multiplier],
                    "name": [f"{prefix}{basepath}"],
                }
            )

        basepath = "test_session"
        save_path = str(tmp_path)

        main_loop(
            basepath,
            save_path,
            func_with_kwargs,
            format_type="pickle",
            multiplier=5,
            prefix="processed_",
        )

        # Check result
        expected_file = encode_file_path(basepath, save_path, "pickle")

        with open(expected_file, "rb") as f:
            result = pickle.load(f)

        assert result["value"].iloc[0] == 50  # 10 * 5
        assert result["name"].iloc[0] == "processed_test_session"

    def test_main_loop_function_returns_none(self, tmp_path):
        """Test main_loop when function returns None."""

        def func_returns_none(basepath):
            return None

        basepath = "test_session"
        save_path = str(tmp_path)

        main_loop(basepath, save_path, func_returns_none, format_type="pickle")

        # Check file was created and contains None
        expected_file = encode_file_path(basepath, save_path, "pickle")

        with open(expected_file, "rb") as f:
            result = pickle.load(f)

        assert result is None

    def test_main_loop_error_handling_reraise(self, tmp_path):
        """Test main_loop re-raises errors when skip_if_error=False."""

        def error_func(basepath):
            raise ValueError("Intentional error")

        basepath = "test_session"
        save_path = str(tmp_path)

        with pytest.raises(ValueError, match="Intentional error"):
            main_loop(basepath, save_path, error_func, skip_if_error=False)


class TestRunEdgeCases:
    """Test edge cases for run function."""

    def test_run_with_duplicate_basepaths(self, tmp_path):
        """Test run with duplicate basepaths in DataFrame."""
        df = pd.DataFrame(
            {
                "basepath": ["session1", "session1", "session2"],
                "metadata": ["info1", "info2", "info3"],
            }
        )

        def dummy_func(basepath):
            return pd.DataFrame({"session": [basepath], "count": [1]})

        save_path = str(tmp_path)

        run(df, save_path, dummy_func, parallel=False, format_type="pickle")

        # Should only create files for unique basepaths
        files = glob.glob(os.path.join(save_path, "*.pkl"))
        assert len(files) == 2  # Only session1 and session2

    def test_run_creates_save_directory(self, tmp_path):
        """Test run creates save directory if it doesn't exist."""
        df = pd.DataFrame({"basepath": ["session1"]})

        def dummy_func(basepath):
            return pd.DataFrame({"session": [basepath]})

        # Use a subdirectory that doesn't exist
        save_path = str(tmp_path / "new_directory")
        assert not os.path.exists(save_path)

        run(df, save_path, dummy_func, parallel=False, format_type="pickle")

        # Directory should be created
        assert os.path.exists(save_path)

        # File should be created
        files = glob.glob(os.path.join(save_path, "*.pkl"))
        assert len(files) == 1

    def test_run_with_custom_num_cores(self, tmp_path):
        """Test run with custom number of cores."""
        df = pd.DataFrame({"basepath": ["session1", "session2"]})

        def dummy_func(basepath):
            return pd.DataFrame({"session": [basepath]})

        save_path = str(tmp_path)

        with patch("neuro_py.process.batch_analysis.Parallel") as mock_parallel:
            run(
                df,
                save_path,
                dummy_func,
                parallel=True,
                num_cores=2,
                format_type="pickle",
            )

            # Check that Parallel was called with correct n_jobs
            mock_parallel.assert_called_once_with(n_jobs=2)

    def test_run_with_default_num_cores(self, tmp_path):
        """Test run with default number of cores (all available)."""
        df = pd.DataFrame({"basepath": ["session1"]})

        def dummy_func(basepath):
            return pd.DataFrame({"session": [basepath]})

        save_path = str(tmp_path)

        with patch("neuro_py.process.batch_analysis.Parallel") as mock_parallel:
            with patch(
                "neuro_py.process.batch_analysis.multiprocessing.cpu_count",
                return_value=8,
            ):
                run(df, save_path, dummy_func, parallel=True, format_type="pickle")

                # Check that Parallel was called with all available cores
                mock_parallel.assert_called_once_with(n_jobs=8)


class TestLoadResultsEdgeCases:
    """Test edge cases for load_results function."""

    def test_load_results_with_dict_containing_single_df(self, tmp_path):
        """Test loading results when files contain dict with single DataFrame."""
        save_path = str(tmp_path)

        # Create test data as dict with single DataFrame
        test_data = {"results": pd.DataFrame({"session": ["session1"], "value": [1]})}

        filepath = tmp_path / "test.pkl"
        with open(filepath, "wb") as f:
            pickle.dump(test_data, f)

        # Load results
        results = load_results(save_path, format_type="pickle")

        # Should extract the DataFrame from the dict
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 1
        assert results["session"].iloc[0] == "session1"

    def test_load_results_with_dict_containing_dataframe_key(self, tmp_path):
        """Test loading results when files contain dict with 'dataframe' key."""
        save_path = str(tmp_path)

        # Create test data as dict with 'dataframe' key
        test_data = {
            "dataframe": pd.DataFrame({"session": ["session1"], "value": [1]}),
            "metadata": {"info": "test"},
        }

        filepath = tmp_path / "test.pkl"
        with open(filepath, "wb") as f:
            pickle.dump(test_data, f)

        # Load results
        results = load_results(save_path, format_type="pickle")

        # Should extract the DataFrame from the 'dataframe' key
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 1
        assert results["session"].iloc[0] == "session1"

    def test_load_results_empty_directory(self, tmp_path):
        """Test load_results with empty directory."""
        save_path = str(tmp_path)

        with pytest.raises(ValueError, match="No valid results found"):
            load_results(save_path)

    def test_load_results_invalid_files(self, tmp_path):
        """Test load_results with files that can't be loaded."""
        save_path = str(tmp_path)

        # Create a file that looks like pickle but isn't
        filepath = tmp_path / "invalid.pkl"
        with open(filepath, "w") as f:
            f.write("not a pickle file")

        with pytest.raises(ValueError, match="No valid results found"):
            load_results(save_path)

    def test_load_results_mixed_valid_invalid_files(self, tmp_path):
        """Test load_results with mix of valid and invalid files."""
        save_path = str(tmp_path)

        # Create valid file
        valid_data = pd.DataFrame({"session": ["session1"], "value": [1]})
        valid_filepath = tmp_path / "valid.pkl"
        with open(valid_filepath, "wb") as f:
            pickle.dump(valid_data, f)

        # Create invalid file
        invalid_filepath = tmp_path / "invalid.pkl"
        with open(invalid_filepath, "w") as f:
            f.write("not a pickle file")

        # Should load only valid files
        with patch("builtins.print") as mock_print:
            results = load_results(save_path, format_type="pickle")

            # Should have printed error message for invalid file
            mock_print.assert_called()

        # Should still get valid results
        assert len(results) == 1
        assert results["session"].iloc[0] == "session1"


class TestLoadSpecificDataEdgeCases:
    """Test edge cases for load_specific_data function."""

    def test_load_specific_data_pathlib_path(self, tmp_path):
        """Test load_specific_data with pathlib.Path object."""
        from pathlib import Path

        # Create test data
        data = {"key1": pd.DataFrame({"col1": [1, 2, 3]})}

        filepath = tmp_path / "test.pkl"
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

        # Load with pathlib.Path
        result = load_specific_data(Path(filepath), key="key1")

        pd.testing.assert_frame_equal(data["key1"], result)

    def test_load_specific_data_hdf5_empty_file(self, tmp_path):
        """Test load_specific_data with empty HDF5 file."""
        filepath = tmp_path / "empty.h5"

        # Create empty HDF5 file
        with h5py.File(filepath, "w"):
            pass  # Create empty file

        # Should return empty dict
        result = load_specific_data(filepath)
        assert result == {}

    def test_load_specific_data_pickle_key_from_non_dict(self, tmp_path):
        """Test load_specific_data requesting key from non-dict pickle."""
        # Create test data that's not a dict
        data = pd.DataFrame({"col1": [1, 2, 3]})

        filepath = tmp_path / "test.pkl"
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

        # Should return None when requesting key from non-dict
        result = load_specific_data(filepath, key="nonexistent")
        assert result is None


class TestFilePathEncodingEdgeCases:
    """Test edge cases for file path encoding."""

    def test_encode_decode_path_with_spaces(self, tmp_path):
        """Test encoding/decoding paths with spaces."""
        basepath = "/path/with spaces/in name"
        save_path = str(tmp_path)

        encoded = encode_file_path(basepath, save_path)
        decoded = decode_file_path(encoded)

        assert os.path.normpath(decoded) == os.path.normpath(basepath)

    def test_encode_decode_path_with_special_chars(self, tmp_path):
        """Test encoding/decoding paths with special characters."""
        basepath = "/path/with-special_chars.and.dots/file@name"
        save_path = str(tmp_path)

        encoded = encode_file_path(basepath, save_path)
        decoded = decode_file_path(encoded)

        assert os.path.normpath(decoded) == os.path.normpath(basepath)

    def test_encode_decode_very_long_path(self, tmp_path):
        """Test encoding/decoding very long paths."""
        long_segment = "a" * 100
        basepath = f"/very/long/path/{long_segment}/more/{long_segment}/segments"
        save_path = str(tmp_path)

        encoded = encode_file_path(basepath, save_path)
        decoded = decode_file_path(encoded)

        assert os.path.normpath(decoded) == os.path.normpath(basepath)

    def test_encode_decode_mixed_separators(self, tmp_path):
        """Test encoding/decoding paths with mixed separators."""
        basepath = "C:\\Windows\\Path/with/mixed\\separators"
        save_path = str(tmp_path)

        encoded = encode_file_path(basepath, save_path)
        decoded = decode_file_path(encoded)

        # Should normalize to OS-appropriate separators
        expected = basepath.replace("\\", "/").replace("/", os.sep)
        assert decoded == expected
