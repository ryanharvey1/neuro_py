import glob
import multiprocessing
import os
import pickle
import traceback
from collections.abc import Callable
from typing import Literal, Optional, Union

import h5py
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm


def encode_file_path(basepath: str, save_path: str, format_type: str = "pickle") -> str:
    """
    Encode file path to be used as a filename.

    Parameters
    ----------
    basepath : str
        Path to the session to be encoded.
    save_path : str
        Directory where the encoded file will be saved.
    format_type : str, optional
        File format type ("pickle" or "hdf5"). Defaults to "pickle".

    Returns
    -------
    str
        Encoded file path suitable for use as a filename.

    Examples
    -------
    >>> basepath = r"Z:\\Data\\AYAold\\AB3\\AB3_38_41"
    >>> save_path = r"Z:\\home\\ryanh\\projects\\ripple_heterogeneity\\replay_02_17_23"
    >>> encode_file_path(basepath, save_path)
    "Z:\\home\\ryanh\\projects\\ripple_heterogeneity\\replay_02_17_23\\Z---___Data___AYAold___AB3___AB3_38_41.pkl"
    """
    # Normalize path separators to forward slashes for consistent encoding
    basepath = basepath.replace("\\", "/")
    save_path = os.path.normpath(save_path)

    # Encode with consistent separators
    encoded_name = basepath.replace("/", "___").replace(":", "---")

    # Add extension
    extension = ".h5" if format_type == "hdf5" else ".pkl"
    encoded_name += extension

    # Join using os.path.join for proper OS-specific path joining
    return os.path.join(save_path, encoded_name)


def decode_file_path(save_file: str) -> str:
    """
    Decode an encoded file path to retrieve the original session path.

    Parameters
    ----------
    save_file : str
        Encoded file path that includes the original session path.

    Returns
    -------
    str
        Original session path before encoding.

    Examples
    -------
    >>> save_file = r"Z:\\home\\ryanh\\projects\\ripple_heterogeneity\\replay_02_17_23\\Z---___Data___AYAold___AB3___AB3_38_41.pkl"
    >>> decode_file_path(save_file)
    "Z:\\Data\\AYAold\\AB3\\AB3_38_41"
    """

    # get basepath from save_file
    basepath = os.path.basename(save_file).replace("___", "/").replace("---", ":")
    # also remove file extension
    basepath = os.path.splitext(basepath)[0]

    # Convert to OS-appropriate path separators
    basepath = basepath.replace("/", os.sep)

    return basepath


def _is_homogeneous_array_compatible(value: object) -> bool:
    """
    Check if a nested list/array structure can be converted to a homogeneous numpy array.

    Parameters
    ----------
    value : any
        The value to check

    Returns
    -------
    bool
        True if the value can be converted to a homogeneous numpy array
    """
    try:
        # Try to create a numpy array - if it fails, it's inhomogeneous
        np.array(value)
        return True
    except (ValueError, TypeError):
        return False


def _save_inhomogeneous_data_hdf5(group: h5py.Group, key: str, value: object) -> None:
    """
    Save inhomogeneous data to HDF5 using different strategies.

    Parameters
    ----------
    group : h5py.Group
        HDF5 group to save to
    key : str
        Key name for the data
    value : any
        The inhomogeneous data to save
    """
    # Strategy 1: Try to save as individual arrays if it's a list of arrays
    if isinstance(value, (list, tuple)) and len(value) > 0:
        # Check if all elements are numpy arrays
        if all(isinstance(item, np.ndarray) for item in value):
            # Create a group for this inhomogeneous array collection
            inhom_group = group.create_group(f"{key}_inhomogeneous")

            # Save each array separately with its index
            for i, arr in enumerate(value):
                inhom_group.create_dataset(f"array_{i}", data=arr)

            # Save metadata about the structure
            inhom_group.attrs["type"] = "inhomogeneous_array_list"
            inhom_group.attrs["length"] = len(value)
            return

        # Check if it's a nested list where each element is a list of arrays
        elif (
            len(value) > 0
            and isinstance(value[0], (list, tuple))
            and len(value[0]) > 0
            and isinstance(value[0][0], np.ndarray)
        ):
            # Handle nested structure like [[array1, array2, ...], [array3, array4, ...]]
            inhom_group = group.create_group(f"{key}_inhomogeneous")

            for i, sublist in enumerate(value):
                subgroup = inhom_group.create_group(f"sublist_{i}")
                for j, arr in enumerate(sublist):
                    subgroup.create_dataset(f"array_{j}", data=arr)
                subgroup.attrs["length"] = len(sublist)

            inhom_group.attrs["type"] = "nested_inhomogeneous_array_list"
            inhom_group.attrs["length"] = len(value)
            return

    # Strategy 2: Fall back to pickle serialization for complex structures
    pickled_data = pickle.dumps(value)
    # Store as bytes dataset
    group.create_dataset(f"{key}_pickled", data=np.void(pickled_data))
    group.attrs[f"{key}_pickled_type"] = "pickled_object"


def _load_inhomogeneous_data_hdf5(group: h5py.Group, key: str) -> object:
    """
    Load inhomogeneous data from HDF5.

    Parameters
    ----------
    group : h5py.Group
        HDF5 group to load from
    key : str
        Key name for the data

    Returns
    -------
    any
        The loaded inhomogeneous data
    """
    # Check for inhomogeneous array collection
    if f"{key}_inhomogeneous" in group:
        inhom_group = group[f"{key}_inhomogeneous"]
        data_type = inhom_group.attrs.get("type", "")

        if data_type == "inhomogeneous_array_list":
            # Load list of arrays
            length = inhom_group.attrs["length"]
            arrays = []
            for i in range(length):
                arrays.append(inhom_group[f"array_{i}"][:])
            return arrays

        elif data_type == "nested_inhomogeneous_array_list":
            # Load nested list of arrays
            length = inhom_group.attrs["length"]
            result = []
            for i in range(length):
                subgroup = inhom_group[f"sublist_{i}"]
                sublength = subgroup.attrs["length"]
                sublist = []
                for j in range(sublength):
                    sublist.append(subgroup[f"array_{j}"][:])
                result.append(sublist)
            return result

    # Check for pickled object
    elif f"{key}_pickled" in group:
        pickled_data = group[f"{key}_pickled"][()]
        return pickle.loads(pickled_data.tobytes())

    else:
        raise KeyError(f"Inhomogeneous data with key '{key}' not found")


def _save_to_hdf5(data: Union[pd.DataFrame, dict], filepath: str) -> None:
    """
    Save data to HDF5 format with support for inhomogeneous arrays.

    Parameters
    ----------
    data : Union[pd.DataFrame, dict]
        Data to save. Can be a DataFrame or dict containing DataFrames/arrays.
    filepath : str
        Path to save the HDF5 file.
    """
    with h5py.File(filepath, "w") as f:
        if isinstance(data, pd.DataFrame):
            # Save DataFrame directly
            _save_dataframe_to_hdf5(data, f, "dataframe")
        elif isinstance(data, dict):
            # Save each item in the dictionary
            for key, value in data.items():
                if value is None:
                    # Handle None values by storing as pickled data
                    _save_inhomogeneous_data_hdf5(f, key, value)
                elif isinstance(value, pd.DataFrame):
                    _save_dataframe_to_hdf5(value, f, key)
                elif isinstance(value, np.ndarray):
                    f.create_dataset(key, data=value)
                elif isinstance(value, (int, float, bool)):
                    f.attrs[key] = value
                elif isinstance(value, str):
                    f.attrs[key] = np.bytes_(
                        value.encode("utf-8")
                    )  # Store string as bytes
                elif isinstance(value, (list, tuple)):
                    # Check if it can be converted to homogeneous array
                    if _is_homogeneous_array_compatible(value):
                        # Convert to numpy array for storage
                        f.create_dataset(key, data=np.array(value))
                    else:
                        # Handle inhomogeneous data
                        _save_inhomogeneous_data_hdf5(f, key, value)
                else:
                    # For other types, try to pickle them
                    try:
                        _save_inhomogeneous_data_hdf5(f, key, value)
                    except Exception as e:
                        # Final fallback: store as string representation
                        f.attrs[f"{key}_str"] = np.bytes_(str(value).encode("utf-8"))
                        print(
                            f"Warning: Saved {key} as string representation due to: {e}"
                        )


def _save_dataframe_to_hdf5(
    df: pd.DataFrame, h5_group: h5py.Group, group_name: str
) -> None:
    """
    Save a pandas DataFrame to an HDF5 group.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    h5_group : h5py.Group
        HDF5 group or file to save to.
    group_name : str
        Name of the group to create.
    """
    group = h5_group.create_group(group_name)

    # Convert column names to strings for HDF5 compatibility
    string_columns = [str(col) for col in df.columns]

    # Save each column
    for orig_col, str_col in zip(df.columns, string_columns):
        try:
            series = df[orig_col]

            # Preserve pandas StringDtype explicitly
            if isinstance(series.dtype, pd.StringDtype):
                string_data = series.astype("string").fillna("").astype(str).values
                dset = group.create_dataset(str_col, data=string_data.astype("S"))
                dset.attrs["pandas_dtype"] = "string"

            elif series.dtype == "object":
                # Handle object columns (strings, mixed types)
                string_data = series.astype(str).values
                group.create_dataset(str_col, data=string_data.astype("S"))

            else:
                group.create_dataset(str_col, data=series.values)
        except Exception as e:
            print(f"Warning: Could not save column {orig_col}: {e}")

    # Save index
    try:
        if hasattr(df.index, "values"):
            index_values = np.array(df.index)
            # Convert string-like index to bytes
            if index_values.dtype.kind in ("O", "U", "S") and len(index_values) > 0:
                encoded = np.array([str(x).encode("utf-8") for x in index_values])
                group.create_dataset("_index", data=encoded)
            else:
                group.create_dataset("_index", data=index_values)
        else:
            group.create_dataset("_index", data=np.array(df.index))
    except Exception as e:
        print(f"Warning: Could not save index: {e}")

    # Save original column names and their types for proper reconstruction
    group.attrs["columns"] = string_columns
    group.attrs["original_columns"] = [
        str(col) for col in df.columns
    ]  # String representation
    group.attrs["column_types"] = [
        str(type(col).__name__) for col in df.columns
    ]  # Type info


def _load_from_hdf5(filepath: str) -> Union[pd.DataFrame, dict]:
    """
    Load data from HDF5 format with support for inhomogeneous arrays.

    Parameters
    ----------
    filepath : str
        Path to the HDF5 file.

    Returns
    -------
    Union[pd.DataFrame, dict]
        Loaded data.
    """
    with h5py.File(filepath, "r") as f:
        if "dataframe" in f and len(f.keys()) == 1:
            # Single DataFrame case
            return _load_dataframe_from_hdf5(f["dataframe"])
        else:
            # Multiple items case
            result = {}

            # First, identify which keys are inhomogeneous data
            inhom_keys = set()
            for key in f.keys():
                if key.endswith("_inhomogeneous"):
                    original_key = key.replace("_inhomogeneous", "")
                    inhom_keys.add(original_key)
                elif key.endswith("_pickled"):
                    original_key = key.replace("_pickled", "")
                    if f"{key}_type" in f.attrs:
                        inhom_keys.add(original_key)

            # Load datasets and groups
            for key in f.keys():
                if isinstance(f[key], h5py.Group):
                    # Check if this is an inhomogeneous array group
                    if key.endswith("_inhomogeneous") or key.endswith("_pickled"):
                        # This will be handled later by the inhomogeneous data loader
                        continue
                    else:
                        # This is a regular DataFrame group
                        result[key] = _load_dataframe_from_hdf5(f[key])
                else:
                    # Skip internal keys that will be handled by inhomogeneous data loader
                    if key.endswith("_inhomogeneous") or key.endswith("_pickled"):
                        continue

                    # Handle datasets - check if it's a scalar dataset
                    dataset = f[key]
                    if dataset.shape == ():  # Scalar dataset
                        result[key] = dataset[()]  # Use [()] for scalar datasets
                    else:
                        result[key] = dataset[:]  # Use [:] for array datasets

            # Load attributes (including scalar values)
            for key, value in f.attrs.items():
                if key.endswith("_pickled_type"):
                    continue  # Skip metadata attributes
                elif isinstance(value, (int, float, bool)):
                    result[key] = value
                elif isinstance(value, (np.bytes_, bytes)):
                    result[key] = value.decode("utf-8")  # Convert bytes back to string
                else:
                    result[key] = value

            # Load inhomogeneous data
            for key in inhom_keys:
                try:
                    result[key] = _load_inhomogeneous_data_hdf5(f, key)
                except Exception as e:
                    print(
                        f"Warning: Could not load inhomogeneous data for key {key}: {e}"
                    )

            return result


def _load_dataframe_from_hdf5(h5_group: h5py.Group) -> pd.DataFrame:
    """
    Load a pandas DataFrame from an HDF5 group.

    Parameters
    ----------
    h5_group : h5py.Group
        HDF5 group containing the DataFrame data.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.
    """
    data = {}

    # Get column names
    if "columns" in h5_group.attrs:
        string_columns = h5_group.attrs["columns"]
        # Handle empty columns case
        if len(string_columns) == 0:
            string_columns = []
        # Handle both bytes and str column names
        elif isinstance(string_columns[0], bytes):
            string_columns = [col.decode("utf-8") for col in string_columns]
        elif isinstance(string_columns[0], str):
            string_columns = list(string_columns)  # already strings
        else:
            # Handle case where columns might be numeric types already
            string_columns = [str(col) for col in string_columns]

        # Try to load original column names if stored as numeric array
        if "_original_columns" in h5_group:
            columns = h5_group["_original_columns"][:].tolist()
        elif "original_columns" in h5_group.attrs and "column_types" in h5_group.attrs:
            # Legacy approach: reconstruct from string representation and type info
            original_columns = h5_group.attrs["original_columns"]
            column_types = h5_group.attrs["column_types"]

            # Convert string representations back to original types
            reconstructed_columns = []
            for orig_col, col_type in zip(original_columns, column_types):
                if col_type == "int64" or col_type == "int":
                    reconstructed_columns.append(int(orig_col))
                elif col_type == "float64" or col_type == "float":
                    reconstructed_columns.append(float(orig_col))
                elif col_type == "bool":
                    reconstructed_columns.append(orig_col.lower() == "true")
                else:
                    reconstructed_columns.append(orig_col)  # Keep as string

            columns = reconstructed_columns
        elif len(string_columns) > 0:
            # Fallback: try to convert numeric strings back to numbers
            columns = []
            for col in string_columns:
                try:
                    # Convert to string first if it's not already
                    col_str = str(col)

                    # Try to convert to float first (to handle both int and float)
                    float_val = float(col_str)

                    # If it's a whole number, convert to int
                    if float_val.is_integer():
                        columns.append(int(float_val))
                    else:
                        columns.append(float_val)

                except (ValueError, TypeError):
                    # Keep as string if conversion fails
                    columns.append(col)
        else:
            # Empty columns case
            columns = []
    else:
        # Fallback: get all keys except index
        string_columns = [key for key in h5_group.keys() if key != "_index"]
        columns = string_columns

    # Load each column using string keys
    for str_col, final_col in zip(string_columns, columns):
        if str_col in h5_group:
            dset = h5_group[str_col]
            col_data = dset[:]
            # Handle fixed-length bytes strings
            if isinstance(col_data, np.ndarray) and col_data.dtype.kind == "S":
                col_data = col_data.astype(str)
            # Handle variable-length bytes arrays (object of bytes)
            elif isinstance(col_data, np.ndarray) and col_data.dtype.kind == "O":
                if len(col_data) and isinstance(col_data[0], (bytes, np.bytes_)):
                    col_data = np.array([x.decode("utf-8") for x in col_data])

            # Restore pandas StringDtype when explicitly marked
            if "pandas_dtype" in dset.attrs and dset.attrs["pandas_dtype"] == "string":
                data[final_col] = pd.Series(col_data, dtype="string")
            else:
                data[final_col] = col_data

    # Load index
    if "_index" in h5_group:
        index = h5_group["_index"][:]
        # Handle string index
        if index.dtype.kind == "S":
            index = index.astype(str)
    else:
        index = None

    return pd.DataFrame(data, index=index, columns=columns)


def main_loop(
    basepath: str,
    save_path: str,
    func: Callable,
    overwrite: bool = False,
    skip_if_error: bool = False,
    format_type: Literal["pickle", "hdf5"] = "pickle",
    **kwargs,
) -> None:
    """
    main_loop: file management & run function

    Parameters
    ----------
    basepath : str
        Path to session.
    save_path : str
        Path to save results to (will be created if it doesn't exist).
    func : Callable
        Function to run on each basepath in df (see run).
    overwrite : bool, optional
        Whether to overwrite existing files in save_path. Defaults to False.
    skip_if_error : bool, optional
        Whether to skip if an error occurs. Defaults to False.
    format_type : Literal["pickle", "hdf5"], optional
        File format to use for saving. Defaults to "pickle".
    kwargs : dict
        Keyword arguments to pass to func (see run).

    Returns
    -------
    None
    """
    # get file name from basepath
    save_file = encode_file_path(basepath, save_path, format_type)

    # if file exists and overwrite is False, skip
    if os.path.exists(save_file) and not overwrite:
        return

    # calc some features
    if skip_if_error:
        try:
            results = func(basepath, **kwargs)
        except Exception:
            traceback.print_exc()
            print(f"Error in {basepath}")
            return
    else:
        results = func(basepath, **kwargs)

    # save file
    if format_type == "hdf5":
        _save_to_hdf5(results, save_file)
    else:
        with open(save_file, "wb") as f:
            pickle.dump(results, f)


def run(
    df: pd.DataFrame,
    save_path: str,
    func: Callable,
    parallel: bool = True,
    verbose: bool = False,
    overwrite: bool = False,
    skip_if_error: bool = False,
    num_cores: int = None,
    format_type: Literal["pickle", "hdf5"] = "pickle",
    **kwargs,
) -> None:
    """
    Run a function on each basepath in the DataFrame and save results.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a 'basepath' column.
    save_path : str
        Path to save results to (will be created if it doesn't exist).
    func : Callable
        Function to run on each basepath (see main_loop).
    parallel : bool, optional
        Whether to run in parallel. Defaults to True.
    verbose : bool, optional
        Whether to print progress. Defaults to False.
    overwrite : bool, optional
        Whether to overwrite existing files in save_path. Defaults to False.
    skip_if_error : bool, optional
        Whether to skip processing if an error occurs. Defaults to False.
    num_cores : int, optional
        Number of CPU cores to use (if None, will use all available cores). Defaults to None.
    format_type : Literal["pickle", "hdf5"], optional
        File format to use for saving. Defaults to "pickle".
    kwargs : dict
        Additional keyword arguments to pass to func.

    Returns
    -------
    None
    """
    # find sessions to run
    basepaths = pd.unique(df.basepath)
    # create save_path if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # run in parallel if parallel is True
    if parallel:
        # get number of cores
        if num_cores is None:
            num_cores = multiprocessing.cpu_count()
        # run in parallel
        Parallel(n_jobs=num_cores)(
            delayed(main_loop)(
                basepath,
                save_path,
                func,
                overwrite,
                skip_if_error,
                format_type,
                **kwargs,
            )
            for basepath in tqdm(basepaths)
        )
    else:
        # run in serial
        for basepath in tqdm(basepaths):
            if verbose:
                print(basepath)
            # run main_loop on each basepath in df
            main_loop(
                basepath,
                save_path,
                func,
                overwrite,
                skip_if_error,
                format_type,
                **kwargs,
            )


def load_results(
    save_path: str,
    verbose: bool = False,
    add_save_file_name: bool = False,
    format_type: Optional[Literal["pickle", "hdf5"]] = None,
) -> pd.DataFrame:
    """
    Load results from pickled pandas DataFrames or HDF5 files in the specified directory.

    Parameters
    ----------
    save_path : str
        Path to the folder containing pickled results.
    verbose : bool, optional
        Whether to print progress for each file. Defaults to False.
    add_save_file_name : bool, optional
        Whether to add a column with the name of the save file. Defaults to False.
    format_type : Optional[Literal["pickle", "hdf5"]], optional
        File format to load. If None, will auto-detect based on file extension.

    Returns
    -------
    pd.DataFrame
        Concatenated pandas DataFrame with all results.

    Raises
    ------
    ValueError
        If the specified folder does not exist or is empty.
    """

    if not os.path.exists(save_path):
        raise ValueError(f"folder {save_path} does not exist")

    # Determine file pattern based on format_type
    if format_type == "pickle":
        file_pattern = "*.pkl"
    elif format_type == "hdf5":
        file_pattern = "*.h5"
    else:
        # Auto-detect: look for both formats
        file_pattern = "*"

    sessions = glob.glob(os.path.join(save_path, file_pattern))

    # Filter by supported extensions if auto-detecting
    if format_type is None:
        sessions = [s for s in sessions if s.endswith((".pkl", ".h5"))]

    # Sort sessions for consistent ordering
    sessions.sort()

    results = []

    for session in sessions:
        if verbose:
            print(session)

        # Determine format based on file extension
        if session.endswith(".h5"):
            try:
                results_ = _load_from_hdf5(session)
            except Exception as e:
                print(f"Error loading HDF5 file {session}: {e}")
                continue
        else:
            try:
                with open(session, "rb") as f:
                    results_ = pickle.load(f)
            except Exception as e:
                print(f"Error loading pickle file {session}: {e}")
                continue

        if results_ is None:
            continue

        # Convert to DataFrame if it's a dict containing a single DataFrame
        if (
            isinstance(results_, dict)
            and len(results_) == 1
            and isinstance(list(results_.values())[0], pd.DataFrame)
        ):
            results_ = list(results_.values())[0]
        elif (
            isinstance(results_, dict)
            and "dataframe" in results_
            and isinstance(results_["dataframe"], pd.DataFrame)
        ):
            results_ = results_["dataframe"]

        # Ensure we have a DataFrame
        if not isinstance(results_, pd.DataFrame):
            if verbose:
                print(f"Skipping {session}: not a DataFrame")
            continue

        if add_save_file_name:
            results_["save_file_name"] = os.path.basename(session)

        results.append(results_)

    if not results:
        raise ValueError(f"No valid results found in {save_path}")

    results = pd.concat(results, ignore_index=True, axis=0)

    return results


def load_specific_data(
    filepath: Union[str, os.PathLike], key: Optional[str] = None
) -> Union[pd.DataFrame, dict, np.ndarray, None]:
    """
    Load specific data from a file (supports selective loading for HDF5).

    Parameters
    ----------
    filepath : Union[str, os.PathLike]
        Path to the file to load. Can be string or path-like object.
    key : Optional[str], optional
        Specific key/dataset to load (only for HDF5). If None, loads all data.

    Returns
    -------
    Union[pd.DataFrame, dict, np.ndarray, None]
        Loaded data, or None if key not found in file.
    """
    filepath_str = str(filepath)

    if filepath_str.endswith(".h5"):
        if key is None:
            return _load_from_hdf5(filepath_str)
        else:
            with h5py.File(filepath_str, "r") as f:
                if key in f:
                    if isinstance(f[key], h5py.Group):
                        return _load_dataframe_from_hdf5(f[key])
                    else:
                        return f[key][:]
                elif f"{key}_inhomogeneous" in f or f"{key}_pickled" in f:
                    return _load_inhomogeneous_data_hdf5(f, key)
                else:
                    # Silent return - empty files are expected
                    return None
    else:
        with open(filepath_str, "rb") as f:
            data = pickle.load(f)

        if key is not None:
            if isinstance(data, dict):
                return data.get(key)  # Returns None if key not found
            else:
                return None  # Return None if key is requested but data is not a dict
        else:
            return data
