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
    basepath = basepath.replace("\\", "/").replace("/", "/")
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

    # Get just the filename portion (without directory path)
    filename = os.path.basename(save_file)

    # Remove extension
    filename = os.path.splitext(filename)[0]

    # Convert back to original path format
    # First decode the encoded separators back to forward slashes
    decoded = filename.replace("---", ":").replace("___", "/")

    # Then convert to OS-specific separators
    return os.path.normpath(decoded)


def _save_to_hdf5(data: Union[pd.DataFrame, dict], filepath: str) -> None:
    """
    Save data to HDF5 format.

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
                if isinstance(value, pd.DataFrame):
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
                    # Convert to numpy array for storage
                    f.create_dataset(key, data=np.array(value))
                else:
                    # For other types, store as string representation
                    f.attrs[f"{key}_str"] = np.bytes_(str(value).encode("utf-8"))


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

    # Save each column
    for col in df.columns:
        try:
            if df[col].dtype == "object":
                # Handle object columns (strings, mixed types)
                string_data = df[col].astype(str).values
                group.create_dataset(col, data=string_data.astype("S"))
            else:
                group.create_dataset(col, data=df[col].values)
        except Exception as e:
            print(f"Warning: Could not save column {col}: {e}")

    # Save index
    try:
        if hasattr(df.index, "values"):
            index_values = df.index.values
            # Convert string index to bytes
            if isinstance(df.index.dtype, object) and isinstance(index_values[0], str):
                index_values = np.array([x.encode("utf-8") for x in index_values])
            group.create_dataset("_index", data=index_values)
        else:
            group.create_dataset("_index", data=np.array(df.index))
    except Exception as e:
        print(f"Warning: Could not save index: {e}")

    # Save column names as strings (not bytes)
    group.attrs["columns"] = list(df.columns)


def _load_from_hdf5(filepath: str) -> Union[pd.DataFrame, dict]:
    """
    Load data from HDF5 format.

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

            # Load datasets
            for key in f.keys():
                if isinstance(f[key], h5py.Group):
                    result[key] = _load_dataframe_from_hdf5(f[key])
                else:
                    result[key] = f[key][:]

            # Load attributes (including scalar values)
            for key, value in f.attrs.items():
                if isinstance(value, (int, float, bool)):
                    result[key] = value
                elif isinstance(value, (np.bytes_, bytes)):
                    result[key] = value.decode("utf-8")  # Convert bytes back to string
                else:
                    result[key] = value

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
        columns = h5_group.attrs["columns"]
        # Handle both bytes and str column names
        if isinstance(columns[0], bytes):
            columns = [col.decode("utf-8") for col in columns]
        elif isinstance(columns[0], str):
            columns = list(columns)  # already strings
    else:
        columns = [key for key in h5_group.keys() if key != "_index"]

    # Load each column
    for col in columns:
        if col in h5_group:
            col_data = h5_group[col][:]
            # Handle string columns
            if col_data.dtype.kind == "S":
                col_data = col_data.astype(str)
            data[col] = col_data

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
) -> Union[pd.DataFrame, dict, np.ndarray]:
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
    Union[pd.DataFrame, dict, np.ndarray]
        Loaded data.
    """
    # Convert path-like objects to string
    filepath_str = str(filepath)

    if filepath_str.endswith(".h5"):
        if key is None:
            return _load_from_hdf5(filepath_str)
        else:
            # Load only specific key
            with h5py.File(filepath_str, "r") as f:
                if key in f:
                    if isinstance(f[key], h5py.Group):
                        return _load_dataframe_from_hdf5(f[key])
                    else:
                        return f[key][:]
                else:
                    raise KeyError(f"Key '{key}' not found in file")
    else:
        # Pickle format - loads everything
        with open(filepath_str, "rb") as f:
            data = pickle.load(f)

        if key is not None and isinstance(data, dict):
            return data[key]
        else:
            return data
