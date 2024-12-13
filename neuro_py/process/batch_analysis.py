import glob
import multiprocessing
import os
import pickle
import traceback
from collections.abc import Callable

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm


def encode_file_path(basepath: str, save_path: str) -> str:
    """
    Encode file path to be used as a filename.

    Parameters
    ----------
    basepath : str
        Path to the session to be encoded.
    save_path : str
        Directory where the encoded file will be saved.

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
    # normalize paths
    basepath = os.path.normpath(basepath)
    save_path = os.path.normpath(save_path)
    # encode file path with unlikely characters
    save_file = os.path.join(
        save_path, basepath.replace(os.sep, "___").replace(":", "---") + ".pkl"
    )
    return save_file


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
    basepath = os.path.basename(save_file).replace("___", os.sep).replace("---", ":")
    # also remove file extension
    basepath = os.path.splitext(basepath)[0]

    return basepath


def main_loop(
    basepath: str,
    save_path: str,
    func: Callable,
    overwrite: bool = False,
    skip_if_error: bool = False,
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
    kwargs : dict
        Keyword arguments to pass to func (see run).

    Returns
    -------
    None
    """
    # get file name from basepath
    save_file = encode_file_path(basepath, save_path)

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
                basepath, save_path, func, overwrite, skip_if_error, **kwargs
            )
            for basepath in tqdm(basepaths)
        )
    else:
        # run in serial
        for basepath in tqdm(basepaths):
            if verbose:
                print(basepath)
            # run main_loop on each basepath in df
            main_loop(basepath, save_path, func, overwrite, skip_if_error, **kwargs)


def load_results(
    save_path: str, verbose: bool = False, add_save_file_name: bool = False
) -> pd.DataFrame:
    """
    Load results from pickled pandas DataFrames in the specified directory.

    Parameters
    ----------
    save_path : str
        Path to the folder containing pickled results.
    verbose : bool, optional
        Whether to print progress for each file. Defaults to False.
    add_save_file_name : bool, optional
        Whether to add a column with the name of the save file. Defaults to False.

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

    sessions = glob.glob(os.path.join(save_path, "*.pkl"))

    results = []

    for session in sessions:
        if verbose:
            print(session)
        with open(session, "rb") as f:
            results_ = pickle.load(f)
        if results_ is None:
            continue

        if add_save_file_name:
            results_["save_file_name"] = os.path.basename(session)

        results.append(results_)

    results = pd.concat(results, ignore_index=True, axis=0)

    return results
