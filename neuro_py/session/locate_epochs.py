import logging
import re
from typing import List, Tuple, Union

import numpy as np
import pandas as pd


def find_pre_task_post(
    env: Union[List[str], np.ndarray], pre_post_label: str = "sleep"
) -> Tuple[Union[np.ndarray, None], Union[List[int], None]]:
    """
    Finds the first contiguous epochs that meet the pre/task/post pattern in the environment list.

    Parameters
    ----------
    env : list or np.ndarray
        List or array of environment labels (e.g., 'sleep', 'wmaze', etc.).
    pre_post_label : str, optional
        Label used to identify pre and post sleep epochs (default is 'sleep').

    Returns
    -------
    dummy : np.ndarray or None
        A boolean array where the identified pre/task/post epochs are marked as True.
        If no pattern is found, returns None.
    indices : list or None
        A list of indices where the pre/task/post epochs are found. If no pattern is found, returns None.

    Example
    -------
    >>> env = ['sleep', 'wmaze', 'sleep']
    >>> find_pre_task_post(env)
    (array([ True,  True,  True]), [0, 1, 2])

    Notes
    -----
    This function identifies a pattern where the pre-task-post epochs are of the form:
    - pre-sleep (pre_post_label)
    - task (any label other than pre_post_label)
    - post-sleep (pre_post_label)

    The function returns the indices of the first occurrence of such a pattern.
    """
    if len(env) < 3:
        return None, None
    numeric_idx = (pre_post_label == env) * 1
    dummy = np.zeros_like(numeric_idx) == 1
    if all(numeric_idx[:3] == [1, 0, 1]):
        dummy[:3] = True
        return dummy, [0, 1, 2]
    else:
        for i in np.arange(len(numeric_idx) + 3):
            if 3 + i > len(numeric_idx):
                return None, None
            if all(numeric_idx[0 + i : 3 + i] == [1, 0, 1]):
                dummy[0 + i : 3 + i] = True
                return dummy, [0, 1, 2] + i


def compress_repeated_epochs(epoch_df, epoch_name=None):
    """
    Compress repeated epochs in an epoch DataFrame. If consecutive epochs have the same name,
    they will be combined into a single epoch with the earliest startTime and the latest stopTime.

    Parameters
    ----------
    epoch_df : pd.DataFrame
        A DataFrame containing epoch information. Must have columns `environment`, `startTime`, and `stopTime`.
    epoch_name : str, optional
        If provided, only compress epochs with this specific name. If None, compress all consecutive epochs with the same name.

    Returns
    -------
    pd.DataFrame
        A DataFrame where consecutive epochs with the same name are compressed into a single epoch.

    Example
    -------
    >>> epoch_df = pd.DataFrame({
    ...     'environment': ['sleep', 'sleep', 'wmaze', 'wmaze', 'sleep'],
    ...     'startTime': [0, 100, 200, 300, 400],
    ...     'stopTime': [99, 199, 299, 399, 499]
    ... })
    >>> compress_repeated_epochs(epoch_df)
      environment  startTime  stopTime
    0       sleep          0       199
    1       wmaze        200       399
    2       sleep        400       499
    """
    if epoch_name is None:
        match = np.zeros([epoch_df.environment.shape[0]])
        match[match == 0] = np.nan
        for i, ep in enumerate(epoch_df.environment[:-1]):
            if np.isnan(match[i]):
                # find match in current and next epoch
                if ep == epoch_df.environment.iloc[i + 1]:
                    match[i : i + 2] = i
                    # given match, see if there are more matches
                    for match_i in np.arange(1, epoch_df.environment[:-1].shape[0]):
                        if i + 1 + match_i == epoch_df.environment.shape[0]:
                            break
                        if ep == epoch_df.environment.iloc[i + 1 + match_i]:
                            match[i : i + 1 + match_i + 1] = i
                        else:
                            break
    else:
        match = np.zeros([epoch_df.environment.shape[0]])
        match[match == 0] = np.nan
        for i, ep in enumerate(epoch_df.environment[:-1]):
            if np.isnan(match[i]):
                # find match in current and next epoch
                if (ep == epoch_df.environment.iloc[i + 1]) & (ep == epoch_name):
                    match[i : i + 2] = i
                    # given match, see if there are more matches
                    for match_i in np.arange(1, epoch_df.environment[:-1].shape[0]):
                        if i + 1 + match_i == epoch_df.environment.shape[0]:
                            break
                        if ep == epoch_df.environment.iloc[i + 1 + match_i]:
                            match[i : i + 1 + match_i + 1] = i
                        else:
                            break

    for i in range(len(match)):
        if np.isnan(match[i]):
            # make nans large numbers that are unlikely to be real epoch
            match[i] = (i + 1) * 2000

    # iter through each epoch indicator to get start and stop
    results = pd.DataFrame()
    no_nan_match = match[~np.isnan(match)]
    for m in pd.unique(no_nan_match):
        temp_dict = {}
        for item in epoch_df.keys():
            temp_dict[item] = epoch_df[match == m][item].iloc[0]

        temp_dict["startTime"] = epoch_df[match == m].startTime.min()
        temp_dict["stopTime"] = epoch_df[match == m].stopTime.max()

        temp_df = pd.DataFrame.from_dict(temp_dict, orient="index").T

        results = pd.concat([results, temp_df], ignore_index=True)
    return results


def find_multitask_pre_post(
    env: pd.Series,
    task_tag: Union[None, str] = None,
    post_sleep_flank: bool = False,
    pre_sleep_common: bool = False,
) -> Union[List[List[int]], None]:
    """
    Find the row indices for pre-task/post-task sleep epochs in the given environment from a DataFrame column.

    Parameters
    ----------
    env : pd.Series
        Column from the DataFrame representing the session epochs data.
    task_tag : str, optional
        A string indicating the task(s) (e.g., "linear", "linear|box") to filter for.
        If None, all non-sleep epochs are considered as task epochs.
    post_sleep_flank : bool, optional
        If True, ensure that the post-task sleep epoch directly follows the task.
    pre_sleep_common : bool, optional
        If True, use the first pre-task sleep epoch as the pre-task sleep for all tasks.

    Returns
    -------
    list of list of int, or None
        A list of indices for pre-task, task, and post-task epochs in the format [pre_task, task, post_task].
        If no such sequence is found, returns None.

    Example
    -------
    >>> epoch_df = pd.DataFrame({
    ...     'environment': ['sleep', 'linear', 'sleep', 'box', 'sleep']
    ... })
    >>> find_multitask_pre_post(epoch_df['environment'], task_tag='linear')
    [[0, 1, 2]]
    """
    # Find the row indices that contain the search string in the specified column
    if task_tag is None:
        task_bool = ~env.str.contains("sleep", case=False)
    else:
        task_bool = env.str.contains(task_tag, case=False)
    sleep_bool = env.str.contains("sleep", case=False)

    task_idx = np.where(task_bool)[0]
    task_idx = np.delete(task_idx, task_idx == 0, 0)
    sleep_idx = np.where(sleep_bool)[0]

    pre_task_post = []
    for task in task_idx:
        temp = sleep_idx - task
        pre_task = sleep_idx[temp < 0]
        post_task = sleep_idx[temp > 0]

        if len(post_task) == 0:
            logging.warning("no post_task sleep for task epoch " + str(task))
        elif len(pre_task) == 0:
            logging.warning("no pre_task sleep for task epoch " + str(task))
        else:
            pre_task_post.append([pre_task[-1], task, post_task[0]])

    if len(pre_task_post) == 0:
        pre_task_post = None

    # search for epochs where the last epoch is 1 more than the first epoch
    if post_sleep_flank and pre_task_post is not None:
        pre_task_post_ = []
        for seq in pre_task_post:
            if seq[-1] - seq[1] == 1:
                pre_task_post_.append(seq)
        pre_task_post = pre_task_post_

    # make the first pre task sleep the same pre task in subsequent tasks
    if pre_sleep_common and pre_task_post is not None:
        pre_task_post_ = []
        for seq in pre_task_post:
            pre_task_post_.append([pre_task_post[0][0], seq[1], seq[2]])
        pre_task_post = pre_task_post_

    return pre_task_post


def find_epoch_pattern(
    env: Union[List[str], pd.Series], pattern: List[str]
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[None, None]]:
    """
    Finds the first occurrence of a contiguous pattern of epochs in the environment list.

    Parameters
    ----------
    env : list or pd.Series
        The environment list or pandas Series representing the epochs.
    pattern : list of str
        The pattern to search for in the environment list.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray) or (None, None)
        Returns a tuple where the first element is a boolean mask indicating the positions of the found pattern,
        and the second element is an array of indices where the pattern occurs.
        If the pattern is not found, returns (None, None).

    Example
    -------
    >>> epoch_df = loading.load_epoch(basepath)
    >>> pattern_idx,_ = find_epoch_pattern(epoch_df.environment,['sleep','linear','sleep'])
    >>> epoch_df.loc[pattern_idx]
        name	                startTime	stopTime	environment	behavioralParadigm	notes
    0	preSleep_210411_064951	0.0000	    9544.56315	sleep	    NaN	                NaN
    1	maze_210411_095201	    9544.5632	11752.80635	linear	    novel	            novel
    2	postSleep_210411_103522	11752.8064	23817.68955	sleep	    novel	            novel
    """

    env = list(env)
    pattern = list(pattern)

    if len(env) < len(pattern):
        return None, None

    dummy = np.zeros(len(env))

    for i in range(len(env) - len(pattern) + 1):
        if pattern == env[i : i + len(pattern)]:
            dummy[i : i + len(pattern)] = 1
            dummy = dummy == 1
            return dummy, np.arange(i, i + len(pattern))
    return None, None


def find_env_paradigm_pre_task_post(
    epoch_df: pd.DataFrame, env: str = "sleep", paradigm: str = "memory"
) -> np.ndarray:
    """
    Find indices of epochs that match a sequence of environment and paradigm
    patterns, specifically looking for a pre-task-post structure.

    Parameters
    ----------
    epoch_df : pd.DataFrame
        DataFrame containing epoch information with columns such as 'environment' and 'behavioralParadigm'.
    env : str, optional
        The environment pattern to search for (default is "sleep").
    paradigm : str, optional
        The behavioral paradigm pattern to search for (default is "memory").

    Returns
    -------
    np.ndarray
        A boolean array where `True` indicates that the epoch is part of a pre-task-post sequence
        (i.e., sleep-task-sleep) based on the provided environment and paradigm.

    Example
    -------
    >>> epoch_df = pd.DataFrame({
    ...     'name': ['EE.042', 'EE.045', 'EE.046', 'EE.049', 'EE.050'],
    ...     'startTime': [0.0, 995.9384, 3336.3928, 5722.444, 7511.244],
    ...     'stopTime': [995.9384, 3336.3928, 5722.444, 7511.244, 9387.644],
    ...     'environment': ['sleep', 'tmaze', 'sleep', 'tmaze', 'sleep'],
    ...     'behavioralParadigm': [np.nan, 'Spontaneous alternation task', np.nan, 'Working memory task', np.nan]
    ... })
    >>> idx = find_env_paradigm_pre_task_post(epoch_df)
    >>> epoch_df[idx]
          name  startTime   stopTime environment        behavioralParadigm
    2  EE.046   3336.3928  5722.444       sleep                        NaN
    3  EE.049   5722.444   7511.244      tmaze         Working memory task
    4  EE.050   7511.244   9387.644       sleep                        NaN
    """
    # compress back to back sleep epochs
    epoch_df_ = compress_repeated_epochs(epoch_df, epoch_name="sleep")
    # make col with env and paradigm
    epoch_df_["sleep_ind"] = (
        epoch_df_.environment + "_" + epoch_df_.behavioralParadigm.astype(str)
    )
    # locate env and paradigm of choice with this col
    epoch_df_["sleep_ind"] = epoch_df_["sleep_ind"].str.contains(env + "|" + paradigm)
    # the pattern we are looking for is all True

    # https://stackoverflow.com/questions/48710783/pandas-find-and-index-rows-that-match-row-sequence-pattern
    pat = np.asarray([True, True, True])
    N = len(pat)
    idx = (
        epoch_df_["sleep_ind"]
        .rolling(window=N, min_periods=N)
        .apply(lambda x: (x == pat).all())
        .mask(lambda x: x == 0)
        .bfill(limit=N - 1)
        .fillna(0)
        .astype(bool)
    ).values
    return idx


def find_pre_task_post_optimize_novel(
    epoch_df: pd.DataFrame, novel_indicators: List[Union[int, str]] = [1, "novel", "1"]
) -> Union[pd.DataFrame, None]:
    """
    Find pre-task-post epochs in the DataFrame, optimizing for novel epochs.

    Parameters
    ----------
    epoch_df : pd.DataFrame
        DataFrame containing epochs information with 'environment' and 'behavioralParadigm' columns.
    novel_indicators : list of [int, str], optional
        List of indicators used to identify novel epochs in the 'behavioralParadigm' column (default is [1, "novel", "1"]).

    Returns
    -------
    pd.DataFrame or None
        A DataFrame with pre-task-post epochs, or None if no such pattern is found.

    Example
    -------
    epoch_df = loading.load_epoch(basepath)
    epoch_df = find_pre_task_post_optimize_novel(epoch_df)
    """
    # set sleep to nan
    epoch_df.loc[epoch_df.environment == "sleep", "behavioralParadigm"] = np.nan
    # Search for novel epochs
    novel_mask = epoch_df.behavioralParadigm.isin(novel_indicators)
    if novel_mask.any():
        # Find the first novel epoch
        idx = np.where(novel_mask)[0][0]
        # Select the first novel epoch and the epochs before and after it
        mask = np.hstack([idx - 1, idx, idx + 1])
        # If any of the epochs are negative, skip (this means the novel epoch was the first epoch)
        if any(mask < 0):
            pass
        else:
            epoch_df_temp = epoch_df.loc[mask]
            # Find pre task post epochs in this subset
            idx = find_pre_task_post(epoch_df_temp.environment)
            # If no pre task post epochs are found, skip
            if idx is None or idx[0] is None:
                pass
            else:
                epoch_df = epoch_df_temp.reset_index(drop=True)
    # Find the first pre task post epoch in epoch_df, if the df was modified that will be used
    idx, _ = find_pre_task_post(epoch_df.environment)
    if idx is None:
        return None
    epoch_df = epoch_df.loc[idx].reset_index(drop=True)
    return epoch_df


def get_experience_level(behavioralParadigm: pd.Series) -> int:
    """
    Extract the experience level from the behavioralParadigm column.

    The experience level is the number of times the animal has run the task,
    inferred from the behavioralParadigm column.

    Parameters
    ----------
    behavioralParadigm : pd.Series
        A single entry or value from the behavioralParadigm column of an epoch.

    Returns
    -------
    int
        The experience level as an integer. Returns NaN if experience cannot be determined.

    Examples
    --------
    experience = get_experience_level(current_epoch_df.iloc[1].behavioralParadigm)
    """
    if behavioralParadigm == "novel":
        experience = 1
    else:
        try:
            # extract first number from string
            experience = int(re.findall(r"\d+", behavioralParadigm)[0])
        except Exception:
            try:
                # extract experience level from behavioralParadigm column if it is a number
                experience = int(behavioralParadigm)
            except Exception:
                experience = np.nan
    return experience
