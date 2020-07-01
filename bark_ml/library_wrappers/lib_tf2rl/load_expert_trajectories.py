import os
import pickle
import numpy as np
from collections import defaultdict
from typing import Tuple
from bark_ml.library_wrappers.lib_tf2rl.load_save_utils import *

def load_expert_trajectories(dirname: str) -> Tuple[dict, float]:
    """Loads all expert trajectories and flattens the observations and actions.

    Args:
        dirname (str): The directory to search for expert trajectories files

    Returns:
        Tuple[dict, float]: The expert trajectories, The time between two consecutive observations
    """
    expert_trajectory_files = load_expert_trajectory_files(dirname)
    expert_trajectories = {'obs': [], 'next_obs': [], 'act': []}

    for _, content in expert_trajectory_files.items():
        for key, value in content.items():
            expert_trajectories[key].extend(value)

    return expert_trajectories

def load_expert_trajectory_files(dirname: str) -> dict:
    """Loads all found pickle files in the directory.

    Args:
        dirname (str): The directory to search for expert trajectories files

    Raises:
        ValueError: If no valid expert trajctories could be found in the given directory.

    Returns:
        dict: The expert trajectories by filename
    """
    pickle_files = list_files_in_dir(dirname, file_ending='.pkl')
    expert_trajectories = {}

    for pickle_file in pickle_files:
        loaded_expert_trajectories = load_expert_trajectory_file(pickle_file)
        if loaded_expert_trajectories:
            expert_trajectories[pickle_file] = loaded_expert_trajectories

    if not expert_trajectories:
        raise ValueError(f"Could not find valid expert trajectories in {dirname}.")
    return expert_trajectories

def load_expert_trajectory_file(filepath: str) -> dict:
    """Loads the given expert trajectory pickle file.

    Args:
        filepath (str): The path to the expert trajectory file

    Raises:
        ValueError: The given path does not exist
        ValueError: The given path is not a .pkl file

    Returns:
        dict: The expert trajectories in the file
    """
    if not os.path.exists(filepath):
        raise ValueError(f"{filepath} does not exist.")

    if not filepath.endswith(".pkl"):
        raise ValueError(f"{filepath} is not a .pkl file.")

    with open(filepath, "rb") as pickle_file:
        expert_trajectories = dict(pickle.load(pickle_file))

    if set(expert_trajectories.keys()) != set(['obs', 'next_obs', 'act']):
        return None

    # for key, value in expert_trajectories.items():
    #     if ("obs" not in value or
    #             "act" not in value or
    #             "done" not in value or
    #             "time" not in value or
    #             "merge" not in value):
    #         return None

    return expert_trajectories
