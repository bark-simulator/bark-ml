import os
import pickle
import numpy as np
from collections import defaultdict

from bark_ml.library_wrappers.lib_tf2rl.load_save_utils import *

def load_expert_trajectories(dirname: str):
    """
    Loads all expert trajectories and flattens the observations and actions.
    """
    expert_trajectory_files = load_expert_trajectory_files(dirname)
    expert_trajectories = defaultdict(list)

    for _, content in expert_trajectory_files.items():
        for _, per_agent_values in content.items():
            for key, value in per_agent_values.items():
                expert_trajectories[key].extend(value)

    dt = expert_trajectories['time'][1] - expert_trajectories['time'][0]

    return {
        'obses': np.array(expert_trajectories['obs'])[:-1],
        'next_obses': np.array(expert_trajectories['obs'])[1:],
        'acts': np.array(expert_trajectories['act'])[:-1],
        # 'dones': expert_trajectories[1:],
        # 'merges': expert_trajectories[1:]
        }, dt

def load_expert_trajectory_files(dirname: str):
    """
    Loads all found pickle files in the directory.
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

def load_expert_trajectory_file(filepath: str):
    """
    Loads the given expert trajectory pickle file.
    """
    if not os.path.exists(filepath):
        raise ValueError(f"{filepath} does not exist.")

    if not filepath.endswith(".pkl"):
        raise ValueError(f"{filepath} is not a .pkl file.")

    with open(filepath, "rb") as pickle_file:
        expert_trajectories = dict(pickle.load(pickle_file))

    for key, value in expert_trajectories.items():
        if ("obs" not in value or
                "act" not in value or
                "done" not in value or
                "time" not in value or
                "merge" not in value):
            return None

    return expert_trajectories
