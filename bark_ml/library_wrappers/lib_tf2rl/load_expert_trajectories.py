import os
import pickle
import numpy as np
from collections import defaultdict
from typing import Tuple
from bark_ml.library_wrappers.lib_tf2rl.load_save_utils import *
from tf2rl.experiments.utils import load_trajectories

def load_expert_trajectories(dirname: str, normalize_features=False, env=None) -> dict:
    """Loads all found expert trajectories files in the directory.

    Args:
        dirname (str): The directory to search for expert trajectories files

    Raises:
        ValueError: If no valid expert trajctories could be found in the given directory.

    Returns:
        dict: The expert trajectories in tf2rl format: {'obs': [], 'next_obs': [], 'act': []}
    """
    joblib_files = list_files_in_dir(os.path.expanduser(dirname), file_ending='.jblb')
    expert_trajectories = load_trajectories(joblib_files)

    if normalize_features:
        #assert env not None
        for key in ['obses', 'next_obses']:
            expert_trajectories[key] = normalize(features=expert_trajectories[key],
                high=env.observation_space.high,
                low=env.observation_space.low
                )
        expert_trajectories['acts'] = normalize(features=expert_trajectories['acts'],
            high=env.action_space.high,
            low=env.action_space.low
            )
    
    if not expert_trajectories:
        raise ValueError(f"Could not find valid expert trajectories in {dirname}.")

    assert len(expert_trajectories['obses']) == len(expert_trajectories['next_obses'])
    assert len(expert_trajectories['obses']) == len(expert_trajectories['acts'])

    assert expert_trajectories['obses'].shape[1] == 16
    assert expert_trajectories['next_obses'].shape[1] == 16
    assert expert_trajectories['acts'].shape[1] == 2

    return expert_trajectories


def normalize(features, high, low):
    """normalizes a feature vector to be between -1 and 1
    low and high are the original bounds of the feature.
    """
    norm_features = features - low
    norm_features /= (high - low)
    norm_features = norm_features * 2. - 1.
    return norm_features