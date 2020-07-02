import os
import pickle
import numpy as np
from collections import defaultdict
from typing import Tuple
from bark_ml.library_wrappers.lib_tf2rl.load_save_utils import *
from tf2rl.experiments.utils import load_trajectories

def load_expert_trajectories(dirname: str) -> dict:
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
    
    if not expert_trajectories:
        raise ValueError(f"Could not find valid expert trajectories in {dirname}.")

    assert len(expert_trajectories['obses']) == len(expert_trajectories['next_obses'])
    assert len(expert_trajectories['obses']) == len(expert_trajectories['acts'])

    assert expert_trajectories['obses'].shape[1] == 16
    assert expert_trajectories['next_obses'].shape[1] == 16
    assert expert_trajectories['acts'].shape[1] == 2

    return expert_trajectories