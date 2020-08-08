import os
import pickle
import numpy as np
import joblib
from collections import defaultdict
from typing import Tuple


#from tf2rl.experiments.utils import load_trajectories

from bark_project.bark.runtime.commons.parameters import ParameterServer

from bark_ml.environments.blueprints import ContinuousHighwayBlueprint, \
  ContinuousMergingBlueprint, ContinuousIntersectionBlueprint
from bark_ml.environments.single_agent_runtime import SingleAgentRuntime
from bark_ml.library_wrappers.lib_tf2rl.tf2rl_wrapper import TF2RLWrapper

def load_expert_trajectories(dirname: str, normalize_features=False, env=None, subset_size = -1) -> (dict, float, int):
    """Loads all found expert trajectories files in the directory.

    Args:
        dirname (str): The directory to search for expert trajectories files

    Raises:
        ValueError: If no valid expert trajctories could be found in the given directory.

    Returns:
        dict: The expert trajectories in tf2rl format: {'obs': [], 'next_obs': [], 'act': []}
        float: The average number of trajectory points per trajectory
        int: The number of loaded trajectories
    """
    joblib_files = list_files_in_dir(os.path.expanduser(dirname), file_ending='.jblb')

    if subset_size > len(joblib_files):
        raise ValueError(f'Found {len(joblib_files)} expert trajectories. {subset_size} requested. Aborting!')

    if subset_size > 0:
        indices = np.random.choice(len(joblib_files), subset_size, replace=False)
        joblib_files = np.array(joblib_files)[indices]

    expert_trajectories = load_trajectories(joblib_files)
    if not expert_trajectories:
        raise ValueError(f"Could not find valid expert trajectories in {dirname}.")
    
    if normalize_features:
        assert env is not None, "if normalization is used the environment has to be provided."
        bounds = GetBounds(env)
        for key in ['obses', 'next_obses']:
            expert_trajectories[key] = normalize(features=expert_trajectories[key],
            high=bounds[1],
            low=bounds[0]
            )

        expert_trajectories['acts'] = normalize(features=expert_trajectories['acts'],
            high=env.action_space.high,
            low=env.action_space.low
            )

        valid_obs_act_pairs = list(range(len(expert_trajectories['obses'])))
        for i in range(len(expert_trajectories['obses'])):
            for key in expert_trajectories:
                if np.max(np.abs(expert_trajectories[key][i])) > 1:
                    valid_obs_act_pairs.remove(i)
                    break

        for key in expert_trajectories:
            expert_trajectories[key] = expert_trajectories[key][valid_obs_act_pairs]
    
        if len(expert_trajectories['obses']) == 0:
            raise ValueError(f"No expert trajectories in the observation/action space.")    

    assert len(expert_trajectories['obses']) == len(expert_trajectories['next_obses'])
    assert len(expert_trajectories['obses']) == len(expert_trajectories['acts'])

    assert expert_trajectories['obses'].shape[1] == 16
    assert expert_trajectories['next_obses'].shape[1] == 16
    assert expert_trajectories['acts'].shape[1] == 2

    return expert_trajectories, len(expert_trajectories['obses']) / len(joblib_files), len(joblib_files)

def normalize(features, high, low):
    """normalizes a feature vector to be between -1 and 1
    low and high are the original bounds of the feature.
    """
    norm_features = features - low
    norm_features /= (high - low)
    norm_features = norm_features
    return norm_features

def list_files_in_dir(dir_path: str, file_ending: str = '') -> list:
    """Lists all files in the given directory ending with the given ending.

    Args:
        dir_path (str): The path to the directory in which to search for files
        file_ending (str, optional): The file ending to filter the found files. Defaults to ''.

    Raises:
        ValueError: If the given path is not a directory

    Returns:
        list: The files in the directory, filtered by the ending. 
    """
    if not os.path.isdir(dir_path):
        raise ValueError(f'Cannot list files in {dir_path}. Not a directory.')
    files = [f for f in os.listdir(
        dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    files = [f for f in files if str(f).endswith(file_ending)]
    files = [os.path.join(dir_path, f) for f in files]
    return files

def GetBounds(env):
    env.reset()
    return list(np.array([
        env._observer._world_x_range,
        env._observer._world_y_range,
        env._observer._ThetaRange,
        env._observer._VelocityRange
        ] * (env._observer._max_num_vehicles + 1)).transpose())

def load_trajectories(filenames, max_steps=None):
    assert len(filenames) > 0
    paths = []
    for filename in filenames:
        paths.append(joblib.load(filename))

    def get_obs_and_act(path):
        obses = path['obs'][:-1]
        next_obses = path['obs'][1:]
        actions = path['act'][:-1]
        obses_norm = path['obs_norm'][:-1]
        next_obses_norm = path['obs_norm'][1:]
        if max_steps is not None:
            return obses[:max_steps], next_obses[:max_steps], actions[:max_steps-1]
        else:
            return obses, next_obses,obses_norm, next_obses_norm, actions

    for i, path in enumerate(paths):
        if i == 0:
            obses, next_obses,obses_norm, next_obses_norm, acts = get_obs_and_act(path)
        else:
            obs, next_obs,obs_norm, next_obs_norm, act = get_obs_and_act(path)
            obses = np.vstack((obs, obses))
            next_obses = np.vstack((next_obs, next_obses))
            obses_norm = np.vstack((obs_norm, obses_norm))
            next_obses_norm=np.vstack((next_obs_norm, next_obses_norm))
            acts = np.vstack((act, acts))
    return {'obses': obses, 'next_obses': next_obses, 'obses_norm':obses_norm, 'next_obses_norm':next_obses_norm,'acts': acts}

def store_normalized_trajectories(filenames, output_dir):
    assert len(filenames) > 0
    paths = []
    for filename in filenames:
        head, tail = os.path.split(filename)
        unnormalized_traj=joblib.load(filename)
        normalized_traj={'obs':[], 'act':[]}
        normalized_traj['obs']=unnormalized_traj['obs_norm']
        normalized_traj['act']=unnormalized_traj['act']
        joblib.dump(normalized_traj, os.path.join(output_dir, tail))

#large_data_store/expert_trajectories/sac/merging_not_normalized
dirname="~/Repositories/bark-ml/large_data_store/expert_trajectories/sac/merging_not_normalized"

joblib_files = list_files_in_dir(os.path.expanduser(dirname), file_ending='.jblb')

#indices = np.random.choice(len(joblib_files), 10, replace=False)
indices = np.arange(len(joblib_files))
joblib_files = np.array(joblib_files)[indices]

#store_normalized_trajectories(joblib_files, "/home/marvin/Repositories/bark-ml/large_data_store/expert_trajectories/sac/merging_normalized")


#expert_trajectories = load_trajectories(joblib_files)

params = ParameterServer(filename="examples/example_params/gail_params.json")

params["World"]["remove_agents_out_of_map"] = True
params["ML"]["Settings"]["GPUUse"] = -1

# create environment
bp = ContinuousMergingBlueprint(params,
                                number_of_senarios=1,
                                random_seed=0)


env = SingleAgentRuntime(blueprint=bp,
                        render=False)

# wrapped environment for compatibility with tf2rl
wrapped_env = TF2RLWrapper(env, 
normalize_features=params["ML"]["Settings"]["NormalizeFeatures"])

#for keys in expert_trajectories.keys():
#   print (keys)

bounds = GetBounds(env)
for key in ['obses', 'next_obses']:
    expert_trajectories[key] = normalize(features=expert_trajectories[key],
    high=bounds[1],
    low=bounds[0]
    )

print(expert_trajectories['obses'])

diffs=expert_trajectories['obses']-expert_trajectories['obses_norm']

print(diffs)
