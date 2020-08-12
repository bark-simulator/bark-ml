import os
import pickle
import numpy as np
from collections import defaultdict
from typing import Tuple

# BARK-ML imports
from bark_ml.library_wrappers.lib_tf2rl.load_save_utils import *
from bark_ml.library_wrappers.lib_tf2rl.normalization_utils import normalize
from bark_ml.observers.nearest_state_observer import NearestAgentsObserver

# TF2RL imports
from tf2rl.experiments.utils import load_trajectories


def GetFeatureSpace(env):
  observation = env.reset()
  observer = NearestAgentsObserver()
  observer._max_num_vehicles = int(len(observation) / 4) - 1
  observer.Reset(env._world)
  observer._VelocityRange = [0., 50.]
  space = list(np.array([
      observer._world_x_range,
      observer._world_y_range,
      observer._ThetaRange,
      observer._VelocityRange
  ] * (observer._max_num_vehicles + 1)).transpose())

  class FeatureSpace:
    def __init__(self, low, high):
      self.high = high
      self.low = low
  return FeatureSpace(space[0], space[1])


def load_expert_trajectories(
        dirname: str, normalize_features=False, env=None, subset_size=-1) -> (
        dict, float, int):
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
  joblib_files = list_files_in_dir(dirname, file_ending='.jblb')

  if subset_size > len(joblib_files):
    raise ValueError(
      f'Found {len(joblib_files)} expert trajectories. {subset_size} requested. Aborting!')

  if subset_size > 0:
    indices = np.random.choice(len(joblib_files), subset_size, replace=False)
    joblib_files = np.array(joblib_files)[indices]

  expert_trajectories = load_trajectories(joblib_files)
  if not expert_trajectories:
    raise ValueError(f"Could not find valid expert trajectories in {dirname}.")

  if normalize_features:
    assert env is not None, "if normalization is used the environment has to be provided."
    bounds = GetFeatureSpace(env)
    for key in ['obses', 'next_obses']:
      expert_trajectories[key] = normalize(
          feature=expert_trajectories[key],
          feature_space=bounds
      )

    expert_trajectories['acts'] = normalize(
        feature=expert_trajectories['acts'],
        feature_space=env.action_space
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
      raise ValueError(
        f"No expert trajectories in the observation/action space.")

  for key in ['obses', 'next_obses', 'acts']:
    if str(type(np.array(expert_trajectories[key]))) != 'numpy.ndarray':
      expert_trajectories[key] = np.array(expert_trajectories[key])

  assert len(
      expert_trajectories['obses']) == len(
      expert_trajectories['next_obses'])
  assert len(expert_trajectories['obses']) == len(expert_trajectories['acts'])

  assert expert_trajectories['obses'].shape[1] == 16
  assert expert_trajectories['next_obses'].shape[1] == 16
  assert expert_trajectories['acts'].shape[1] == 2

  return expert_trajectories, len(
      expert_trajectories['obses']) / len(joblib_files), len(joblib_files)
