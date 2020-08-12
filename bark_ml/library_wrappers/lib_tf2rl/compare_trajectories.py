import numpy as np
from bark_ml.library_wrappers.lib_tf2rl.load_expert_trajectories import *
from bark_ml.library_wrappers.lib_tf2rl.load_save_utils import *

needed_keys = ['obses', 'next_obses', 'acts']


def calculate_norm(vectors: np.ndarray) -> float:
  """Calculates the mean of the L2 norm of the row vectors.
  Args:
      vectors (np.ndarray): The row vector matrix of values
  Returns:
      float: The mean norm
  """
  rms = np.linalg.norm(vectors, axis=1)
  rms = np.sum(rms)
  return rms / len(vectors)


def compare_trajectories(first: dict, second: dict) -> dict:
  """Compares two trajectories.
  Calculates the L2 norm between each pair of points with the same index in the trajectories.
  For trajectory points that don't have a partner in the other trajectory, the plain L2 norm of the point is calculated.
  The final norms are summed.
  Args:
      first (dict): The first trajectory {'obses': [], 'next_obses': [], 'acts': []}
      second (dict): The first trajectory {'obses': [], 'next_obses': [], 'acts': []}
  Raises:
      ValueError: If a key is not present that identifies a trajectory.
  Returns:
      dict: {'obses': float, 'next_obses': float, 'acts': float}
  """
  distances = {}
  min_length = np.inf

  for key in needed_keys:
    if key not in first or key not in second:
      raise ValueError(
        f'{key} not in one or both dicts. Invalid trajectories.')
    min_length = min(len(first[key]), len(second[key]), min_length)

  for key in needed_keys:
    rms = np.array(first[key][:min_length]
                   ) - np.array(second[key][:min_length])
    distances[key] = calculate_norm(rms)

    if min_length < len(first[key]):
      rms = np.array(first[key][min_length:])
      distances[key] += calculate_norm(rms)
    elif min_length < len(second[key]):
      rms = np.array(second[key][min_length:])
      distances[key] += calculate_norm(rms)

  return distances


def calculate_mean_action(trajectories: dict) -> np.ndarray:
  """Calculates the mean action of the given trajectories over 
  the columns of the row vector matrix of actions.
  Args:
      trajectories (dict): The trajectories that produced the actions.
  Raises:
      ValueError: If no actions can be found in the trajectories.
  Returns:
      np.ndarray: The mean action.
  """
  if 'acts' not in trajectories:
    raise ValueError(f'actions not found. Invalid trajectories.')
  actions = np.array(trajectories['acts'])
  mean_action = np.sum(actions, axis=0) / len(actions)
  return mean_action
