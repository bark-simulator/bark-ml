import os
import os.path
import argparse
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import numpy as np
from typing import Tuple
import joblib
from IPython.display import clear_output

import math
import gym
from absl import app
from absl import flags

# Bark
from bark.runtime.scenario.scenario_generation.interaction_dataset_scenario_generation import \
    InteractionDatasetScenarioGeneration
from bark.runtime.commons.parameters import ParameterServer
from bark.runtime.viewer.matplotlib_viewer import MPViewer
from bark.runtime.viewer.video_renderer import VideoRenderer

from bark.runtime.viewer.video_renderer import VideoRenderer
from bark.runtime.scenario.scenario import Scenario

# Bark-ML
from bark_ml.observers.nearest_state_observer import NearestAgentsObserver
from bark_ml.observers.observer import StateObserver
from bark_ml.library_wrappers.lib_tf2rl.load_save_utils import *

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "map_file",
    help="The absolute path to the map file in Xodr format that should be replayed.",
    default=None)

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "tracks_dir",
    help="The absolute path to the dir containing the corresponding tracks as csv.",
    default=None)

flags.DEFINE_string(
    "expert_trajectories_path",
    help="The absolute path to the dir where the expert trajectories are safed.",
    default=None)

flags.DEFINE_enum(
    "renderer", "", ["", "matplotlib", "pygame"],
    "The renderer to use during replay of the interaction dataset.")


def get_track_files(tracks_dir: str) -> list:
  """Extracts all track file names in the given directory.

  Args:
      tracks_dir (str): The directory to search for track files

  Returns:
      list: The track files
  """
  return list_files_in_dir(tracks_dir, ".csv")


def create_parameter_servers_for_scenarios(
        map_file: str, tracks_dir: str) -> dict:
  """Generate a parameter server for every track file in the given directory.

  Args:
      map_file (str): The path of the map_file
      tracks_dir (str): The directory containing the track files

  Raises:
      ValueError: Map is not in Xodr format.

  Returns:
      dict: The parameter servers by mao and track files.
  """
  import pandas as pd

  if not map_file.endswith(".xodr"):
    raise ValueError(
        f"Map file has to be in Xodr file format. Given: {map_file}")

  tracks = get_track_files(tracks_dir)

  param_servers = {}
  for track in tracks:
    df = pd.read_csv(track)
    track_ids = df.track_id.unique()
    start_ts = df.timestamp_ms.min()
    end_ts = df.timestamp_ms.max()
    num_rows = len(df.index)

    param_server = ParameterServer()
    param_server["Scenario"]["Generation"][
        "InteractionDatasetScenarioGeneration"]["MapFilename"] = map_file
    param_server["Scenario"]["Generation"][
        "InteractionDatasetScenarioGeneration"]["TrackFilename"] = track
    param_server["Scenario"]["Generation"][
        "InteractionDatasetScenarioGeneration"]["TrackIds"] = list(track_ids)
    param_server["Scenario"]["Generation"][
        "InteractionDatasetScenarioGeneration"]["StartTs"] = start_ts
    param_server["Scenario"]["Generation"][
        "InteractionDatasetScenarioGeneration"]["EndTs"] = end_ts
    param_server["Scenario"]["Generation"][
        "InteractionDatasetScenarioGeneration"]["EgoTrackId"] = track_ids[0]
    map_id = map_file.split("/")[-1].replace(".xodr", "")
    track_id = track.split("/")[-1].replace(".csv", "")
    param_servers[map_id, track_id] = param_server

  return param_servers


def create_scenario(param_server: ParameterServer) -> Tuple[Scenario, float, float]:
  """Creates a bark scenario based on the given parameter server.

  Args:
      param_server (ParameterServer): The parameter server

  Returns:
      Tuple[Scenario, float, float]: The bark scenario, the start timestamp, the end timestamp
  """
  scenario_generation = InteractionDatasetScenarioGeneration(
      num_scenarios=1, random_seed=0, params=param_server)
  return (scenario_generation.get_scenario(0),
          param_server["Scenario"]["Generation"]["InteractionDatasetScenarioGeneration"]["StartTs"],
          param_server["Scenario"]["Generation"]["InteractionDatasetScenarioGeneration"]["EndTs"])


def calculate_action(
        observations: list, time_step=0.1, wheel_base=2.7) -> list:
  """Calculate the action based on some observations.

  Args:
      observations (list): The observations.
      time_step (float, optional): The timestap between two observations. Defaults to 0.1.
      wheel_base (float, optional): The wheelbase of the car form which the observations are. Defaults to 2.7.

  Returns:
      list: The action vector
  """

  assert(time_step >= 0)
  if time_step < 1e-6:
    return [0.0, 0.0]

  thetas = []
  velocities = []
  for element in observations:
    thetas.append(element[2])
    velocities.append(element[3])

  # Calculate streering angle and acceleration
  d_theta = 0
  acceleration = 0
  current_velocity = 0
  if len(observations) == 2:
    # Calculate streering angle
    d_theta = (thetas[1] - thetas[0]) / time_step

    # Calculate acceleration
    current_velocity = velocities[0]
    acceleration = (velocities[1] - current_velocity) / time_step
  else:
    assert(len(observations) == 3)

    # Fit thetas into a parabola and get the derivative
    p = np.polyfit([0, time_step, 2 * time_step], thetas, 2)
    d_theta = 2 * p[0] * thetas[1] + p[1]

    # Check collinearity of velocity values
    current_velocity = velocities[1]
    # Fit velocities into a parabola and get the derivative
    p = np.polyfit([0, time_step, 2 * time_step], velocities, 2)
    acceleration = 2 * p[0] * current_velocity + p[1]

  # Approximate slow velocities to zero to avoid noisy outputs
  if current_velocity <= 1e-5:
    if acceleration <= 1e-5:
      return [0.0, 0.0]
    current_velocity = acceleration * 0.01 * time_step
    d_theta = 0.01 * d_theta

  steering_angle = math.atan((wheel_base * d_theta) / current_velocity)
  action = [acceleration, steering_angle]

  return action


def measure_world(world_state, observer: StateObserver,
                  observer_not_normalized: StateObserver) -> dict:
  """Measures the given world state using the observer.

  Args:
      world_state (World): The bark world state
      observer (StateObserver): The observer used to observe the world with normalization of the observations.
      observer_not_normalized (StateObserver): The observer used to observe the world.

  Returns:
      dict: The observation, time and merge measurement
  """

  agents_valid = list(world_state.agents_valid.keys())

  observations = {}

  observed_worlds = world_state.Observe(agents_valid)

  for obs_world in observed_worlds:
    agent_id = obs_world.ego_agent.id
    obs = np.array(observer.Observe(obs_world))
    obs_not_normalized = np.array(observer_not_normalized.Observe(obs_world))
    observations[agent_id] = {
        "obs_norm": obs,
        "obs": obs_not_normalized,
        "time": world_state.time,
    }

  return observations


def store_expert_trajectories(
        map_file: str, track: str, expert_trajectories_path: str,
        expert_trajectories: dict) -> list:
  """Stores the expert trajectories to a joblib binary file.

  Args:
      map_file (str): The name of the map file that was simulated
      track (str): The name of the track file that was simulated
      expert_trajectories_path (str): The output path
      expert_trajectories (dict): The observations and actions

  Returns:
      list: The final filenames of the stored expert trajectories
  """
  directory = os.path.join(expert_trajectories_path,
                           map_file)
  Path(directory).mkdir(parents=True, exist_ok=True)

  filenames = []

  for agent_id, agent_trajectory in expert_trajectories.items():
    filename = os.path.join(directory, f"{track}_agentid{agent_id}.jblb")

    joblib.dump(expert_trajectories[agent_id], filename)
    filenames.append(filename)
  return filenames


def get_viewer(param_server: ParameterServer, renderer: str):
  """Getter for a viewer to display the simulation.

  Args:
      param_server (ParameterServer): The parameters that specify the scenario.
      renderer (str): The renderer type used. [pygame, matplotlib]

  Returns:
      bark.runtime.viewer.Viewer: A viewer depending on the renderer type
  """
  fig = plt.figure(figsize=[10, 10])

  if renderer == "pygame":
    from bark.runtime.viewer.pygame_viewer import PygameViewer
    viewer = PygameViewer(params=param_server,
                          use_world_bounds=True, axis=fig.gca())
  else:
    from bark.runtime.viewer.matplotlib_viewer import MPViewer
    viewer = MPViewer(params=param_server,
                      use_world_bounds=True, axis=fig.gca())
  return viewer


def simulate_scenario(param_server: ParameterServer, sim_time_step: float,
                      renderer: str = "") -> dict:
  """Simulates one scenario with the given parameters.

  Args:
      param_server (ParameterServer): The parameters that specify the scenario.
      sim_time_step (float): The timestep used for the simulation.
      renderer (str, optional): The renderer used during simulation [pygame, matplotlib]. Defaults to "".

  Returns:
      dict: The expert trajectories recorded during simulation.
  """
  scenario, start_ts, end_ts = create_scenario(param_server)

  sim_steps = int((end_ts - start_ts) / sim_time_step)
  sim_time_step_seconds = sim_time_step / 1000

  param_server["ML"]["StateObserver"]["MaxNumAgents"] = 3

  observer = NearestAgentsObserver(param_server)
  observer_not_normalized = NearestAgentsObserver(param_server)
  observer_not_normalized._NormalizationEnabled = False

  expert_trajectories = {}

  viewer = None
  if renderer:
    viewer = get_viewer(param_server, renderer)

  # Run the scenario in a loop
  world_state = scenario.GetWorldState()
  for i in range(0, sim_steps):
    if i % 25 == 0:
      print(f"Simulated {i}/{sim_steps} timesteps.")
    if viewer:
      if renderer == "matplotlib_jupyter":
        viewer = get_viewer(param_server, renderer)
      clear_output(wait=True)
      viewer.clear()
      viewer.drawWorld(world_state, scenario.eval_agent_ids)

    observations = measure_world(
      world_state, observer, observer_not_normalized)
    for agent_id, values in observations.items():
      if agent_id not in expert_trajectories:
        expert_trajectories[agent_id] = defaultdict(list)
      expert_trajectories[agent_id]["obs_norm"].append(values["obs_norm"])
      expert_trajectories[agent_id]["obs"].append(values["obs"])
      expert_trajectories[agent_id]["time"].append(values["time"])
      expert_trajectories[agent_id]["wheelbase"].append(2.7)

    world_state.Step(sim_time_step_seconds)

  if viewer:
    plt.close()
  return expert_trajectories


def generate_expert_trajectories_for_scenario(param_server: ParameterServer,
                                              sim_time_step: float,
                                              renderer: str = "") -> dict:
  """Simulates a scenario, measures the environment and calculates the actions.

  Args:
      param_server (ParameterServer): The parameters that specify the scenario.
      sim_time_step (float): The timestep used for the simulation.
      renderer (str, optional): The renderer used during simulation [pygame, matplotlib]. Defaults to "".

  Returns:
      dict: The expert trajectories recorded during simulation.
  """
  expert_trajectories = simulate_scenario(
      param_server, sim_time_step, renderer)

  for agent_id in expert_trajectories:
    num_observations = len(expert_trajectories[agent_id]["obs"])
    for i in range(0, num_observations - 1):
      agent_trajectory = expert_trajectories[agent_id]

      current_time = agent_trajectory["time"][i]
      next_time = agent_trajectory["time"][i + 1]

      expert_trajectories[agent_id]["done"].append(0)

      current_obs = []
      # if i == 0:
      #     current_obs = agent_trajectory["obs"][i:i + 2]
      # else:
      #     current_obs = agent_trajectory["obs"][i - 1:i + 2]
      current_obs = agent_trajectory["obs"][i:i + 2]

      time_step = next_time - current_time
      action = calculate_action(current_obs, time_step, 2.7)

      expert_trajectories[agent_id]["act"].append(action)

    # add zeros depending on the action-state size
    expert_trajectories[agent_id]["act"].append(
        [0] * len(expert_trajectories[agent_id]["act"][0]))
    expert_trajectories[agent_id]["done"].append(1)

    assert len(expert_trajectories[agent_id]["obs_norm"]
               ) == len(expert_trajectories[agent_id]["act"])
    assert len(expert_trajectories[agent_id]["obs_norm"]
               ) == len(expert_trajectories[agent_id]["obs"])
    assert len(expert_trajectories[agent_id]["obs_norm"]
               ) == len(expert_trajectories[agent_id]["time"])
    assert len(expert_trajectories[agent_id]["obs_norm"]
               ) == len(expert_trajectories[agent_id]["done"])

  return expert_trajectories


def generate_and_store_expert_trajectories(map_file: str, track: str,
                                           expert_trajectories_path: str,
                                           param_server: ParameterServer,
                                           sim_time_step: float,
                                           renderer: str = "") -> list:
  """Generates and stores the expert trajectories for one scenario.

  Args:
      map_file (str): The name of the map file that was simulated
      track (str): The name of the track file that was simulated
      expert_trajectories_path (str): The output path
      param_server (ParameterServer): The parameters that specify the scenario.
      sim_time_step (float): The timestep used for the simulation.
      renderer (str, optional): The renderer used during simulation [pygame, matplotlib]. Defaults to "".

  Returns:
      list: The final filenames of the stored expert trajectories
  """
  print(f"********** Simulating: {map_file}, {track} **********")
  expert_trajectories = generate_expert_trajectories_for_scenario(
      param_server, sim_time_step, renderer)
  filenames = store_expert_trajectories(
      map_file, track, expert_trajectories_path, expert_trajectories)
  print(f"********** Finished: {map_file}, {track} **********")
  return filenames


def main_function(argv: list):
  """The main function.

  Args:
      argv (list): The command line arguments 

  Raises:
      ValueError: map_file flag is invalid
      ValueError: tracks_dir is invalid
  """
  map_file = os.path.expanduser(
      str(FLAGS.map_file))
  tracks_dir = os.path.expanduser(
      str(FLAGS.tracks_dir))
  expert_trajectories_path = os.path.expanduser(
      str(FLAGS.expert_trajectories_path))
  if not os.path.exists(map_file):
    raise ValueError(
        f"Map file not found at location: {map_file}")
  if not os.path.exists(tracks_dir):
    raise ValueError(
        f"Tracks not found at location: {tracks_dir}")
  param_servers = create_parameter_servers_for_scenarios(
      map_file, tracks_dir)

  sim_time_step = 200
  for map_file, track in param_servers.keys():
    generate_and_store_expert_trajectories(
        map_file, track, expert_trajectories_path,
        param_servers[(map_file, track)],
        sim_time_step, FLAGS.renderer)


if __name__ == "__main__":
  flags.mark_flag_as_required("map_file")
  flags.mark_flag_as_required("tracks_dir")
  flags.mark_flag_as_required("expert_trajectories_path")
  app.run(main_function)
