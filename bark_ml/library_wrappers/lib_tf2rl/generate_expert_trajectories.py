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

from modules.runtime.viewer.video_renderer import VideoRenderer

from bark_ml.library_wrappers.lib_tf2rl.load_save_utils import *

# Bark
from modules.runtime.scenario.scenario_generation.interaction_dataset_scenario_generation import \
    InteractionDatasetScenarioGeneration
from modules.runtime.commons.parameters import ParameterServer
from modules.runtime.viewer.matplotlib_viewer import MPViewer
from modules.runtime.viewer.video_renderer import VideoRenderer

# Bark-ML
from bark_ml.observers.nearest_state_observer import NearestAgentsObserver

import gym
from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string("map_file",
                    help="The absolute path to the map file in Xodr format that should be replayed.",
                    default=None)

FLAGS = flags.FLAGS
flags.DEFINE_string("tracks_folder",
                    help="The absolute path to the folder containing the corresponding tracks as csv.",
                    default=None)

flags.DEFINE_string("expert_trajectories_path",
                    help="The absolute path to the folder where the expert trajectories are safed.",
                    default=None)

flags.DEFINE_enum("renderer",
                  "",
                  ["", "matplotlib", "pygame"],
                  "The renderer to use during replay of the interaction dataset.")

def get_track_files(tracks_folder: str):
    """
    Extracts all map file paths from the interaction dataset
    """
    map_files = list_files_in_dir(tracks_folder, '.csv')
    return map_files


def create_parameter_servers_for_scenarios(map_file: str, tracks_folder: str):
    """
    Creates the parameter server and defines the scenario.
    """
    import pandas as pd

    if not map_file.endswith('.xodr'):
        raise ValueError(f"Map file has to be in Xodr file format. Given: {map_file}")

    tracks = get_track_files(tracks_folder)

    param_servers = {}
    for track in tracks:
        df = pd.read_csv(track)
        track_ids = df.track_id.unique()
        start_ts = df.timestamp_ms.min()
        end_ts = df.timestamp_ms.max()

        param_server = ParameterServer()
        param_server["Scenario"]["Generation"]["InteractionDatasetScenarioGeneration"]["MapFilename"] = map_file
        param_server["Scenario"]["Generation"]["InteractionDatasetScenarioGeneration"]["TrackFilename"] = track
        param_server["Scenario"]["Generation"]["InteractionDatasetScenarioGeneration"]["TrackIds"] = list(track_ids)
        param_server["Scenario"]["Generation"]["InteractionDatasetScenarioGeneration"]["StartTs"] = start_ts
        param_server["Scenario"]["Generation"]["InteractionDatasetScenarioGeneration"]["EndTs"] = end_ts
        param_server["Scenario"]["Generation"]["InteractionDatasetScenarioGeneration"]["EgoTrackId"] = track_ids[0]
        map_id = map_file.split('/')[-1].replace('.xodr', '')
        track_id = track.split('/')[-1].replace('.csv', '')
        param_servers[map_id, track_id] = param_server

    return param_servers


def create_scenario(param_server: ParameterServer):
    """
    Creates the actual scenario.
    """
    scenario_generation = InteractionDatasetScenarioGeneration(
        num_scenarios=1, random_seed=0, params=param_server)
    return (scenario_generation.get_scenario(0),
            param_server["Scenario"]["Generation"]["InteractionDatasetScenarioGeneration"]["StartTs"],
            param_server["Scenario"]["Generation"]["InteractionDatasetScenarioGeneration"]["EndTs"])

def is_collinear(values, time_step, threshold=1e-6):
    y1 = values[1] - values[0]
    y2 = values[2] - values[0]
    return abs(y2 - 2 * y1) < threshold

def calculate_action(observations, time_step=0.1, wheel_base=2.7):
    """
    Calculate the action based on the cars previous and current state
    """
    import math
    from numpy import polyfit

    if time_step == 0:
        return [0.0, 0.0]

    thetas = []
    velocities = []
    for element in observations:
        thetas.append(element[2])
        velocities.append(element[3])

    # Calculate streering angle and acceleration
    d_theta = 0
    acceleration = 0
    if len(observations) == 2:
        # Calculate streering angle
        d_theta = (thetas[1] - thetas[0]) / time_step
        # Calculate acceleration
        acceleration = (velocities[1] - velocities[0]) / time_step
    else:
        assert(len(observations) == 3)

        # Check collinearity of theta values
        if is_collinear(thetas, time_step):
            # Fit thetas into a line and get the derivative
            d_theta = (thetas[2] - thetas[1]) / time_step
        else:
            # Fit thetas into a parabola and get the derivative
            p = polyfit([0, time_step, 2*time_step], thetas, 2)
            d_theta = 2*p[0]*thetas[1] + p[1]

        # Check collinearity of velocity values
        if is_collinear(velocities, time_step):
            # Fit fit velocities into a line and get the derivative
            acceleration = (velocities[2] - velocities[1]) / time_step
        else:
            # Fit velocities into a parabola and get the derivative
            p = polyfit([0, time_step, 2*time_step], velocities, 2)
            acceleration = 2*p[0]*velocities[1] + p[1]

    steering_angle = math.atan(wheel_base * d_theta)
    action = [steering_angle, acceleration]

    return action


def measure_world(world_state, observer):
    """
    Append the current timestamp trajectory of active agents to expert trajectory
    """
    agents_valid = list(world_state.agents_valid.keys())

    observations = {}

    observed_worlds = world_state.Observe(agents_valid)

    for obs_world in observed_worlds:
        agent_id = obs_world.ego_agent.id
        obs = np.array(observer.Observe(obs_world))
        observations[agent_id] = {
            # TODO Discuss the 'obs'. 
            # obs_world should be on what we train, maybe stacked with the obs.
            # The obs_world is what our cars sensors are measuring.
            # The 'obs' are just the ego state + two nearest cars.
            # We should change this to use obs_world as 'obs'
            # We have to ask Tobias if the observer is a class that the car can access
            # when our code is deployed. I don't think so. I think the obs_world is 
            # what we should use.
            # "world_state": obs_world,
            "obs": obs,
            "time": world_state.time,
            "merge": None
        }

        if agent_id in agents_valid:
            try:
                observations[agent_id]['merge'] = obs_world.lane_corridor.center_line.bounding_box[0].x(
                ) > 900
            except:
                pass

    return observations


def store_expert_trajectories(map: str, track: str, expert_trajectories_path: str, expert_trajectories: dict):
    """
    Stores the expert trajectories
    """
    directory = os.path.join(expert_trajectories_path,
                             map)
    Path(directory).mkdir(parents=True, exist_ok=True)
    filename = os.path.join(directory, f"{track}.pkl")

    with open(filename, 'wb') as handle:
        pickle.dump(expert_trajectories, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)

    return filename


def simulate_scenario(param_server, sim_time_step: float, renderer: str = ""):
    """
    Simulates one scenario.
    """
    scenario, start_ts, end_ts = create_scenario(param_server)

    sim_steps = int((end_ts - start_ts)/(sim_time_step))
    sim_time_step_seconds = sim_time_step / 1000

    observer = NearestAgentsObserver(param_server)
    expert_trajectories = {}
    
    if renderer:
        fig = plt.figure(figsize=[10, 10])

        if renderer == "pygame":
            from modules.runtime.viewer.pygame_viewer import PygameViewer
            viewer = PygameViewer(params=param_server, use_world_bounds=True, axis=fig.gca())
        else:
            from modules.runtime.viewer.matplotlib_viewer import MPViewer
            viewer = MPViewer(params=param_server, use_world_bounds=True, axis=fig.gca())

    # Run the scenario in a loop
    world_state = scenario.GetWorldState()
    for _ in range(0, sim_steps):
        world_state.Step(sim_time_step_seconds)
    
        if renderer:
            viewer.clear()
            viewer.drawWorld(world_state, scenario._eval_agent_ids)

        observations = measure_world(world_state, observer)

        for agent_id, values in observations.items():
            if agent_id not in expert_trajectories:
                expert_trajectories[agent_id] = defaultdict(list)
            expert_trajectories[agent_id]['obs'].append(values['obs'])
            expert_trajectories[agent_id]['time'].append(values['time'])
            expert_trajectories[agent_id]['merge'].append(values['merge'])
            expert_trajectories[agent_id]['wheelbase'].append(2.7)
    
    if renderer:
        plt.close()
    return expert_trajectories


def generate_expert_trajectories_for_scenario(param_server, sim_time_step: float, renderer: str = ""):
    """
    Simulates a scenario, measures the environment and calculates the actions.
    """
    expert_trajectories = simulate_scenario(param_server, sim_time_step, renderer)

    for agent_id in expert_trajectories:
        num_observations = len(expert_trajectories[agent_id]['obs'])
        for i in range(0, num_observations - 1):
            agent_trajectory = expert_trajectories[agent_id]

            current_time = agent_trajectory["time"][i]
            next_time = agent_trajectory["time"][i + 1]

            expert_trajectories[agent_id]['done'].append(0)

            current_obs = []
            if i == 0:
                current_obs = agent_trajectory["obs"][i:i + 2]
            else:
                current_obs = agent_trajectory["obs"][i - 1:i + 2]

            time_step = next_time - current_time
            action = calculate_action(current_obs, time_step, 2.7)
            
            expert_trajectories[agent_id]['act'].append(action)

        # add zeros depending on the action-state size
        expert_trajectories[agent_id]['act'].append(
            [0] * len(expert_trajectories[agent_id]['act'][0]))
        expert_trajectories[agent_id]['done'].append(1)

        assert len(expert_trajectories[agent_id]['obs']
                   ) == len(expert_trajectories[agent_id]['act'])
        assert len(expert_trajectories[agent_id]['obs']
                   ) == len(expert_trajectories[agent_id]['time'])
        assert len(expert_trajectories[agent_id]['obs']
                   ) == len(expert_trajectories[agent_id]['merge'])
        assert len(expert_trajectories[agent_id]['obs']
                   ) == len(expert_trajectories[agent_id]['done'])
    return expert_trajectories


def generate_and_store_expert_trajectories(map: str, track: str, expert_trajectories_path: str, param_server, sim_time_step: float, renderer: str = ""):
    """
    Generates and stores the expert trajectories for one scenario.
    """
    print(f"********** Simulating: {map}, {track} **********")
    expert_trajectories = generate_expert_trajectories_for_scenario(
        param_server, sim_time_step, renderer)
    filename = store_expert_trajectories(
        map, track, expert_trajectories_path, expert_trajectories)
    print(f"********** Finished: {map}, {track} **********")
    return filename

def main_function(argv):
    """ main """
    map_file = os.path.expanduser(
        str(FLAGS.map_file))
    tracks_folder = os.path.expanduser(
        str(FLAGS.tracks_folder))
    expert_trajectories_path = os.path.expanduser(
        str(FLAGS.expert_trajectories_path))
    if not os.path.exists(map_file):
        raise ValueError(
            f"Maps file not found at location: {map_file}")
    if not os.path.exists(tracks_folder):
        raise ValueError(
            f"Tracks not found at location: {tracks_folder}")
    param_servers = create_parameter_servers_for_scenarios(map_file, tracks_folder)

    sim_time_step = 100
    for map, track in param_servers.keys():
        generate_and_store_expert_trajectories(map, track, expert_trajectories_path, param_servers[(map, track)], sim_time_step, FLAGS.renderer)
 

if __name__ == '__main__':
    flags.mark_flag_as_required('map_file')
    flags.mark_flag_as_required('tracks_folder')
    flags.mark_flag_as_required('expert_trajectories_path')
    app.run(main_function)
