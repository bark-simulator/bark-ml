import os
import os.path
import argparse
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

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
flags.DEFINE_string("interaction_dataset_path",
                    help="The absolute path to the local interaction dataset clone.",
                    default=None)

FLAGS = flags.FLAGS
flags.DEFINE_string("expert_trajectories_path",
                    help="The absolute path to the folder where the expert trajectories are safed.",
                    default=None)


def list_files_in_dir(dir_path: str, file_ending: str):
    """
    Lists all files in the given dir ending with the given ending.
    """
    files = [f for f in os.listdir(
        dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    files = [f for f in files if str(f).endswith(file_ending)]
    files = [os.path.join(dir_path, f) for f in files]
    return files


def list_dirs_in_dir(dir_path: str):
    """
    Lists all dirs in the given dir.
    """
    dirs = [f for f in os.listdir(
        dir_path) if not os.path.isfile(os.path.join(dir_path, f)) and not f == '.git']
    dirs = [os.path.join(dir_path, f) for f in dirs]
    return dirs


def get_map_files(interaction_dataset_path: str):
    """
    Extracts all track file paths from the interaction dataset
    """
    map_files = []
    for scene in list_dirs_in_dir(interaction_dataset_path):
        map_files.extend(list_files_in_dir(
            os.path.join(scene, "map"), '.xodr'))
    return map_files


def get_track_files(interaction_dataset_path: str):
    """
    Extracts all map file paths from the interaction dataset
    """
    map_files = []
    for scene in list_dirs_in_dir(interaction_dataset_path):
        map_files.extend(list_files_in_dir(
            os.path.join(scene, "tracks"), '.csv'))
    return map_files


def create_parameter_servers_for_scenarios(interaction_dataset_path: str):
    """
    Creates the parameter server and defines the scenario.
    """
    import pandas as pd

    maps = get_map_files(interaction_dataset_path)
    tracks = get_track_files(interaction_dataset_path)

    param_servers = {}
    for track in tracks:
        for map in maps:
            df = pd.read_csv(track)
            track_ids = df.track_id.unique()
            start_ts = df.timestamp_ms.min()
            end_ts = df.timestamp_ms.max()

            param_server = ParameterServer()
            param_server["Scenario"]["Generation"]["InteractionDatasetScenarioGeneration"]["MapFilename"] = map
            param_server["Scenario"]["Generation"]["InteractionDatasetScenarioGeneration"]["TrackFilename"] = track
            param_server["Scenario"]["Generation"]["InteractionDatasetScenarioGeneration"]["TrackIds"] = track_ids
            param_server["Scenario"]["Generation"]["InteractionDatasetScenarioGeneration"]["StartTs"] = start_ts
            param_server["Scenario"]["Generation"]["InteractionDatasetScenarioGeneration"]["EndTs"] = end_ts
            param_server["Scenario"]["Generation"]["InteractionDatasetScenarioGeneration"]["EgoTrackId"] = track_ids[0]
            map_id = map.replace(
                '/map', '').replace(os.path.join(interaction_dataset_path, ''), '').replace('.xodr', '')
            track_id = track.replace(
                '/tracks', '').replace(os.path.join(interaction_dataset_path, ''), '').replace('.csv', '')
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


def calculate_action(current_observation, next_observation, current_time, next_time, wheel_base):
    """
    Calculate the action based on the cars previous and current state
    """
    import math

    delta_t = next_time - current_time
    if delta_t == 0:
        print("Zero division")
        return [0, 0]

    action = []

    # Calculate streering angle
    d_theta = (next_observation[2] - current_observation[2]) / delta_t
    steering_angle = math.atan(wheel_base * d_theta)
    action.append(steering_angle)

    # Calculate acceleration
    acceleration = (next_observation[3] - current_observation[3]) / delta_t
    action.append(acceleration)

    return action


def measure_world(world_state, observer):
    """
    Append the current timestamp trajectory of active agents to expert trajectory
    """
    agents_valid = list(world_state.agents_valid.keys())

    observations = {}

    for obs_world in world_state.Observe(agents_valid):
        agent_id = obs_world.ego_agent.id
        observations[agent_id] = {
            "obs": observer.Observe(obs_world),
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
    filename = os.path.join(directory, f"{track.split('/')[1]}.pkl")

    with open(filename, 'wb') as handle:
        pickle.dump(expert_trajectories, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)


def simulate_scenario(param_server, speed_factor=1):
    """
    Simulates one scenario.
    """
    scenario, start_ts, end_ts = create_scenario(param_server)

    sim_step_time = 100*speed_factor/1000
    sim_steps = int((end_ts - start_ts)/(sim_step_time*1000))

    observer = NearestAgentsObserver(param_server)
    expert_trajectories = {}

    # Run the scenario in a loop
    world_state = scenario.GetWorldState()
    for _ in range(0, sim_steps):
        world_state.DoPlanning(sim_step_time)
        world_state.DoExecution(sim_step_time)

        observations = measure_world(world_state, observer)

        for agent_id, values in observations.items():
            if agent_id not in expert_trajectories:
                expert_trajectories[agent_id] = defaultdict(list)
            expert_trajectories[agent_id]['obs'].append(values['obs'])
            expert_trajectories[agent_id]['time'].append(values['time'])
            expert_trajectories[agent_id]['merge'].append(values['merge'])
    return expert_trajectories


def generate_expert_trajectories_for_scenario(param_server, speed_factor=1):
    """
    Simulates a scenario, measures the environment and calculates the actions.
    """
    expert_trajectories = simulate_scenario(param_server, speed_factor)

    for agent_id in expert_trajectories:
        num_observations = len(expert_trajectories[agent_id]['obs'])
        for i in range(0, num_observations - 1):
            agent_trajectory = expert_trajectories[agent_id]

            current_time = agent_trajectory["time"][i]
            next_time = agent_trajectory["time"][i + 1]

            current_observation = agent_trajectory["obs"][i]
            next_observation = agent_trajectory["obs"][i + 1]

            action = calculate_action(
                current_observation, next_observation, current_time, next_time)
            expert_trajectories[agent_id]['act'].append(action)
            expert_trajectories[agent_id]['done'].append(0)

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


def generate_and_store_expert_trajectories(map: str, track: str, expert_trajectories_path: str, param_server):
    """
    Generates and stores the expert trajectories for one scenario.
    """
    print(f"********** Simulating: {map}, {track} **********")
    expert_trajectories = generate_expert_trajectories_for_scenario(
        param_server)
    store_expert_trajectories(
        map, track, expert_trajectories_path, expert_trajectories)
    print(f"********** Finished: {map}, {track} **********")


def main_function(argv):
    """ main """
    interaction_dataset_path = os.path.expanduser(
        str(FLAGS.interaction_dataset_path))
    expert_trajectories_path = os.path.expanduser(
        str(FLAGS.expert_trajectories_path))
    if not os.path.exists(interaction_dataset_path):
        raise ValueError(
            f"Interaction dataset not found at location: {interaction_dataset_path}")
    param_servers = create_parameter_servers_for_scenarios(
        interaction_dataset_path)

    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = []
        for map, track in param_servers.keys():
            futures.append(executor.submit(generate_and_store_expert_trajectories,
                                           map, track, expert_trajectories_path, param_servers[(map, track)]))

        for future in futures:
            future.result()


if __name__ == '__main__':
    flags.mark_flag_as_required('interaction_dataset_path')
    flags.mark_flag_as_required('expert_trajectories_path')
    app.run(main_function)
