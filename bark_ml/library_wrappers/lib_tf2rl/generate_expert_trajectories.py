import os
import os.path
import argparse
import pickle
import matplotlib.pyplot as plt
from pathlib import Path

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

def list_files_in_dir(dir_path: str, file_ending: str):
    """
    Lists all files in the given dir ending with the given ending.
    """
    files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    files = [f for f in files if str(f).endswith(file_ending)]
    files = [os.path.join(dir_path, f) for f in files]
    return files

def get_map_files(interaction_dataset_path: str):
    """
    Extracts all track file paths from the interaction dataset
    """
    maps_path = os.path.join(interaction_dataset_path, "DR_DEU_Merging_MT/map/")
    return list_files_in_dir(maps_path, '.xodr')

def get_track_files(interaction_dataset_path: str):
    """
    Extracts all map file paths from the interaction dataset
    """
    maps_path = os.path.join(interaction_dataset_path, "DR_DEU_Merging_MT/tracks/")
    return list_files_in_dir(maps_path, '.csv')

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
            # param_server["Scenario"]["Generation"]["InteractionDatasetScenarioGeneration"]["EgoTrackId"] = egoTrackId
            param_servers[(map, track)] = param_server

    return param_servers


def create_scenario_generator(param_server: ParameterServer):
    """
    Creates the actual scenario.
    """
    scenario_generation = InteractionDatasetScenarioGeneration(
        num_scenarios=1, random_seed=0, params=param_server)
    return scenario_generation.get_scenario(0)


def Calc_action(obs, next_obs, delta_t):
    """Calculate the action lists based on the cars current and next state
    Input: 
        obs : list of observations as a sequence of lists [[x,y,theta,v],[x,y,theta,v]]
        next_obs : list of observations as a sequence of lists [[x,y,theta,v],[x,y,theta,v]]
        delta_t : the time between two consecutive frames
    Output: list with actions [steering_rate,accelaration_rate]
    """
    if delta_t == 0:
        print("Zero division")
        return [0, 0]

    action = []

    # calculate steering rate
    action.append((next_obs[2] - obs[2]) / delta_t)

    # just for debugging
    action.append((next_obs[3] - obs[3]) / delta_t)

    return action


# TODO Remove expert_traj param
def Append_timestamp_trajectory(exp_traj, obs_world_list, world, observer):
    """Append the current timestamp trajectory of active agents to expert trajectory
    """
    agents_valid = list(world.agents_valid.keys())

    for obs_world in obs_world_list:

        agent_id = obs_world.ego_agent.id

        if agent_id in agents_valid:
            if not exp_traj[agent_id]['obs']:
                current_obs = observer.Observe(obs_world)
                current_time = [world.time]

                # Calculations for the current step
                exp_traj[agent_id]['obs'].append(current_obs)
                exp_traj[agent_id]['time'].append(current_time)

                # Other values for the current step are calculated during the next time step

                # TBD: check Road-Corridor for first time - for some reasons there is not always a lane available
                try:
                    exp_traj[agent_id]['merge'].append(
                        obs_world.lane_corridor.center_line.bounding_box[0].x() > 900)
                except:
                    pass

            else:
                current_obs = observer.Observe(obs_world)
                current_time = [world.time]

                previous_obs = exp_traj[agent_id]['obs'][-1]
                previous_time = exp_traj[agent_id]['time'][-1]

                # Calculations for previous step
                exp_traj[agent_id]['done'].append([0])
                exp_traj[agent_id]['act'].append(Calc_action(previous_obs, current_obs,
                                                             current_time[0] - previous_time[0]))

                # Calculations for current step
                exp_traj[agent_id]['obs'].append(current_obs)
                exp_traj[agent_id]['time'].append(current_time)

                # Check the road corridor if not already defined
                if not exp_traj[agent_id]['merge']:
                    try:
                        exp_traj[agent_id]['merge'].append(
                            obs_world.lane_corridor.center_line.bounding_box[0].x() > 900)
                    except:
                        pass

    return exp_traj


def generate_expert_trajectories(expert_traj, world_state: object, observer, sim_step_time, sim_time_steps: int):
    """
    Generates the observation
    """
    active_agent_ids = list(world_state.agents.keys())
    observed_worlds = world_state.Observe(active_agent_ids)
    return Append_timestamp_trajectory(
        expert_traj, observed_worlds, world_state, observer)

def store_expert_trajectories(expert_traj: dict):
    """
    Stores the expert trajectories
    """
    import numpy as np

    #Transform lists into np array
    for key in expert_traj:
        expert_traj[key]['obs'] = np.array(expert_traj[key]['obs'])
        expert_traj[key]['act'] = np.array(expert_traj[key]['act'])
        expert_traj[key]['time'] = np.array(expert_traj[key]['time'])
        expert_traj[key]['done'] = np.array(expert_traj[key]['done'])
        expert_traj[key]['merge'] = np.array(expert_traj[key]['merge'])

    #Store to pickle file
    directory = os.path.join(
        os.path.expanduser('~'), 
        "Repositories/gail-4-bark/bark-ml/expert_trajectories")
    Path(directory).mkdir(parents=True, exist_ok=True)
    filename = os.path.join(directory, 'expert_traj.pickle')

    with open(filename, 'wb') as handle:
        pickle.dump(expert_traj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #Well, now it should hopefully be saved

def main_function(argv):
    """ main """
    interaction_dataset_path = os.path.expanduser(
        str(FLAGS.interaction_dataset_path))
    if not os.path.exists(interaction_dataset_path):
        raise ValueError(
            f"Interaction dataset not found at location: {interaction_dataset_path}")
    param_server = create_parameter_servers_for_scenarios(interaction_dataset_path)
    scenario = create_scenario_generator(param_server)

    # Simulation-specific configurations
    startTs = param_server["Scenario"]["Generation"]["InteractionDatasetScenarioGeneration"]["StartTs"]
    endTs = param_server["Scenario"]["Generation"]["InteractionDatasetScenarioGeneration"]["EndTs"]

    speed_factor = 1
    sim_step_time = 100*speed_factor/1000
    sim_steps = int((endTs - startTs)/(sim_step_time*1000))

    observer = NearestAgentsObserver(param_server)

    # Run the scenario in a loop
    world_state = scenario.GetWorldState()

        # TODO Remove
    expert_traj = {}
    for agent_id in list(world_state.agents.keys()):
        expert_traj[agent_id] = {'obs' : [],
                                'act': [],
                                'done': [],
                                'time': [],
                                'merge': []
                                }

    for _ in range(0, sim_steps):
        world_state.DoPlanning(sim_step_time)
        world_state.DoExecution(sim_step_time)
        generate_expert_trajectories(expert_traj, world_state,
         observer, sim_step_time=sim_step_time, sim_time_steps=sim_steps)

    for agent_id in expert_traj:    
        #add zeros depending on the action-state size
        expert_traj[agent_id]['act'].append([0,0])
        expert_traj[agent_id]['done'].append([1])

    store_expert_trajectories(expert_traj)

if __name__ == '__main__':
    flags.mark_flag_as_required('interaction_dataset_path')
    app.run(main_function)
