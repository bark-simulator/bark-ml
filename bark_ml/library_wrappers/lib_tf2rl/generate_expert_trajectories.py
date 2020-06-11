from modules.runtime.scenario.scenario_generation.interaction_dataset_scenario_generation import \
    InteractionDatasetScenarioGeneration
from modules.runtime.commons.parameters import ParameterServer
from modules.runtime.viewer.matplotlib_viewer import MPViewer
from modules.runtime.viewer.video_renderer import VideoRenderer
import os
import os.path
import argparse
import matplotlib.pyplot as plt

from pathlib import Path

import gym
from absl import app
from absl import flags


FLAGS = flags.FLAGS
flags.DEFINE_string("interaction_dataset_path",
                    help="The absolute path to the local interaction dataset clone.",
                    default=None)


def define_scenario(interaction_dataset_path: str,
                    trackIds: list = [63, 64, 65, 66, 67, 68],
                    startTs: int = 232000,
                    endTs: int = 259000,
                    egoTrackId: int = 66,
                    ):
    """
    Creates the parameter server and defines the scenario.
    """
    param_server = ParameterServer()
    param_server["Scenario"]["Generation"]["InteractionDatasetScenarioGeneration"]["MapFilename"] = os.path.join(
        interaction_dataset_path, "DR_DEU_Merging_MT/map/DR_DEU_Merging_MT_v01_shifted.xodr")
    param_server["Scenario"]["Generation"]["InteractionDatasetScenarioGeneration"]["TrackFilename"] = os.path.join(
        interaction_dataset_path, "DR_DEU_Merging_MT/tracks/vehicle_tracks_013.csv")
    param_server["Scenario"]["Generation"]["InteractionDatasetScenarioGeneration"]["TrackIds"] = trackIds
    param_server["Scenario"]["Generation"]["InteractionDatasetScenarioGeneration"]["StartTs"] = startTs
    param_server["Scenario"]["Generation"]["InteractionDatasetScenarioGeneration"]["EndTs"] = endTs
    param_server["Scenario"]["Generation"]["InteractionDatasetScenarioGeneration"]["EgoTrackId"] = egoTrackId
    return param_server

def create_scenario(param_server: ParameterServer, num_scenarios:int=1, random_seed: int =0):
    """
    Creates the actual scenario.
    """
    scenario_generation = InteractionDatasetScenarioGeneration(num_scenarios=1, random_seed=0, params=param_server)
    scenario = scenario_generation.get_scenario(0)
    return scenario

def generate_expert_trajectories(argv):
    """ main """
    interaction_dataset_path = os.path.expanduser(
        str(FLAGS.interaction_dataset_path))
    if not os.path.exists(interaction_dataset_path):
        raise ValueError(
            f"Interaction dataset not found at location: {interaction_dataset_path}")
    param_server = define_scenario(interaction_dataset_path)
    scenario = create_scenario(param_server)

    # Initialize
    sim_step_time = 0.2
    sim_time_steps = 130

    # Run the scenario in a loop
    world_state = scenario.GetWorldState()
    for _ in range(0, sim_time_steps):
        world_state.DoPlanning(sim_step_time)
        world_state.DoExecution(sim_step_time)

if __name__ == '__main__':
    flags.mark_flag_as_required('interaction_dataset_path')
    app.run(generate_expert_trajectories)
