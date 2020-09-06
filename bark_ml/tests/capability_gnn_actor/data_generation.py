# Copyright (c) 2020 fortiss GmbH
#
# Authors: Silvan Wimmer
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import os
import numpy as np
import time
import pickle
import logging
from collections import OrderedDict

# BARK imports
from bark.runtime.commons.parameters import ParameterServer
from bark.runtime.viewer.matplotlib_viewer import MPViewer
from bark.runtime.viewer.video_renderer import VideoRenderer


# BARK-ML imports
from bark_ml.environments.blueprints import ContinuousHighwayBlueprint, \
  ContinuousMergingBlueprint, ContinuousIntersectionBlueprint
from bark_ml.environments.single_agent_runtime import SingleAgentRuntime
from bark_ml.observers.graph_observer import GraphObserver

class DataGenerator():
  """DataGenerator generates a dataset for a supervised learning setting.

  Args:
      num_scenarios: An integer specifing the number of scenarios to
                     generate data for.
      dump_dir: A path specifing the location where to save the data
                [default: None -> data will not be saved].
      render: A boolean value indicating whether the scenario runs
              should be rendered while generating data (rendering
              slows the data generation process down a lot).
      params: A `ParameterServer` instance that enables further configuration
              of the observer. Defaults to a new instance.
  """

  def __init__(self,
               num_scenarios=3,
               dump_dir=None,
               render=False,
               params=ParameterServer()):
    """Inits DataGenerator with the parameters (see class definition)."""
    self._dump_dir = dump_dir
    self._num_scenarios = num_scenarios
    self._params = params
    self._bp = ContinuousHighwayBlueprint(self._params,\
      number_of_senarios=self._num_scenarios, random_seed=0)
    self._observer = GraphObserver(params=self._params)
    self._env = SingleAgentRuntime(blueprint=self._bp, observer=self._observer,
                                   render=render)

  def run_scenarios(self):
    """Run all scenarios sequentially
    
    Generates a dataset by running a scenario at a time.
    
    Returns:
      data_all_scenarios: list sceanrio_data (with scenario_data being
                          a list of datapoints (see definition of datapoint
                          in _run_scenario))"""
    data_all_scenarios = list()
    for _ in range(0, self._num_scenarios):
      # Generate Scenario
      scenario, idx = self._bp._scenario_generation.get_next_scenario()

      # Log progress of scenario generation in 20% steps
      part = max(1, int(self._num_scenarios/5))
      if idx%part == 0:
        msg = "Running data_generation on scenario "+ str(idx+1) + \
          "/"+ str(self._num_scenarios)
        logging.info(msg)

      data_scenario = self._run_scenario(scenario)
      # Save the data for this run in a seperate file and append it to dataset
      self._save_data(data_scenario)
      data_all_scenarios.append(data_scenario)
      
    return data_all_scenarios

  def _run_scenario(self, scenario):
    """Runs a specific scenario for a predefined number of steps.

    Args:
      scenario: bark-scenario
        
    Returns:
      scenario_data: list containing all data_points of the scenario run
                     (one datapoint is a Dict with entries "observation"
                     and "label". The observation is a Tensor, the label is a
                     OrderedDict with entries "steering" and "acceleration" for
                     the ego node)"""
    
    scenario_data = list()
    # Set boundaries for random actions
    low = np.array([-0.5, -0.02])
    high = np.array([0.5, 0.02])

    self._env.reset(scenario=scenario)
    done = False
    
    while done is False:
      action = np.random.uniform(low=low, high=high, size=(2, ))
      observation, reward, done, info = self._env.step(action)

      # Calculate the labels for the supervised setting
      action_labels = self._calc_labels(observation)

      # Save datum in data_scenario
      datum = dict()
      datum["observation"] = observation
      datum["label"] = action_labels
      scenario_data.append(datum)

    return scenario_data
  
  def _calc_labels(self, observation):
    """Calculates the perfect action labels for one observation

    Args:
      observation: An observation coming from the GraphObserver
    
    Returns:
      action_labels: OrderedDicts containing the perfect action values
                     (steering, acceleration) for the ego agent
    
    Raises:
      KeyError: A needed node attribute seems not to be present in the
                observation
    """
    # Get relevant meta data from the observer
    attribute_keys = self._observer._enabled_node_attribute_keys
    norm_data = self._observer.normalization_data
    normalize = self._observer._normalize_observations
    feature_len = self._observer.feature_len
    
    # Extract features of ego node
    node_features = observation[:feature_len]
      
    # Extract relevant features for further calculation
    goal_theta = node_features[attribute_keys.index("goal_theta")].numpy()
    theta = node_features[attribute_keys.index("theta")].numpy()
    vel = node_features[attribute_keys.index("vel")].numpy()
    goal_vel = node_features[attribute_keys.index("goal_vel")].numpy()
    goal_d = node_features[attribute_keys.index("goal_d")].numpy()

    # Denormalize features if they were normalized at first place
    if normalize:
      goal_theta = self._denormalize_value(goal_theta, norm_data["theta"])
      theta = self._denormalize_value(theta, norm_data["theta"])
      vel = self._denormalize_value(vel, norm_data["vel"])
      goal_vel = self._denormalize_value(goal_vel, norm_data["vel"])
      goal_d = self._denormalize_value(goal_d, norm_data["distance"])

    # Calculate "perfect" steering
    steering = goal_theta - theta

    # Calculate "perfect" acceleration
    d_vel = goal_vel - vel
    # Equation derived from:
    #   d_goal = 0.5*acc*t² + v_0*t
    #   d_vel = acc*t
    acc = (1./goal_d)*d_vel*(d_vel/2+vel)

    # Normalize labels if the node features were normalized at the beginning
    if normalize:
      # Manually define ranges for steering and acc to norm data
      range_steering = [-0.1, 0.1]
      range_acc = [-0.6, 0.6]
      steering = self._normalize_value(steering, range_steering)
      acc = self._normalize_value(acc, range_acc)
    
    action_labels = OrderedDict()
    action_labels["steering"] = steering
    action_labels["acceleration"] = acc
    
    return action_labels

  def _normalize_value(self, value, range):
    """Normalization of value to range

    If the `value` is outside the given range, it's clamped 
    to the bound of [-1, 1]
    """
    normed = 2 * (value - range[0]) / (range[1] - range[0]) - 1
    normed = max(-1, normed) # values lower -1 clipped
    normed = min(1, normed) # values bigger 1 clipped
    return normed
  
  def _denormalize_value(self, normed, range):
    """Inverse function of a normalization

    Reconstructs the original value from a given normed value and a range.
    The reconstruction is with errors, if the value was clipped during
    normalization. The expected range of the original normalization is [-1,1].
    
    Args:
      normed: Double or float
      range: List containing lower and upper bound

    Returns:
      value: Double or float"""

    value = 0.5*(normed+1)*(range[1]-range[0]) + range[0]
    return value

  def _save_data(self, data):
    """Save the data in a prespecified directory

    Args:
      data: the data that needs to be saved
    """
    if self._dump_dir is not None:
      # Check if path is existent or generate directory
      if not os.path.exists(self._dump_dir):
        os.makedirs(self._dump_dir)
      path = self._dump_dir+ '/dataset_' + str(int(time.time()*10000)) + '.pickle'
      with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
  params = ParameterServer(filename="examples/example_params/tfa_params.json")
  params["World"]["remove_agents_out_of_map"] = False

  graph_generator = DataGenerator(num_scenarios=100, dump_dir=None, params=params)
  graph_generator.run_scenarios()
