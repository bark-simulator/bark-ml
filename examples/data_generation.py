#Copyright (c) 2019 fortiss GmbH
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import os
import networkx as nx
import numpy as np
import matplotlib as mpl
from absl import app
from absl import flags
import time, json, pickle
from abc import ABC

# BARK imports
from bark.runtime.commons.parameters import ParameterServer
from bark.runtime.viewer.matplotlib_viewer import MPViewer
from bark.runtime.viewer.video_renderer import VideoRenderer


# BARK-ML imports
from bark_ml.environments.blueprints import ContinuousHighwayBlueprint, \
  ContinuousMergingBlueprint, ContinuousIntersectionBlueprint
from bark_ml.environments.single_agent_runtime import SingleAgentRuntime
#from bark_ml.library_wrappers.lib_tf_agents.agents import BehaviorSACAgent, BehaviorPPOAgent, BehaviorGraphSACAgent
#from bark_ml.library_wrappers.lib_tf_agents.runners import SACRunner, PPORunner
from bark_ml.observers.graph_observer import GraphObserver
#from bark_ml.library_wrappers.lib_tf2_gnn import GNNActorNetwork, GNNCriticNetwork

class DataGenerator(ABC):
  """Data Generator class"""
  def __init__(self, num_scenarios=3, dump_dir=None):
    self.dump_dir = dump_dir
    self.num_scenarios = num_scenarios

    # Adapt params (start from default_params)
    self.params = ParameterServer(filename="examples/example_params/tfa_params.json")
    self.params["World"]["remove_agents_out_of_map"] = False
    #self.params["World"]["remove_agents_out_of_map"] = True #seems not to work as intended
    self.bp = ContinuousHighwayBlueprint(self.params, number_of_senarios=num_scenarios, random_seed=0)
    #self.bp = ContinuousIntersectionBlueprint(self.params, number_of_senarios=num_scenarios, random_seed=0)
    self.observer = GraphObserver(normalize_observations=True, output_supervised_data=True, params=self.params)
    self.env = SingleAgentRuntime(blueprint=self.bp, observer=self.observer, render=True)


  def run_scenario(self, scenario):
    """Runs a specific scenario from initial for predefined steps
        Inputs: scenario    :   bark-scenario
        
        Output: scenario_data : list containing all individual data_points of run scenario"""

    data_scenario = list()
    self.env.reset(scenario = scenario)
    done = False
    i = 0
    while done is False:
      #for i in range(10):
      action = np.random.uniform(low=np.array([-0.5, -0.02]), high=np.array([0.5, 0.02]), size=(2, ))
      #agent_actions = observed_next_state[1][0] #use predicted values
      #action = np.array([agent_actions["acceleration"], agent_actions["steering"]])
      print(i)
      i+=1
      observed_next_state, reward, done, info = self.env.step(action)
      
      graph =  observed_next_state[0]
      actions = observed_next_state[1]

      # Save datum in data_scenario
      datum = dict()
      datum["graph"] = nx.node_link_data(graph)
      datum["actions"] = actions
      data_scenario.append(datum)
    return data_scenario

  def run_scenarios(self):
    """Run all scenarios"""

    for _ in range(0, self.num_scenarios):
      scenario, idx = self.bp._scenario_generation.get_next_scenario()
      print("Scenario", idx)
      data_scenario = self.run_scenario(scenario)
      time.sleep(1)
      self.save_data(data_scenario)
      #self.data.append(data_scenario)

  def save_data(self, data):
      # Save data
      if self.dump_dir == None:
        print("Data not saved as dump_dir not specified!")
        #raise Exception("specify dump_dir to tell the system where to store the data")
      else:
        if not os.path.exists(self.dump_dir):
          os.makedirs(self.dump_dir)
        path = self.dump_dir+ '/dataset_' + str(int(time.time())) + '.pickle'
        with open(path, 'wb') as handle:
          pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
          print('---> Dumped dataset to: ' + path)


# Main part
if __name__ == '__main__':
  graph_generator = DataGenerator(num_scenarios=100, dump_dir='/home/silvan/working_bark/supervised_learning/data/')

  graph_generator.run_scenarios()