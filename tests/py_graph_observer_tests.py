# Copyright (c) 2019 fortiss GmbH
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import os, time, json, pickle
import numpy as np
import unittest
from tf_agents.environments import tf_py_environment

from src.observers.graph_observer import GraphObserver
from modules.runtime.commons.parameters import ParameterServer
from modules.runtime.scenario.scenario_generation.deterministic \
  import DeterministicScenarioGeneration
from src.wrappers.dynamic_model import DynamicModel
from src.evaluators.goal_reached import GoalReached
from modules.runtime.viewer.matplotlib_viewer import MPViewer
from src.rl_runtime import RuntimeRL

class PyGraphObserverTests(unittest.TestCase):

    def test_observer(self):
      #params = ParameterServer(filename="tests/data/deterministic_scenario_test.json")
      params = ParameterServer(filename="tests/data/graph_scenario_test.json")
      base_dir = os.path.dirname(os.path.dirname(__file__))
      params["BaseDir"] = base_dir
      
      scenario_generation = DeterministicScenarioGeneration(num_scenarios=3,
                                                            random_seed=0,
                                                            params=params)      
      observer = GraphObserver(params)
      behavior_model = DynamicModel(params=params)
      evaluator = GoalReached(params=params)
      viewer = MPViewer(params=params, use_world_bounds=True) # follow_agent_id=True)
    
      runtime = RuntimeRL(action_wrapper=behavior_model,
                          observer=observer,
                          evaluator=evaluator,
                          step_time=0.1,
                          viewer=viewer,
                          scenario_generator=scenario_generation)

      scenario = scenario_generation.create_single_scenario()
      graph, actions = runtime.reset(scenario)

      print('\n ------------ Nodes ------------')
      for node in graph.nx_graph.nodes.data():
        print(node)

      print('\n ------------ Edges ------------')
      for node in graph.nx_graph.edges.data():
        print(node)

      print('\n ----------- Actions -----------')
      for item in actions.items():
        print(item)

      # Visualize Movement of vehicles
      data_collector = list()
      steer_bias = -0.1
      acc_bias = -0.8
      for i in range(100):
        #observed_world = runtime.step([-0.8,0.0,0.0,0.0]) # for 2 ego vehicles
        # Generate random steer and acc commands
        if i % 400 == 0:
          steer_bias -= 0.1
          acc_bias += 0
        elif i% 200 == 0:
          steer_bias += 0.1
          acc_bias += 0
        steer = np.random.random()*0.2 + steer_bias
        acc = np.random.random()*2.0 + acc_bias
        # Run step
        returns = runtime.step([acc, steer]) # [acc, steer] with -1<acc<1 and -0.1<steer<0.1
        # Save datum in da
        datum = dict()
        datum["graph_data"] = returns[0][0]
        datum["graph_labels"] = returns[0][1]
        data_collector.append(datum)
        runtime.render()
        time.sleep(0.01)
      # Save data after run
      with open('/home/silvan/working_bark/tests/data/data_collection_agents7.pickle', 'wb') as handle:
        pickle.dump(data_collector, handle, protocol=pickle.HIGHEST_PROTOCOL)
if __name__ == '__main__':
  unittest.main()