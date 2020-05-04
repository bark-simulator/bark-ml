# Copyright (c) 2019 fortiss GmbH
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import os, time
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
                          step_time=0.2,
                          viewer=viewer,
                          scenario_generator=scenario_generation)

      scenario = scenario_generation.create_single_scenario()
      graph = runtime.reset(scenario)

      # Visualize Movement of vehicles
      for i in range(20):
        #observed_world = runtime.step([-0.8,0.0,0.0,0.0]) # for 2 ego vehicles
        graph = runtime.step([0.0,0.0]) # [acc, steer] with -1<acc<1 and -0.1<steer<0.1
        runtime.render()
        #time.sleep(0.01)

if __name__ == '__main__':
  unittest.main()