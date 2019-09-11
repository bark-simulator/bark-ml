# Copyright (c) 2019 fortiss GmbH
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


import unittest
import numpy as np
import os

from modules.runtime.scenario.scenario_generation.uniform_vehicle_distribution \
  import UniformVehicleDistribution
from modules.runtime.scenario.scenario_generation.deterministic \
  import DeterministicScenarioGeneration
from modules.runtime.commons.parameters import ParameterServer
from modules.runtime.viewer.matplotlib_viewer import MPViewer
from src.rl_runtime import RuntimeRL
from src.observers.nearest_state_observer import StateConcatenation
from src.observers.simple_observer import SimpleObserver
from src.wrappers.dynamic_model import DynamicModel
from src.wrappers.motion_primitives import MotionPrimitives
from src.evaluators.goal_reached import GoalReached

class RuntimeRLTests(unittest.TestCase):
  def test_dynamic_behavior_model(self):
    params = ParameterServer(
      filename="data/deterministic_scenario.json")
    scenario_generation = DeterministicScenarioGeneration(num_scenarios=3,
                                                          random_seed=0,
                                                          params=params)
    state_observer = SimpleObserver(params=params)
    action_wrapper = DynamicModel(params=params)
    evaluator = GoalReached(params=params)
    viewer = MPViewer(params=params,
                      x_range=[-30,30],
                      y_range=[-20,40],
                      follow_agent_id=True) #use_world_bounds=True) # 

    runtimerl = RuntimeRL(action_wrapper=action_wrapper,
                          observer=state_observer,
                          evaluator=evaluator,
                          step_time=0.2,
                          viewer=viewer,
                          scenario_generator=scenario_generation)
    runtimerl.reset()
    #for key, agent in runtimerl._world.agents.items():
    #  print(key, agent.id)

    for _ in range(0, 25): # run each scenario for 10 steps
      action = np.array([0,0]) #action_wrapper.action_space.sample()
      next_observed_state, reward, done, info = runtimerl.step(action)
      print("State: {} \n Reward: {} \n Done {}, Info: {} \n \
          =================================================". \
        format(next_observed_state, reward, done, info))
      #if info["success"] or done:
      #  break

  @unittest.skip("...")
  def test_motion_primitives_concat_state(self):
    params = ParameterServer(filename="data/highway_merging.json")
    scenario_generation = UniformVehicleDistribution(num_scenarios=2,
                                                     random_seed=0,
                                                     params=params)
    state_observer = StateConcatenation(params=params)
    action_wrapper = MotionPrimitives(params=params)
    evaluator = GoalReached(params=params)
    viewer = MPViewer(params=params,
                      x_range=[-30,30],
                      y_range=[-20,40],
                      follow_agent_id=True) #use_world_bounds=True) # 

    runtimerl = RuntimeRL(action_wrapper=action_wrapper,
                          observer=state_observer,
                          evaluator=evaluator,
                          step_time=0.2,
                          viewer=viewer,
                          scenario_generator=scenario_generation)


    for _ in range(0, 2): # run 5 scenarios in a row, repeating after 3
      observed_state = runtimerl.reset()
      for _ in range(0, 25): # run each scenario for 10 steps
        next_observed_state, reward, done, info = \
          runtimerl.step(action_wrapper.action_space.sample())
        #runtimerl.render()
        if info["success"] or done:
          print("State: {} \n Reward: {} \n Done {}, Info: {} \n \
              =================================================". \
            format( next_observed_state, reward, done, info))
          break


if __name__ == '__main__':
  unittest.main()