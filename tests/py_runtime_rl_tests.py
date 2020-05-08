# Copyright (c) 2019 fortiss GmbH
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


import unittest
import numpy as np
import os
import matplotlib
#matplotlib.use('PS')
import time
from modules.runtime.scenario.scenario_generation.uniform_vehicle_distribution \
  import UniformVehicleDistribution
from modules.runtime.scenario.scenario_generation.deterministic \
  import DeterministicScenarioGeneration
from modules.runtime.commons.parameters import ParameterServer
from modules.runtime.viewer.matplotlib_viewer import MPViewer
from src.rl_runtime import RuntimeRL
from src.observers.nearest_state_observer import ClosestAgentsObserver
from src.observers.simple_observer import SimpleObserver
from src.wrappers.dynamic_model import DynamicModel
from src.wrappers.motion_primitives import MotionPrimitives
from src.evaluators.goal_reached import GoalReached

class PyRuntimeRLTests(unittest.TestCase):
  @unittest.skip
  def test_runtime_rl(self):
    params = ParameterServer(
      filename="tests/data/deterministic_scenario_test.json")
    scenario_generation = DeterministicScenarioGeneration(num_scenarios=2,
                                                          random_seed=0,
                                                          params=params)
    state_observer = SimpleObserver(params=params)
    action_wrapper = DynamicModel(params=params)
    evaluator = GoalReached(params=params)
    viewer = MPViewer(params=params,
                      x_range=[-30,30],
                      y_range=[-40,40],
                      use_world_bounds=True) #use_world_bounds=True) # 

    runtimerl = RuntimeRL(action_wrapper=action_wrapper,
                          observer=state_observer,
                          evaluator=evaluator,
                          step_time=0.2,
                          viewer=viewer,
                          scenario_generator=scenario_generation,
                          render=False)

    start_time = time.time()
    for _ in range(0, 100):
      runtimerl.reset()
      done = False
      reward = 0.
      for _ in range(0, 50): # run each scenario for 10 steps
        action = action_wrapper.action_space.sample() / 100 # to go straight
        next_observed_state, reward, done, info = \
          runtimerl.step(action)
        # observer
        self.assertEqual(len(next_observed_state), 16)
        np.testing.assert_array_equal(
          next_observed_state[0:4],
          state_observer._normalize(runtimerl._world.agents[100].state)[1:5])
        np.testing.assert_array_equal(
          next_observed_state[4:8],
          state_observer._normalize(runtimerl._world.agents[101].state)[1:5])
        if done:
          print("State: {} \n Reward: {} \n Done {}, Info: {} \n \
              =================================================". \
            format(next_observed_state, reward, done, info))
          break
      # must assert to equal as the agent reaches the goal in the
      # specified number of steps
      self.assertEqual(done, True)
      # goal must have been reached which returns a reward of 1.
      self.assertEqual(reward, 1.)
      self.assertEqual(runtimerl._world.agents[100].id, 100)
      self.assertEqual(runtimerl._world.agents[101].id, 101)
    end_time = time.time()
    print("100 runs took {}s.".format(str(end_time-start_time)))

  def test_motion_primitives_concat_state(self):
    params = ParameterServer(
      filename="tests/data/deterministic_scenario_test.json")
    scenario_generation = DeterministicScenarioGeneration(num_scenarios=3,
                                                          random_seed=0,
                                                          params=params)
    state_observer = SimpleObserver(params=params)
    action_wrapper = MotionPrimitives(params=params)
    evaluator = GoalReached(params=params)
    viewer = MPViewer(params=params,
                      x_range=[-30,30],
                      y_range=[-40,40],
                      use_world_bounds=True)

    runtimerl = RuntimeRL(action_wrapper=action_wrapper,
                          observer=state_observer,
                          evaluator=evaluator,
                          step_time=0.2,
                          viewer=viewer,
                          scenario_generator=scenario_generation,
                          render=False)

    params.save(filename="./default_params_runtime_rl_motion_primitives.json")

    for _ in range(0, 3):
      runtimerl.reset()
      for _ in range(0, 100): # run each scenario for 10 steps
        action = action_wrapper.action_space.sample()
        next_observed_state, reward, done, info = \
          runtimerl.step(action)
        runtimerl.render()
        if done:
          print("State: {} \n Reward: {} \n Done {}, Info: {} \n \
              =================================================". \
            format(next_observed_state, reward, done, info))
          break



if __name__ == '__main__':
  unittest.main()