# Copyright (c) 2020 fortiss GmbH
#
# Authors: Patrick Hart, Julian Bernhard, Klemens Esterle, and
# Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


import unittest
import numpy as np
import time

from bark.runtime.commons.parameters import ParameterServer
from bark_ml.environments.blueprints import ContinuousHighwayBlueprint, \
  ContinuousSingleLaneBlueprint
from bark_ml.environments.single_agent_runtime import SingleAgentRuntime
from bark_ml.evaluators.evaluator_configs import GoalReached
from bark_ml.evaluators.general_evaluator import GeneralEvaluator
from bark_ml.core.evaluators import GoalReachedEvaluator


class PyEvaluatorTests(unittest.TestCase):
  def test_goal_reached_evaluator(self):
    params = ParameterServer()
    bp = ContinuousHighwayBlueprint(params)
    env = SingleAgentRuntime(blueprint=bp, render=True)
    env.reset()
    world = env._world

    eval_id = env._scenario._eval_agent_ids[0]
    observed_world = world.Observe([eval_id])[0]
    evaluator = GoalReached(params)
    action = np.array([0., 0.], dtype=np.float32)
    start_time = time.time()
    print(evaluator.Evaluate(observed_world, action))
    end_time = time.time()
    print(f"It took {end_time-start_time} seconds.")


  def test_goal_reached_cpp_evaluator(self):
    params = ParameterServer()
    bp = ContinuousHighwayBlueprint(params)
    env = SingleAgentRuntime(blueprint=bp, render=True)
    env.reset()
    world = env._world

    eval_id = env._scenario._eval_agent_ids[0]
    observed_world = world.Observe([eval_id])[0]
    evaluator = GoalReachedEvaluator(params)
    action = np.array([0., 0.], dtype=np.float32)
    start_time = time.time()
    print(evaluator.Evaluate(observed_world, action))
    end_time = time.time()
    print(f"The goal reached took {end_time-start_time} seconds.")

  def test_reward_shaping_evaluator(self):
    params = ParameterServer()
    bp = ContinuousHighwayBlueprint(params)
    env = SingleAgentRuntime(blueprint=bp, render=True)
    env.reset()
    world = env._world

    eval_id = env._scenario._eval_agent_ids[0]
    observed_world = world.Observe([eval_id])[0]
    evaluator = GoalReachedEvaluator(params)
    action = np.array([0., 0.], dtype=np.float32)
    start_time = time.time()
    print(evaluator.Evaluate(observed_world, action))
    end_time = time.time()
    print(f"The reward shaping evaluator took {end_time-start_time} seconds.")

  def test_general_evaluator(self):
    params = ParameterServer()
    bp = ContinuousSingleLaneBlueprint(params)
    env = SingleAgentRuntime(blueprint=bp, render=True)
    evaluator = GeneralEvaluator(params)
    env._evaluator = evaluator
    env.reset()
    for _ in range(0, 4):
      state, terminal, reward, info = env.step(np.array([0., 0.]))
      print(terminal, reward)

if __name__ == '__main__':
  unittest.main()