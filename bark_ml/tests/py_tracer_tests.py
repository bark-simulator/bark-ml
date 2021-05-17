# Copyright (c) 2020 fortiss GmbH
#
# Authors: Patrick Hart
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


import unittest
import numpy as np


# BARK
from bark.runtime.commons.parameters import ParameterServer

# BARK-ML
from bark_ml.library_wrappers.lib_tf_agents.agents.sac_agent import BehaviorSACAgent
from bark_ml.environments.blueprints import ContinuousHighwayBlueprint
from bark_ml.environments.single_agent_runtime import SingleAgentRuntime
from bark_ml.commons.tracer import Tracer


class PyTracerTests(unittest.TestCase):
  """Tracer tests."""

  def test_tracer(self):
    params = ParameterServer()
    bp = ContinuousHighwayBlueprint(params)
    tracer = Tracer()
    env = SingleAgentRuntime(blueprint=bp, render=False)
    for i in range(0, 2):
      env.reset()
      for _ in range(0, 10):
        action = np.random.uniform(low=-0.1, high=0.1, size=(2, ))
        data = (observed_next_state, reward, done, info) = env.step(action)
        tracer.Trace(data, num_episode=i)

    # NOTE: test basic tracing
    self.assertEqual(len(tracer._states), 20)
    for i in range(0, 20):
      self.assertEqual("is_terminal" in tracer._states[i].keys(), True)
      self.assertEqual("reward" in tracer._states[i].keys(), True)
      self.assertEqual("collision" in tracer._states[i].keys(), True)
      self.assertEqual("drivable_area" in tracer._states[i].keys(), True)
      self.assertEqual("goal_reached" in tracer._states[i].keys(), True)
      self.assertEqual("step_count" in tracer._states[i].keys(), True)

    # NOTE: test reset
    tracer.Reset()
    self.assertEqual(len(tracer._states), 0)

  def test_trace_dict(self):
    """Make sure tracing of dictionaries works as well."""
    tracer = Tracer()
    for j in range(0, 5):
      for i in range(0, 10):
        eval_dict = {"step_count": i}
        tracer.Trace(eval_dict, num_episode=j)
    self.assertEqual(len(tracer._states), 50)
    for i in range(0, 50):
      self.assertEqual("step_count" in tracer._states[i].keys(), True)
      self.assertEqual("num_episode" in tracer._states[i].keys(), True)

  def test_tracing_bark_world(self):
    params = ParameterServer()
    bp = ContinuousHighwayBlueprint(params)
    tracer = Tracer()
    env = SingleAgentRuntime(blueprint=bp, render=False)
    sac_agent = BehaviorSACAgent(
      environment=env,
      params=params)
    env.ml_behavior = sac_agent
    # NOTE: this also tests if a BARK agent is self-contained
    env.ml_behavior.set_actions_externally = False
    env.reset()
    bark_world = env._world
    for j in range(0, 2):
      for _ in range(0, 5):
        bark_world.Step(0.2)
        eval_dict = bark_world.Evaluate()
        eval_dict["is_terminal"] = eval_dict["collision"] or \
          eval_dict["goal_reached"] or eval_dict["drivable_area"]
        eval_dict["reward"] = 0
        tracer.Trace(eval_dict, num_episode=j)
    self.assertEqual(len(tracer._states), 10)
    # print(tracer._states)


if __name__ == '__main__':
  unittest.main()