# Copyright (c) 2020 fortiss GmbH
#
# Authors: Patrick Hart
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


import unittest
import numpy as np
import gym


# BARK
from bark.runtime.commons.parameters import ParameterServer

# BARK-ML
from bark_ml.library_wrappers.lib_tf_agents.agents.sac_agent import BehaviorSACAgent
from bark_ml.library_wrappers.lib_tf_agents.agents.ppo_agent import BehaviorPPOAgent
from bark_ml.environments.blueprints import ContinuousMergingBlueprint
from bark_ml.environments.single_agent_runtime import SingleAgentRuntime
from bark_ml.library_wrappers.lib_tf_agents.agents import BehaviorGraphSACAgent
from bark_ml.observers.graph_observer import GraphObserver
import bark_ml.environments.gym  # pylint: disable=unused-import


class PyBarkBehaviorModelTests(unittest.TestCase):
  def test_sac_agent(self):
    params = ParameterServer()
    env = gym.make("highway-v0")
    sac_agent = BehaviorSACAgent(environment=env, params=params)
    ppo_agent = BehaviorPPOAgent(environment=env, params=params)
    behaviors = [ppo_agent, sac_agent]
    for ml_agent in behaviors:
      env.ml_behavior = ml_agent
      env.reset()
      eval_id = env._scenario._eval_agent_ids[0]
      cloned_world = env._world.Copy()
      self.assertEqual(env._world.agents[eval_id].behavior_model, ml_agent)
      for _ in range(0, 5):
        env._world.Step(0.2)
      self.assertEqual(cloned_world.agents[eval_id].behavior_model, ml_agent)
      for _ in range(0, 5):
        cloned_world.Step(0.2)
      for cloned_agent, agent in zip(env._world.agents.values(),
                                     cloned_world.agents.values()):
        # NOTE: should be the same as mean is taken from the agents
        np.testing.assert_array_equal(cloned_agent.state, agent.state)

  def test_sac_graph_agent(self):
    params = ParameterServer()
    bp = ContinuousMergingBlueprint(params,
                                    num_scenarios=2500,
                                    random_seed=0)
    observer = GraphObserver(params=params)
    env = SingleAgentRuntime(
      blueprint=bp,
      observer=observer,
      render=False)
    sac_agent = BehaviorGraphSACAgent(environment=env,
                                      observer=observer,
                                      params=params)
    env.ml_behavior = sac_agent
    env.reset()
    eval_id = env._scenario._eval_agent_ids[0]
    self.assertEqual(env._world.agents[eval_id].behavior_model, sac_agent)
    for _ in range(0, 5):
      env._world.Step(0.2)


if __name__ == '__main__':
  unittest.main()