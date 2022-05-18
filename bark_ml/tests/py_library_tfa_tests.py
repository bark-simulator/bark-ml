# Copyright (c) 2020 fortiss GmbH
#
# Authors: Patrick Hart
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import unittest
import numpy as np
import gym

# BARK imports
from bark.runtime.commons.parameters import ParameterServer

# BARK-ML imports
from bark_ml.environments.blueprints import ContinuousHighwayBlueprint
from bark_ml.environments.single_agent_runtime import SingleAgentRuntime
import bark_ml.environments.gym  # pylint: disable=unused-import
from bark_ml.library_wrappers.lib_tf_agents.agents.ppo_agent import BehaviorPPOAgent
from bark_ml.library_wrappers.lib_tf_agents.agents.sac_agent import BehaviorSACAgent
from bark_ml.library_wrappers.lib_tf_agents.runners.ppo_runner import PPORunner
from bark_ml.library_wrappers.lib_tf_agents.runners.sac_runner import SACRunner


class PyLibraryWrappersTFAgentTests(unittest.TestCase):
  """TFAgentTests tests."""

  # make sure the agent works
  def test_agent_wrapping(self):
    params = ParameterServer()
    env = gym.make("highway-v0")
    env.reset()
    agent = BehaviorPPOAgent(environment=env,
                     params=params)
    agent = BehaviorSACAgent(environment=env,
                     params=params)

  # assign as behavior model (to check if trained agent can be used)
  def test_behavior_wrapping(self):
    # create scenario
    params = ParameterServer()
    bp = ContinuousHighwayBlueprint(params,
                                    num_scenarios=10,
                                    random_seed=0)
    env = SingleAgentRuntime(blueprint=bp,
                             render=False)
    ml_behaviors = []
    ml_behaviors.append(
      BehaviorPPOAgent(environment=env,
                       params=params))
    ml_behaviors.append(
      BehaviorSACAgent(environment=env,
                       params=params))

    for ml_behavior in ml_behaviors:
      # set agent
      env.ml_behavior = ml_behavior
      env.reset()
      done = False
      while done is False:
        action = np.random.uniform(low=-0.1, high=0.1, size=(2, ))
        observed_next_state, reward, done, info = env.step(action)
        print(f"Observed state: {observed_next_state}, Reward: {reward}, Done: {done}")

      # action is set externally
      ml_behavior._set_action_externally = True
      agent_id = list(env._world.agents.keys())[0]
      observed_world = env._world.Observe([agent_id])[0]
      action = np.random.uniform(low=-0.1, high=0.1, size=(2, ))
      ml_behavior.ActionToBehavior(action)
      a = ml_behavior.Plan(0.2, observed_world)
      action = np.random.uniform(low=-0.1, high=0.1, size=(2, ))
      ml_behavior.ActionToBehavior(action)
      b = ml_behavior.Plan(0.2, observed_world)
      self.assertEqual(np.any(np.not_equal(a, b)), True)

      # action will be calculated within the Plan(..) fct.
      a = ml_behavior.Plan(0.2, observed_world)
      b = ml_behavior.Plan(0.2, observed_world)
      np.testing.assert_array_equal(a, b)


  # agent + runner
  def test_agent_and_runner(self):
    params = ParameterServer()
    bp = ContinuousHighwayBlueprint(params,
                                    num_scenarios=10,
                                    random_seed=0)
    env = SingleAgentRuntime(blueprint=bp, render=False)
    agent = BehaviorPPOAgent(environment=env, params=params)

    # set agent
    params["ML"]["PPORunner"]["NumberOfCollections"] = 2
    params["ML"]["SACRunner"]["NumberOfCollections"] = 2
    params["ML"]["TFARunner"]["EvaluationSteps"] = 2
    env.ml_behavior = agent
    self.assertEqual(env.ml_behavior.set_action_externally, False)
    runner = PPORunner(params=params,
                       environment=env,
                       agent=agent)
    runner.Train()
    self.assertEqual(env.ml_behavior.set_action_externally, True)
    runner.Run()
    self.assertEqual(env.ml_behavior.set_action_externally, True)



if __name__ == '__main__':
  unittest.main()