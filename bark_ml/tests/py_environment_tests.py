# Copyright (c) 2020 fortiss GmbH
#
# Authors: Patrick Hart, Julian Bernhard, Klemens Esterle, and
# Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


import unittest
import numpy as np
import os
import gym
import pprint

from bark.core.models.behavior import BehaviorConstantAcceleration
from bark.runtime.commons.parameters import ParameterServer
from bark_ml.environments.blueprints import ContinuousHighwayBlueprint, \
  DiscreteHighwayBlueprint, ContinuousMergingBlueprint, DiscreteMergingBlueprint, \
  ConfigurableScenarioBlueprint
from bark_ml.environments.single_agent_runtime import SingleAgentRuntime
from bark_ml.environments.counterfactual_runtime import CounterfactualRuntime
from bark_ml.library_wrappers.lib_tf_agents.agents.sac_agent import BehaviorSACAgent
import bark_ml.environments.gym  # pylint: disable=unused-import
from bark_ml.behaviors.cont_behavior import BehaviorContinuousML


class PyEnvironmentTests(unittest.TestCase):
  def test_envs_cont_rl(self):
    params = ParameterServer()
    cont_blueprints = []
    cont_blueprints.append(ContinuousHighwayBlueprint(params))
    cont_blueprints.append(ContinuousMergingBlueprint(params))

    for bp in cont_blueprints:
      env = SingleAgentRuntime(blueprint=bp, render=False)
      env.reset()
      for _ in range(0, 5):
        action = np.random.uniform(low=-0.1, high=0.1, size=(2, ))
        observed_next_state, reward, done, info = env.step(action)
        # print(f"Reward: {reward}, Done: {done}")

  def test_envs_discrete_rl(self):
    params = ParameterServer()
    discrete_blueprints = []
    discrete_blueprints.append(DiscreteHighwayBlueprint(params))
    discrete_blueprints.append(DiscreteMergingBlueprint(params))

    for bp in discrete_blueprints:
      env = SingleAgentRuntime(blueprint=bp, render=False)
      env.reset()
      for _ in range(0, 5):
        action = np.random.randint(low=0, high=3)
        observed_next_state, reward, done, info = env.step(action)
        # print(f"Reward: {reward}, Done: {done}")

  def test_gym_wrapping(self):
    # highway-v0: continuous
    # highway-v1: discrete
    # merging-v0: continuous
    # merging-v1: discrete
    # are registered here: import bark_ml.environments.gym  # pylint: disable=unused-import

    cont_envs = [gym.make("highway-v0"), gym.make("merging-v0")]
    for env in cont_envs:
      env.reset()
      for _ in range(0, 5):
        action = np.random.uniform(low=-0.1, high=0.1, size=(2, ))
        observed_next_state, reward, done, info = env.step(action)
        print(f"Observed state: {observed_next_state}, Reward: {reward}, Done: {done}")

    cont_envs = [gym.make("highway-v1"), gym.make("merging-v1")]
    for env in cont_envs:
      env.reset()
      for _ in range(0, 5):
        action = np.random.randint(low=0, high=3)
        observed_next_state, reward, done, info = env.step(action)
        print(f"Observed state: {observed_next_state}, Reward: {reward}, Done: {done}")

  def test_counterfactual_runtime(self):
    params = ParameterServer()
    bp = ContinuousMergingBlueprint(params)
    # BehaviorConstantAcceleration::ConstAcceleration
    behavior_model_pool = []
    count = 0
    for a in [-3., 0.]:
      local_params = params.AddChild("local_"+str(count))
      local_params["BehaviorConstantAcceleration"]["ConstAcceleration"] = a
      behavior = BehaviorConstantAcceleration(local_params)
      behavior_model_pool.append(behavior)
      count += 1

    env = CounterfactualRuntime(
      blueprint=bp,
      render=True,
      behavior_model_pool=behavior_model_pool,
      params=params)
    sac_agent = BehaviorSACAgent(environment=env,
                                 params=params)
    env.ml_behavior = sac_agent
    env.ml_behavior.set_action_externally = False
    env.reset()
    for _ in range(0, 5):
      action = np.random.uniform(low=-0.1, high=0.1, size=(2, ))
      observed_next_state, reward, done, info = env.step(action)
      print(f"Observed state: {observed_next_state}, Reward: {reward}, Done: {done}")
    pprint.pprint(env.tracer._states)

  def test_configurable_blueprint(self):
    params = ParameterServer(filename="bark_ml/tests/data/highway_merge_configurable.json")
    # continuous model
    ml_behavior = BehaviorContinuousML(params=params)
    bp = ConfigurableScenarioBlueprint(
      params=params,
      ml_behavior=ml_behavior)
    env = SingleAgentRuntime(blueprint=bp, render=False)
    # agent
    sac_agent = BehaviorSACAgent(environment=env,
                                 params=params)
    env.ml_behavior = sac_agent
    # test run
    env.reset()
    for _ in range(0, 5):
      action = np.random.randint(low=0, high=3)
      observed_next_state, reward, done, info = env.step(action)


if __name__ == '__main__':
  unittest.main()