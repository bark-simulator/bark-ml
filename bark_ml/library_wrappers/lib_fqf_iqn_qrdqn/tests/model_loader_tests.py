# Copyright (c) 2020 fortiss GmbH
#
# Authors: Patrick Hart
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

try:
    import debug_settings
except:
    pass

import unittest
import numpy as np
import os
import gym
import matplotlib
import time

# BARK imports
from bark.runtime.commons.parameters import ParameterServer

# BARK-ML imports
from bark_ml.environments.blueprints import \
  DiscreteHighwayBlueprint, DiscreteMergingBlueprint
from bark_ml.environments.single_agent_runtime import SingleAgentRuntime
import bark_ml.environments.gym
from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.model_wrapper \
 import pytorch_script_wrapper

class ModelLoaderTests(unittest.TestCase):
  def test_model_loader(self):
    # env using default params
    env = gym.make("highway-v1")

    networks = ["iqn", "fqf", "qrdqn"]

    action_space_size = env.action_space.n
    state_space_size = env.observation_space.shape[0]

    # a sample random state [0-1] to evaluate actions
    random_state = np.random.rand(state_space_size).tolist()

    # test all networks
    for network in networks:
      # Do inference using C++ wrapped model
      model = pytorch_script_wrapper.ModelLoader(
          os.path.join(
              os.path.dirname(__file__),
              "lib_fqf_iqn_qrdqn_test_data/{}/online_net_script.pt"
              .format(network)), action_space_size, state_space_size)
      model.LoadModel()

      num_iters = 1000  # Number of times to repeat experiment to calcualte runtime

      # Time num_iters iterations for inference using C++ model
      start = time.time()
      for _ in range(num_iters):
        actions_cpp = model.Inference(random_state)
      end = time.time()
      time_cpp = end - start  # todo - how to analyze python vs cpp test time in tests?

      # Load and perform inference using python model
      if network == "iqn":
        agent = IQNAgent(env=env, test_env=env, params=ParameterServer())

      elif network == "fqf":
        agent = FQFAgent(env=env, test_env=env, params=ParameterServer())

      elif network == "qrdqn":
        agent = QRDQNAgent(env=env, test_env=env, params=ParameterServer())

      agent.load_models(
          os.path.join(
              os.path.dirname(__file__),
              "lib_fqf_iqn_qrdqn_test_data",
              network))

      # Time num_iters iterations for inference using python model
      start = time.time()
      for _ in range(num_iters):
        actions_py = agent.calculate_actions(random_state)

      end = time.time()
      time_py = end - start

      # assert that Python and Cpp models are close enough to 6 decimal places
      np.testing.assert_array_almost_equal(
          actions_py.flatten().numpy(),
          np.asarray(actions_cpp),
          decimal=6,
          err_msg="C++ and python models don't match")


if __name__ == '__main__':
  unittest.main()
