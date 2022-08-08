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
import time
import logging

# BARK imports
from bark.runtime.commons.parameters import ParameterServer

from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.model_loader \
  import pytorch_script_wrapper

from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.tests.test_imitation_agent import TestActionWrapper, \
        TestObserver, create_data

from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.agent import ImitationAgent
num_actions = 4
class TestMotionPrimitiveBehavior:
  def __init__(self, num_actions):
    self._num_actions = num_actions

  def GetMotionPrimitives(self):
    return list(range(0,self._num_actions))

class TestDemonstrationCollector:
  def __init__(self):
    self.data = create_data(10000)
    self._observer = TestObserver()
    self._ml_behavior =  TestActionWrapper()
    self.motion_primitive_behavior = TestMotionPrimitiveBehavior(num_actions)

  def GetDemonstrationExperiences(self):
    return self.data

  @property
  def observer(self):
    return self._observer

  @property
  def ml_behavior(self):
    return self._ml_behavior

  def GetDirectory(self):
    return "./save_dir/collections"

def imitation_agent(layer_dims):
    params = ParameterServer()
    params["ML"]["BaseAgent"]["NumSteps"] = 2
    params["ML"]["BaseAgent"]["EvalInterval"] = 1
    params["ML"]["ImitationModel"]["EmbeddingDims"] = layer_dims
    data = create_data(1000)
    demo_train = data[0:1000]
    demo_test = data[51:]
    agent = ImitationAgent(agent_save_dir="./save_dir", demonstration_collector=TestDemonstrationCollector(),
                          params=params)
    agent.run()
    return agent, demo_train, demo_test


class ModelLoaderTests(unittest.TestCase):
  def test_model_loader_imitation(self):
    agent, demo_train, demo_test = imitation_agent([200, 200, 100])
    agent.save(checkpoint_type="last")
    script_file = agent.get_script_filename("last")
    # test all network
    model = pytorch_script_wrapper.ModelLoader()
    model.LoadModel(os.path.abspath(script_file))

    num_iters = 10  # Number of times to repeat experiment to calcualte runtime

    # Time num_iters iterations for inference using C++ model
    start = time.time()
    output_cpp = []
    for idx in range(num_iters):
      output_cpp.append(model.Inference(list(demo_train[idx][0])))
    end = time.time()
    time_cpp = end - start  # todo - how to analyze python vs cpp test time in tests?
    logging.info(f"C++ time: {time_cpp/num_iters}")

    #Time num_iters iterations for inference using python model
    start = time.time()
    output_py = []
    agent.online_net.eval()
    for idx in range(num_iters):
      output_py.append(agent.calculate_actions(demo_train[idx][0]))

    end = time.time()
    time_py = end - start
    logging.info(f"Python time: {time_py/num_iters}")

    np.testing.assert_array_almost_equal(
        np.asarray(output_cpp),
        np.asarray(output_py),
        decimal=6,
        err_msg="C++ and python models don't match")


if __name__ == '__main__':
  unittest.main()
