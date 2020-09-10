# Copyright (c) 2020 fortiss GmbH
#
# Authors: Patrick Hart, Julian Bernhard, Klemens Esterle,
# Tobias Kessler and Mansoor Nasir
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import numpy as np
import gym
import os
import time
from absl import app
from absl import flags

# BARK imports
from bark.runtime.commons.parameters import ParameterServer

# BARK-ML imports
import bark_ml.environments.gym
from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.agent \
 import IQNAgent, FQFAgent, QRDQNAgent
from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.model_wrapper \
 import pytorch_script_wrapper

FLAGS = flags.FLAGS

flags.DEFINE_enum("env", "highway-v1",
                  ["highway-v1", "merging-v1", "intersection-v1"],
                  "Environment the agent should interact in.")

flags.DEFINE_integer("iters",
                     1000,
                     "No of iterations to run the inference.",
                     lower_bound=0)

flags.DEFINE_enum("network", "iqn", ["iqn", "fqf", "qrdqn"],
                  "Environment the agent should interact in.")


def run(argv):
  # env using default params
  env = gym.make(FLAGS.env)

  action_space_size = env.action_space.n
  state_space_size = env.observation_space.shape[0]

  # a sample random state [0-1] to evaluate actions
  random_state = np.random.rand(state_space_size).tolist()

  # Do inference using C++ wrapped model
  model = pytorch_script_wrapper.ModelLoader(
      os.path.join(os.path.dirname(__file__),
                   "model_data/{}/online_net_script.pt".format(FLAGS.network)),
      action_space_size, state_space_size)
  model.LoadModel()

  num_iters = FLAGS.iters  # Number of times to repeat experiment to calcualte runtime

  # Time num_iters iterations for inference using C++ model
  start = time.time()
  for _ in range(num_iters):
    actions_cpp = model.Inference(random_state)
  end = time.time()
  time_cpp = end - start

  # Load and perform inference using python model
  if FLAGS.network == "iqn":
    agent = IQNAgent(env=env, test_env=env, params=ParameterServer())

  elif FLAGS.network == "fqf":
    agent = FQFAgent(env=env, test_env=env, params=ParameterServer())

  elif FLAGS.network == "qrdqn":
    agent = QRDQNAgent(env=env, test_env=env, params=ParameterServer())

  agent.load_models(
      os.path.join(os.path.dirname(__file__), "model_data", FLAGS.network))

  # Time num_iters iterations for inference using python model
  start = time.time()
  for _ in range(num_iters):
    actions_py = agent.calculate_actions(random_state)

  end = time.time()
  time_py = end - start

  # Calcualte relative error between C++ and python models
  relative_error = lambda x, y: np.max(
      np.abs(x - y) / (np.maximum(1e-8,
                                  np.abs(x) + np.abs(y))))
  error = relative_error(actions_py.numpy(), np.asarray(actions_cpp))

  assert error < 1e-5, "C++ and python models don't match"

  # Print report
  print("\nRun Time comparison\n----------------------------------")
  print("Time (cpp): {:.2f} s/{} iters".format(time_py, num_iters))
  print("Time (python): {:.2f} s/{} iters".format(time_cpp, num_iters))
  print("Relative error: {}".format(error))


if __name__ == '__main__':
  app.run(run)
