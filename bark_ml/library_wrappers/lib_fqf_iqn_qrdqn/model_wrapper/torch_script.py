#include "examples/pytorch_script_wrapper.hpp"
import numpy as np
import gym
import os
import time
import bark_ml.environments.gym
from bark.runtime.commons.parameters import ParameterServer
from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.agent import IQNAgent
from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.model_wrapper import pytorch_script_wrapper


env = "highway-v1"
env = gym.make(env)
action_space_size = env.action_space.n
state_space_size = env.observation_space.shape[0]


# a sample random state [0-1] to evaluate actions
random_state = np.random.rand(16).tolist()

# Do inference using C++ wrapped model
model = pytorch_script_wrapper.ModelLoader(os.path.join(os.path.dirname(__file__), "model_data/online_net_script.pt"), action_space_size, state_space_size)
model.LoadModel()

# Time 10000 iterations for inference in cpp
start = time.time()
for i in range(10000):
  actions_cpp = model.Inference(random_state)
end = time.time()
time_cpp = end-start

# Do inference using python model
agent = IQNAgent(env=env, test_env=env, params = ParameterServer())
agent.load_models(os.path.join(os.path.dirname(__file__), "model_data"))

# Time 10000 iterations for inference using python model
start = time.time()
for i in range(10000):
  actions_py = agent.calculate_actions(random_state)
end = time.time()
time_py = end-start

# Print report
print ("----------------------- Time comparison ------------------------")
print("Time(python):{}".format(time_cpp))
print("Time(cpp):{}".format(time_py))
print ("----------------------------------------------------------------\n")

print ("----------------- Action comparison ----------------------------")
print("Time(python):{}".format(actions_py.argmax()))
print("Time(cpp):{}".format(np.asarray(actions_cpp).argmax().item()))
print ("----------------------------------------------------------------\n")

print ("----------------- Action reward comparison -----------------------")
print ("Actions(python model):\n{}\n".format(actions_py.tolist()))
print ("Actions(cpp model):\n{}\n".format(actions_cpp))
print ("------------------------------------------------------------------")

