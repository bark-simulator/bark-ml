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

#np.random.seed(0)
# a sample random state [0-1] to evaluate actions
random_state = np.random.rand(state_space_size).tolist()

# Do inference using C++ wrapped model
model = pytorch_script_wrapper.ModelLoader(os.path.join(os.path.dirname(__file__), "model_data/online_net_script.pt"), action_space_size, state_space_size)
model.LoadModel()

num_iters = 10000 # Number of times to repeat experiment to calcualte runtime

# Time num_iters iterations for inference using C++ model
start = time.time()
for i in range(num_iters):
  actions_cpp = model.Inference(random_state)
end = time.time()
time_cpp = end-start

# Do inference using python model
agent = IQNAgent(env=env, test_env=env, params = ParameterServer())
agent.load_models(os.path.join(os.path.dirname(__file__), "model_data"))

# Time num_iters iterations for inference using python model
start = time.time()
for i in range(num_iters):
  actions_py = agent.calculate_actions(random_state)

end = time.time()
time_py = end-start

# Calcualte relative error between C++ and python models
relative_error = lambda x,y: np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))
error = relative_error(actions_py.numpy(), np.asarray(actions_cpp))

assert error < 5e-6 , "C++ and python models don't match"

# Print report
print ("\nRun Time comparison\n----------------------------------")
print("Time (cpp): {:.2f} s/{} iters".format(time_py, num_iters))
print("Time (python): {:.2f} s/{} iters".format(time_cpp, num_iters))
print("Relative error: {}".format(error))