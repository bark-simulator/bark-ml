# Copyright (c) 2020 Patrick Hart, Julian Bernhard,
# Klemens Esterle, Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import gym
import numpy as np
import bark_ml.environments.gym

# cont. highway env
# env = gym.make("highway-v0")
env = gym.make("merging-v0")
# env = gym.make("intersection-v0")


env.reset()
done = False
while done is False:
  action = np.random.uniform(low=np.array([-0.5, -0.1]), high=np.array([0.5, 0.1]), size=(2, ))
  observed_next_state, reward, done, info = env.step(action)
  print(f"Observed state: {observed_next_state}, Action: {action}, Reward: {reward}, Done: {done}")
