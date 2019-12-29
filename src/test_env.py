# Copyright (c) 2019 Patrick Hart
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


import unittest
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class Environment:
  def __init__(self,
               num_agents=1):
    self._num_agents = num_agents
    self._states = tf.TensorArray(size=21,
                                  dtype=tf.float32,
                                  clear_after_read=False)
    self._i = 0

  def init_states(self, num_agents=1):
    intitial_states = np.random.uniform(size=(num_agents, 4),
                                        low=[-4., 0., 1.45, 9.],
                                        high=[4., 10., 1.55, 11])
    initial_state = tf.convert_to_tensor(intitial_states, dtype=tf.float32)
    self._states.write(0, initial_state)

  def move_agents(self, a, dt=0.1):
    state = self._states.read(self._i)
    next_state = self._states.read(self._i + 1)
    for i in range(0, state.shape[0]):
      next_state = tf.tensor_scatter_nd_update(
        next_state,
        [[i, 0],
         [i, 1],
         [i, 2],
         [i, 3]],
        [state[i, 0] + dt*tf.math.cos(state[i, 2])*state[i, 3],
         state[i, 1] + dt*tf.math.sin(state[i, 2])*state[i, 3],
         state[i, 2] + dt*tf.math.tan(a[i, 0])*state[i, 3],
         state[i, 3] + dt*a[i, 1]])
    self._states.write(self._i + 1, next_state)

  def get_observation(self):
    return self._states.read(self._i)

  def get_reward(self):
    return (self._states.read(self._i)[:, 1] - 0.)**2

  def reset(self):
    self.init_states(self._num_agents)
    self._i = 0
    return self.get_observation()

  def step(self, a, dt=0.1):
    self.move_agents(a, dt)
    self._i += 1
    done = False
    if self._i == 20:
      done = True
    return self.get_observation(), self.get_reward(), done, None

  def render(self):
    pass