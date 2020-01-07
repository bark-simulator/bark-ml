# Copyright (c) 2019 Patrick Hart
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


import unittest
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from gym import spaces

class TestEnvironment:
  def __init__(self,
               num_agents=1,
               episode_len=20,
               dt=0.1,
               min_start_d=1.5,
               col_d=1.):
    self._num_agents = num_agents
    self._states = tf.TensorArray(size=episode_len+1,
                                  dtype=tf.float32,
                                  clear_after_read=False)
    self._i = 0
    self._dt = dt
    self._episode_len = episode_len
    self._done = False
    self._min_start_d = min_start_d
    self._col_d = col_d

  def get_state(self, low, high):
    return np.random.uniform(size=(4), low=low, high=high)
  
  def distance(self, state, states):
    dx = state[0] - states[:, 0]
    dy = state[1] - states[:, 1]
    return tf.math.sqrt(dx**2 + dy**2)
  
  def state_colliding(self, state, states, col_dist=1., min_cols=0):
    d = self.distance(state, states)
    result = tf.math.count_nonzero(tf.math.less_equal(d, col_dist))
    if result > min_cols:
      return True
    return False
  
  def init_states(self, num_agents=1):
    low = [-4., 0., 1.45, 10.]
    high = [4., 2., 1.55, 10.]
    states = np.array([self.get_state(low, high)])
    agent_count = 0
    while agent_count < self._num_agents - 1:
      proposed_state = self.get_state(low, high)
      is_colliding = self.state_colliding(
        proposed_state, states, self._min_start_d)
      if is_colliding == False:
        states = np.vstack((states, proposed_state))
        agent_count += 1
    initial_state = tf.convert_to_tensor(states, dtype=tf.float32)
    self._states.write(0, initial_state)

  def move_agents(self, a, dt=0.1):
    state = self._states.read(self._i)
    next_state = self._states.read(self._i + 1)
    a = tf.reshape(a, [-1, 2])
    # tf.print(tf.shape(state), tf.shape(a), tf.shape(dt))
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
    return tf.reshape(self._states.read(self._i), [-1])

  def get_info(self, a=None):
    # to test if it works
    states = self._states.read(self._i)
    collision_count = 0
    reward = 0.
    norm = 0.
    reward += tf.reduce_sum((states[:, 0] - 0.)**2)
    norm += 1.
    reward += 0.1*tf.reduce_sum((states[:, 3] - 10.)**2)
    norm += 0.1

    # TODO(@hart): include cols in reward
    for state in states:
      if self.state_colliding(state, states, col_dist=1., min_cols=1):
        reward += 1000
        collision_count += 1

    if a is not None:
      reward += 0.1*tf.reduce_sum(a[:, 0]**2)
      norm += 0.1
  
    collision = False
    if collision_count > 0:
      collision = True
  
    return -reward/norm, collision

  def reset(self):
    self._states = tf.TensorArray(size=self._episode_len+1,
                                  dtype=tf.float32,
                                  clear_after_read=False)
    self.init_states(self._num_agents)
    self._i = 0
    return self.get_observation()

  def step(self, a):
    self.move_agents(a, self._dt)
    self._i += 1
    self._done = False
    if self._i == self._episode_len:
      self._done = True
    reward, collision = self.get_info()
    if collision:
      self._done = True
    return self.get_observation(), reward, self._done, None

  def render(self):
    states = self._states.stack()
    if self._i == self._episode_len or self._done:
      plt.plot(states[:self._i, :, 0], states[:self._i, :, 1])
      plt.axis("equal")
      plt.show()

  @property
  def action_space(self):
    """Action space of the agent
    """
    low_actions = [[-0.15, -2.0] for _ in range(self._num_agents)]
    high_actions = [[0.15, 2.0] for _ in range(self._num_agents)]
    return spaces.Box(
      low=np.array(low_actions),
      high=np.array(high_actions))

  @property
  def observation_space(self):
    """Observation space of the agent
    """
    obs = [[20., 80., 3.14, 20.] for _ in range(self._num_agents)]
    obs = np.array(obs, dtype=np.float32)
    obs = np.reshape(obs, (-1))
    return spaces.Box(
      low=-obs,
      high=obs)