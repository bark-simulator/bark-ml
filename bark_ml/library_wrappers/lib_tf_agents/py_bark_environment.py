# Copyright (c) 2020 fortiss GmbH
#
# Authors: Patrick Hart
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import numpy as np
import tensorflow as tf

# tfa specific
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

tf.compat.v1.enable_v2_behavior()


class PyBARKEnvironment(py_environment.PyEnvironment):
  """Wrapper for TensorFlow Agents (https://github.com/tensorflow/agents)

  Arguments:
    py_environment -- Base class for environment from tf_agents
  """

  def __init__(self, env):
    self._env = env
    self._action_spec = array_spec.BoundedArraySpec(
      shape=self._env.action_space.shape,
      dtype=np.float32,
      minimum=self._env.action_space.low,
      maximum=self._env.action_space.high,
      name='action')
    self._observation_spec = array_spec.BoundedArraySpec(
      shape=self._env.observation_space.shape,
      dtype=np.float32,
      minimum=self._env.observation_space.low,
      maximum=self._env.observation_space.high,
      name='observation')
    self._state = np.zeros(shape=self._env.observation_space.shape,
      dtype=np.float32)
    self._episode_ended = False

  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec

  def render(self):
    """Renders the enviornment."""
    return self._env.render()

  def _reset(self):
    """Resets the wrapper."""
    self._state = np.array(self._env.reset(), dtype=np.float32)
    self._episode_ended = False
    return ts.restart(self._state)

  def _step(self, action):
    """Steps the world for a given dt."""
    if self._episode_ended:
      return self.reset()
    state, reward, self._episode_ended, _ = self._env.step(action)
    self._state = np.array(state, dtype=np.float32)
    if self._episode_ended:
      return ts.termination(self._state, reward=reward)
    else:
      return ts.transition(self._state, reward=reward, discount=0.99)