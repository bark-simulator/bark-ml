# Copyright (c) 2020 fortiss GmbH
#
# Authors: Patrick Hart
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import numpy as np
import queue

from bark_ml.environments.single_agent_runtime import SingleAgentRuntime


class SingleAgentDelayRuntime(SingleAgentRuntime):
  """An environment that executes actions with a pre-defined
     delay.
  """
  def __init__(self,
               blueprint=None,
               ml_behavior=None,
               observer=None,
               evaluator=None,
               step_time=None,
               viewer=None,
               scenario_generator=None,
               render=False,
               default_action=None,
               num_delay_steps=None):
    super().__init__(blueprint=blueprint,
                     ml_behavior=ml_behavior,
                     observer=observer,
                     evaluator=evaluator,
                     step_time=step_time,
                     viewer=viewer,
                     scenario_generator=scenario_generator,
                     render=render)
    self._default_action = default_action or [0., 0.]
    self._num_delay_steps = num_delay_steps or 5
    self._action_queue = queue.Queue()

  def step(self, action):
    self._action_queue.put(action)
    action_to_execute = self._action_queue.get()
    return super().step(action_to_execute)

  def reset(self, scenario=None):
    while not self._action_queue.empty():
      self._action_queue.get()
    for _ in range(0, self._num_delay_steps):
      self._action_queue.put(np.array(self._default_action))
    return super().reset(scenario)


class SingleAgentGaussianNoiseRuntime(SingleAgentRuntime):
  """An environment that executes actions with noise.
  """
  def __init__(self,
               blueprint=None,
               ml_behavior=None,
               observer=None,
               evaluator=None,
               step_time=None,
               viewer=None,
               scenario_generator=None,
               render=False,
               sigmas=None):
    super().__init__(blueprint=blueprint,
                     ml_behavior=ml_behavior,
                     observer=observer,
                     evaluator=evaluator,
                     step_time=step_time,
                     viewer=viewer,
                     scenario_generator=scenario_generator,
                     render=render)
    self._sigmas = sigmas or [0., 0.001]

  def step(self, action):
    action += np.random.multivariate_normal(
      mean=list(action), cov=np.diag(self._sigmas))
    # TODO: use params
    action = np.clip(action, [-4, -0.1], [4, 0.1])
    return super().step(action)


class SingleAgentDelayAndGaussianNoiseRuntime(SingleAgentRuntime):
  """An environment that executes actions with a pre-defined
     delay.
  """
  def __init__(self,
               blueprint=None,
               ml_behavior=None,
               observer=None,
               evaluator=None,
               step_time=None,
               viewer=None,
               scenario_generator=None,
               render=False,
               default_action=None,
               num_delay_steps=None,
               sigmas=None):
    super().__init__(blueprint=blueprint,
                     ml_behavior=ml_behavior,
                     observer=observer,
                     evaluator=evaluator,
                     step_time=step_time,
                     viewer=viewer,
                     scenario_generator=scenario_generator,
                     render=render)
    self._default_action = default_action or [0., 0.]
    self._num_delay_steps = num_delay_steps or 5
    self._sigmas = sigmas or [0., 0.001]
    self._action_queue = queue.Queue()

  def step(self, action):
    self._action_queue.put(action)
    action_to_execute = self._action_queue.get()
    action_to_execute += np.random.multivariate_normal(
      mean=list(action_to_execute), cov=np.diag(self._sigmas))
    # TODO: use params
    action_to_execute = np.clip(action_to_execute, [-4, -0.1], [4, 0.1])
    return super().step(action_to_execute)

  def reset(self, scenario=None):
    while not self._action_queue.empty():
      self._action_queue.get()
    for _ in range(0, self._num_delay_steps):
      self._action_queue.put(np.array(self._default_action))
    return super().reset(scenario)
