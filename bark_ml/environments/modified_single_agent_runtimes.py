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
  """An environment that executes the actions with a pre-defined
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
    self._default_action = default_action or [0.01, 0.]
    self._num_delay_steps = num_delay_steps or 5
    self._action_queue = queue.Queue()

  def step(self, action):
    self._action_queue.put(action)
    action_to_execute = self._action_queue.get()
    # print("Current action to execute: ", action_to_execute)
    return super().step(action_to_execute)

  def reset(self, scenario=None):
    while not self._action_queue.empty():
      self._action_queue.get()
    for _ in range(0, self._num_delay_steps):
      self._action_queue.put(np.array(self._default_action))
    return super().reset(scenario)

