# Copyright (c) 2020 Patrick Hart, Julian Bernhard, 
# Klemens Esterle, Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import numpy as np
from bark.runtime.runtime import Runtime
from bark_ml.behaviors.cont_behavior import BehaviorContinuousML
from bark.runtime.commons.parameters import ParameterServer
from bark.core.models.behavior import BehaviorModel, BehaviorDynamicModel


class SingleAgentRuntime(Runtime):
  def __init__(self,
               blueprint=None,
               ml_behavior=None,
               observer=None,
               evaluator=None,
               step_time=None,
               viewer=None,
               scenario_generator=None,
               render=False):
    
    if blueprint is not None:
      self._scenario_generator = blueprint._scenario_generation
      self._viewer = blueprint._viewer
      self._ml_behavior = blueprint._ml_behavior
      self._step_time = blueprint._dt
      self._evaluator = blueprint._evaluator
      self._observer = blueprint._observer
    Runtime.__init__(self,
                     step_time=step_time or self._step_time,
                     viewer=viewer or self._viewer,
                     scenario_generator=scenario_generator or self._scenario_generator,
                     render=render)
    self._ml_behavior = ml_behavior or self._ml_behavior
    self._observer = observer or self._observer
    self._evaluator = evaluator or self._evaluator

  def reset(self, scenario=None):
    """Resets the runtime and its objects
    """
    super().reset(scenario=scenario)
    assert len(self._scenario._eval_agent_ids) == 1, \
      "This runtime only supports a single agent!"
    eval_id = self._scenario._eval_agent_ids[0]
    self._world = self._observer.Reset(self._world)
    self._world = self._evaluator.Reset(self._world)
    self._world.agents[eval_id].behavior_model = self._ml_behavior
    
    # observe
    observed_world = self._world.Observe([eval_id])[0]
    return self._observer.Observe(observed_world)

  def step(self, action):
    # set actions
    eval_id = self._scenario._eval_agent_ids[0]
    if eval_id in self._world.agents:
      self._world.agents[eval_id].behavior_model.ActionToBehavior(action)

    # step and observe
    self._world.Step(self._step_time)
    observed_world = self._world.Observe([eval_id])[0]

    # observe and evaluate
    observed_next_state = self._observer.Observe(observed_world)
    reward, done, info = self._evaluator.Evaluate(
      observed_world=observed_world,
      action=action)
    
    # print(action, observed_next_state, reward)
    # render
    if self._render:
      self.render()
    return observed_next_state, reward, done, info

  @property
  def action_space(self):
    """Action space of the agent
    """
    return self._ml_behavior.action_space

  @property
  def observation_space(self):
    """Observation space of the agent
    """
    return self._observer.observation_space

  @property
  def ml_behavior(self):
    return self._ml_behavior

  @ml_behavior.setter
  def ml_behavior(self, ml_behavior):
    self._ml_behavior = ml_behavior