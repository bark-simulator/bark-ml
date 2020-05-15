# Copyright (c) 2019 Patrick Hart, Julian Bernhard, 
# Klemens Esterle, Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import numpy as np
from modules.runtime.runtime import Runtime


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
    
    # TODO(@hart): CreateFromBlueprint(..)
    if blueprint is not None:
      scenario_generator = blueprint._scenario_generation
      viewer = blueprint._viewer
      ml_behavior = blueprint._ml_behavior
      step_time = blueprint._dt
      evaluator = blueprint._evaluator
      observer = blueprint._observer
    
    Runtime.__init__(self,
                     step_time=step_time,
                     viewer=viewer,
                     scenario_generator=scenario_generator,
                     render=render)
    self._ml_behavior = ml_behavior
    self._observer = observer
    self._evaluator = evaluator

  def reset(self, scenario=None):
    """Resets the runtime and its objects
    """
    super().reset(scenario=scenario)
    assert len(self._scenario._eval_agent_ids) == 1, \
      "This runtime only supports an single agent!"
    eval_id = self._scenario._eval_agent_ids[0]
    self._world = self._observer.Reset(self._world,
                                       self._scenario._eval_agent_ids)
    self._world = self._evaluator.Reset(self._world)
    self._world.agents[eval_id] = self._ml_behavior.Reset()
    
    # observe
    observed_world = self._world.Observe([eval_id])[0]
    return self._observer.observe(observed_world)

  def step(self, action):
    # set actions
    eval_id = self._scenario._eval_agent_ids[0]
    self._world.agents[eval_id].ActionToBehavior(action)

    # step and observe
    self._world.Step(self._step_time)
    observed_world = self._world.Observe([eval_id])[0]

    # observe and evaluate
    observed_next_state = self._observer.observe(observed_world)
    reward, done, info = self._evaluator.evaluate(
      observed_world=observed_world,
      action=action)
  
    # render
    if self._render:
      self.render()
    return observed_next_state, reward, done, info

  @property
  def action_space(self):
    """Action space of the agent
    """
    return self._behavior_ml.action_space

  @property
  def observation_space(self):
    """Observation space of the agent
    """
    return self._observer.observation_space
