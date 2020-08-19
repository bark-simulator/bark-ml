# Copyright (c) 2020 fortiss GmbH
#
# Authors: Patrick Hart
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
import time
import numpy as np
import logging

# bark
from bark.runtime.commons.parameters import ParameterServer

# bark-ml
from bark_ml.environments.single_agent_runtime import SingleAgentRuntime
from bark_ml.commons.tracer import Tracer


class CounterfactualRuntime(SingleAgentRuntime):
  def __init__(self,
               blueprint=None,
               ml_behavior=None,
               observer=None,
               evaluator=None,
               step_time=None,
               viewer=None,
               scenario_generator=None,
               render=False,
               max_col_rate=0.1):
    SingleAgentRuntime.__init__(
      self,
      blueprint=blueprint,
      ml_behavior=ml_behavior,
      observer=observer,
      evaluator=evaluator,
      step_time=step_time,
      viewer=viewer,
      scenario_generator=scenario_generator,
      render=render)
    self._max_col_rate = max_col_rate
    self._logger = logging.getLogger()

  def reset(self, scenario=None):
    """Resets the runtime and its objects
    """
    return SingleAgentRuntime.reset(self, scenario=scenario)

  def GenerateCounterfactualWorlds(self):
    return [self._world.Copy() for _ in range(0, 10)]

  def SimulateWorld(self, world, N=5, **kwargs):
    self.ml_behavior.set_action_externally = False
    eval_id = self._scenario._eval_agent_ids[0]
    self._world.agents[eval_id].behavior_model = self.ml_behavior
    for i in range(0, N):
      world.Step(self._step_time)
      eval_state = world.Evaluate()
      self._tracer.Trace(eval_state, **kwargs)
    self.ml_behavior.set_action_externally = True
  
  def step(self, action):
    """perform the cf evaluation"""
    self._tracer = Tracer()
    # NOTE: clone self._world M times
    # NOTE: this should be a dict {"world_0": {exchanged_behaviors: }}
    cf_worlds = self.GenerateCounterfactualWorlds()
    start_time = time.time()
    for i, cf_world in enumerate(cf_worlds):
      self.SimulateWorld(cf_world, N=5, num_episode=i)
    end_time = time.time()
    dt = end_time-start_time
    self._logger.info(f"It took {dt:.3f} seconds to simulate the" + \
                      f" counterfactual worlds.")
    collision_rate = self._tracer.Query(
      key="collision", group_by="num_episode", agg_type="MEAN").mean()
    self._logger.info(f"The counterfactual worlds have a collision" + \
                      f"-rate of {collision_rate:.3f}.")
    if collision_rate > self._max_col_rate:
      # NOTE: assign a backup model if the collision-rate is too high
      pass
    return SingleAgentRuntime.step(self, action)