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

  def ReplaceBehaviorModel(self, agent_id):
    # NOTE: dummy implementation
    return self._world.Copy()
  
  def GetAgentIds(self):
    # NOTE: only use nearby agents
    agent_ids = list(self._world.agents.keys())
    eval_id = self._scenario._eval_agent_ids[0]
    agent_ids.remove(eval_id)
    return agent_ids

  def GenerateCounterfactualWorlds(self):
    cf_worlds = []
    agent_ids = self.GetAgentIds()
    for agent_id in agent_ids:
      return_dict = {}
      return_dict[agent_id] = self.ReplaceBehaviorModel(agent_id)
      cf_worlds.append(return_dict)
    return cf_worlds

  def SimulateWorld(self, world, local_tracer, N=5, **kwargs):
    self.ml_behavior.set_action_externally = False
    eval_id = self._scenario._eval_agent_ids[0]
    self._world.agents[eval_id].behavior_model = self.ml_behavior
    for i in range(0, N):
      world.Step(self._step_time)
      eval_state = world.Evaluate()
      local_tracer.Trace(eval_state, **kwargs)
    self.ml_behavior.set_action_externally = True
  
  def St(self):
    self._start_time = time.time()

  def Et(self):
    end_time = time.time()
    dt = end_time - self._start_time
    self._logger.info(f"It took {dt:.3f} seconds to simulate all" + \
                      f" counterfactual worlds.")

  def step(self, action):
    """perform the cf evaluation"""
    # simulate counterfactual worlds
    local_tracer = Tracer()
    self.St()
    cf_worlds = self.GenerateCounterfactualWorlds()
    for cf_world in cf_worlds:
      cf_key = list(cf_world.keys())[0]
      self.SimulateWorld(
        cf_world[cf_key], local_tracer, N=5, replaced_agent=cf_key)
    self.Et()

    # evaluate counterfactual worlds
    collision_rate = local_tracer.Query(
      key="collision", group_by="replaced_agent", agg_type="MEAN")
    mean_collision_rate = collision_rate.mean()
    self._logger.info(f"The counterfactual worlds have a collision" + \
                      f"-rate of {mean_collision_rate:.3f}.")
    
    # choose a policy
    if mean_collision_rate > self._max_col_rate:
      # NOTE: assign a backup model if the collision-rate is too high
      pass
    return SingleAgentRuntime.step(self, action)