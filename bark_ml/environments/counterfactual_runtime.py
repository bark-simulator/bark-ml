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
from bark.core.models.behavior import BehaviorIDMLaneTracking

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
               max_col_rate=0.1,
               behavior_model_pool=None,
               ego_rule_based=None,
               params=None):
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
    self._params = params or ParameterServer()
    self._max_col_rate = params["ML"][
      "MaxColRate",
      "Max. collision rate allowed over all counterfactual worlds.", 0.1]
    self._cf_simulation_steps = params["ML"][
      "CfSimSteps",
      "Simulation steps for the counterfactual worlds.", 10]
    self._logger = logging.getLogger()
    self._behavior_model_pool = behavior_model_pool or []
    self._ego_rule_based = ego_rule_based or BehaviorIDMLaneTracking(self._params)
    self._tracer = Tracer()

  def reset(self, scenario=None):
    """resets the runtime and its objects"""
    return SingleAgentRuntime.reset(self, scenario=scenario)

  def ReplaceBehaviorModel(self, agent_id=None, behavior=None):
    """clones the world and replaced the behavior of an agent"""
    cloned_world = self._world.Copy()
    evaluators = self._evaluator._add_evaluators()
    for eval_key, eval in evaluators.items():
      cloned_world.AddEvaluator(eval_key, eval)
    if behavior is not None:
      cloned_world.agents[agent_id].behavior_model = behavior
    return cloned_world
  
  def GetAgentIds(self):
    """returns a list of the other agent's ids"""
    # NOTE: only use nearby agents
    agent_ids = list(self._world.agents.keys())
    eval_id = self._scenario._eval_agent_ids[0]
    agent_ids.remove(eval_id)
    return agent_ids

  def GenerateCounterfactualWorlds(self):
    """generates (len(agents) - 1) x M-behavior counterfactual worlds"""
    cf_worlds = []
    agent_ids = self.GetAgentIds()
    for agent_id in agent_ids:
      for behavior in self._behavior_model_pool:
        return_dict = {}
        return_dict[agent_id] = self.ReplaceBehaviorModel(agent_id, behavior)
        cf_worlds.append(return_dict)
    return cf_worlds

  def SimulateWorld(self, world, local_tracer, N=5, **kwargs):
    """simulates the world for N steps"""
    self.ml_behavior.set_action_externally = False
    eval_id = self._scenario._eval_agent_ids[0]
    self._world.agents[eval_id].behavior_model = self.ml_behavior
    for i in range(0, N):
      # NOTE: clear, draw and save using self._count + num_virtual_world + replaced_agent
      # self._viewer.clear()
      # self._viewer.drawWorld(
      #   world, eval_agent_ids=self._scenario._eval_agent_ids)
      observed_world = world.Observe([eval_id])[0]
      eval_state = observed_world.Evaluate()
      if "states" in eval_state:
        for key, state in eval_state["states"].items():
          eval_state[key] = state
        del eval_state["states"]
      local_tracer.Trace(eval_state, **kwargs)
      if eval_state["collision"] == True or eval_state["drivable_area"] == True:
        break
      world.Step(self._step_time)
    self.ml_behavior.set_action_externally = True
  
  def St(self):
    self._start_time = time.time()

  def Et(self):
    end_time = time.time()
    dt = end_time - self._start_time
    self._logger.info(f"It took {dt:.3f} seconds to simulate all" + \
                      f" counterfactual worlds.")
  @property
  def tracer(self):
    return self._tracer
  
  def TraceCounterfactualWorldStats(self, local_tracer):
    collision_rate = local_tracer.Query(
      key="collision", group_by="replaced_agent", agg_type="MEAN")
    collision_rate_drivable_area = local_tracer.Query(
      key="drivable_area", group_by="replaced_agent", agg_type="MEAN")
    goal_reached = local_tracer.Query(
      key="goal_reached", group_by="replaced_agent", agg_type="MEAN")
    return {"collision": collision_rate.mean(),
            "drivable_area": collision_rate_drivable_area.mean(),
            "goal_reached": goal_reached.mean()}

  def step(self, action):
    """perform the cf evaluation"""
    # simulate counterfactual worlds
    local_tracer = Tracer()
    eval_id = self._scenario._eval_agent_ids[0]
    self.St()
    cf_worlds = self.GenerateCounterfactualWorlds()
    for i, cf_world in enumerate(cf_worlds):
      cf_key = list(cf_world.keys())[0]
      self.SimulateWorld(
        cf_world[cf_key], local_tracer, N=self._cf_simulation_steps,
        replaced_agent=cf_key, num_virtual_world=i)
    self.Et()
    gt_world = self.ReplaceBehaviorModel()
    self.SimulateWorld(
      gt_world, local_tracer, N=self._cf_simulation_steps,
      replaced_agent="None", num_virtual_world=i+1)

    # NOTE: DrawHeatmap(local_tracer)
    # agent_ids = list(self._world.agents.keys())
    # grouped_df = local_tracer.df.groupby(
    #   ["num_virtual_world", "replaced_agent"])["state_0"].apply(
    #     lambda group_series: group_series.tolist())
    # self._logger.info( 
    #   grouped_df.iloc[grouped_df.index.get_level_values("replaced_agent") == agent_ids[0]])
    # self._logger.info( 
    #   grouped_df.iloc[grouped_df.index.get_level_values("replaced_agent") == "None"])
  
    # evaluate counterfactual worlds
    trace = self.TraceCounterfactualWorldStats(local_tracer)
    mean_collision_rate = trace["collision"] + trace["drivable_area"]
    self._logger.info(f"The counterfactual worlds have a collision" + \
                      f"-rate of {mean_collision_rate:.3f}.")

    # choose a policy
    executed_learned_policy = 1
    if mean_collision_rate > self._max_col_rate:
      executed_learned_policy = 0
      self._world.agents[eval_id].behavior_model = self._ego_rule_based
    trace["executed_learned_policy"] = executed_learned_policy
    self._tracer.Trace(trace)
    return SingleAgentRuntime.step(self, action)