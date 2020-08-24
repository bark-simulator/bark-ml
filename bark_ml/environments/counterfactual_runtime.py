# Copyright (c) 2020 fortiss GmbH
#
# Authors: Patrick Hart
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import time
import numpy as np
import logging
import copy
import matplotlib.pyplot as plt

# bark
from bark.runtime.commons.parameters import ParameterServer
from bark.core.models.behavior import BehaviorIDMLaneTracking
from bark.core.world.evaluation import CaptureAgentStates
from bark.runtime.viewer.matplotlib_viewer import MPViewer

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
      "Simulation steps for the counterfactual worlds.", 5]
    self._visualize_cf_worlds = params["ML"][
      "VisualizeCfWorlds",
      "Whether the counterfactual worlds are visualized.", True]
    self._visualize_heatmap = params["ML"][
      "VisualizeCfHeatmap",
      "Whether the heatmap is being visualized.", True]
    self._logger = logging.getLogger()
    self._behavior_model_pool = behavior_model_pool or []
    self._ego_rule_based = ego_rule_based or BehaviorIDMLaneTracking(self._params)
    self._tracer = Tracer()
    if self._visualize_heatmap:
      _, self._axs_heatmap = plt.subplots(1, 1)
    self._count = 0
    self._cf_axs = {}
    
  def reset(self, scenario=None):
    """resets the runtime and its objects"""
    self._count = 0
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
    replaced_agent_id = kwargs.get("replaced_agent", 0)
    if replaced_agent_id not in self._cf_axs and self._visualize_cf_worlds:
      self._cf_axs[replaced_agent_id] = {"ax": plt.subplots(3, 1)[1], "count": 0}
    for i in range(0, N):
      if i == N - 1 and kwargs.get("num_virtual_world", 0) is not None and \
        self._visualize_cf_worlds and replaced_agent_id is not None:
        viewer = MPViewer(
           params=self._params,
           x_range=[-35, 35],
           y_range=[-35, 35],
           follow_agent_id=True,
           axis=self._cf_axs[replaced_agent_id]["ax"][self._cf_axs[replaced_agent_id]["count"]])
        viewer.drawWorld(
          world,
          eval_agent_ids=self._scenario._eval_agent_ids,
          filename="/Users/hart/Development/bark-ml/results/cf_"+str(self._count)+"_replaced_"+str(replaced_agent_id)+".png",
          debug_text=False)
        self._cf_axs[replaced_agent_id]["count"] += 1
      observed_world = world.Observe([eval_id])[0]
      eval_state = observed_world.Evaluate()
      agent_states = CaptureAgentStates(observed_world)
      eval_state = {**eval_state, **agent_states}
      local_tracer.Trace(eval_state, **kwargs)
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
      key="collision", group_by="replaced_agent", agg_type="ANY_TRUE")
    collision_rate_drivable_area = local_tracer.Query(
      key="drivable_area", group_by="replaced_agent", agg_type="ANY_TRUE")
    goal_reached = local_tracer.Query(
      key="goal_reached", group_by="replaced_agent", agg_type="ANY_TRUE")
    return {"collision": collision_rate.mean(),
            "drivable_area": collision_rate_drivable_area.mean(),
            "goal_reached": goal_reached.mean()}

  def DrawHeatmap(self, local_tracer, filename="./"):
    eval_id = self._scenario._eval_agent_ids[0]
    agent_ids = list(self._world.agents.keys())
    agent_ids_removed = copy.copy(agent_ids)
    agent_ids_removed.remove(eval_id)
    arr = np.zeros(
      shape=(len(agent_ids), len(agent_ids_removed)), dtype=np.float32)
    for i, agent_id in enumerate(agent_ids):
      # we view only agent_id
      grouped_df = local_tracer.df.groupby(
        ["num_virtual_world", "replaced_agent"])["state_"+str(agent_id)].apply(
          lambda group_series: group_series.tolist())
      # gt for agent_id
      gt_traj = np.stack(
        grouped_df.iloc[
          grouped_df.index.get_level_values("replaced_agent") == "None"][0])
      for j, aid_r in enumerate(agent_ids_removed):
        a = grouped_df.iloc[
          grouped_df.index.get_level_values("replaced_agent") == aid_r]
        diff = []
        for _, a_ in a.items():
          diff.append(np.mean((np.stack(a_) - gt_traj)**2, axis=(0, 1)))
        if aid_r != agent_id:
          arr[i, j] = np.mean(np.array(diff, dtype=np.float32))
    self._axs_heatmap.imshow(arr)
    self._axs_heatmap.set_yticks(np.arange(len(agent_ids)))
    self._axs_heatmap.set_xticks(np.arange(len(agent_ids_removed)))
    def GetName(idx, eval_id):
      if idx == eval_id:
        idx = "ego"
      return "$v_{"+str(idx)+"}$"
    self._axs_heatmap.set_yticklabels([GetName(x, eval_id) for x in agent_ids])
    self._axs_heatmap.set_xticklabels(
      ["$r_{"+str(x)+"}$" for x in agent_ids_removed])
    # NOTE: save heatmap for _count
    self._axs_heatmap.get_figure().savefig(filename)
    
  def step(self, action):
    """perform the cf evaluation"""
    # simulate counterfactual worlds
    local_tracer = Tracer()
    eval_id = self._scenario._eval_agent_ids[0]
    self.St()
    cf_worlds = self.GenerateCounterfactualWorlds()
    for v in self._cf_axs.values():
      v["count"] = 0
    for i, cf_world in enumerate(cf_worlds):
      cf_key = list(cf_world.keys())[0]
      self.SimulateWorld(
        cf_world[cf_key], local_tracer, N=self._cf_simulation_steps,
        replaced_agent=cf_key, num_virtual_world=i)
    self.Et()
    gt_world = self.ReplaceBehaviorModel()
    self.SimulateWorld(
      gt_world, local_tracer, N=self._cf_simulation_steps,
      replaced_agent="None", num_virtual_world="None")

    if self._visualize_heatmap:
      self.DrawHeatmap(
        local_tracer,
        filename="/Users/hart/Development/bark-ml/results/cf_"+str(self._count)+"_heatmap.png")
  
    # evaluate counterfactual worlds
    trace = self.TraceCounterfactualWorldStats(local_tracer)
    self._logger.info(
      f"The counterfactual worlds have a collision" + \
      f"-rate of {trace['collision']:.3f} and a drivable-area-rate " + \
      f"of {trace['drivable_area']:.3f}.")

    # choose a policy
    executed_learned_policy = 1
    if trace["collision"] > self._max_col_rate or \
      trace["drivable_area"] > self._max_col_rate:
      executed_learned_policy = 0
      self._world.agents[eval_id].behavior_model = self._ego_rule_based
    trace["executed_learned_policy"] = executed_learned_policy
    self._tracer.Trace(trace)
    self._count += 1
    return SingleAgentRuntime.step(self, action)