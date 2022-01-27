# Copyright (c) 2020 fortiss GmbH
#
# Authors: Patrick Hart
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import time
import numpy as np
import logging
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
  """Counterfactual runtime for evaluating behavior policies.

  Based on the publication "Counterfactual Policy Evaluation for
  Decision-Making in Autonomous Driving"
  (https://arxiv.org/abs/2003.11919)
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
      "Whether the counterfactual worlds are visualized.", False]
    self._visualize_heatmap = params["ML"][
      "VisualizeCfHeatmap",
      "Whether the heatmap is being visualized.", False]
    self._results_folder = params["ML"][
      "ResultsFolder",
      "Whether the heatmap is being visualized.", "./"]
    self._logger = logging.getLogger()
    self._behavior_model_pool = behavior_model_pool or []
    self._ego_rule_based = ego_rule_based or BehaviorIDMLaneTracking(self._params)
    self._tracer = Tracer()
    if self._visualize_heatmap:
      _, self._axs_heatmap = plt.subplots(1, 1, constrained_layout=True)
    self._count = 0
    self._cf_axs = {}

  def reset(self, scenario=None):
    """Resets the runtime and its objects."""
    self._count = 0
    return SingleAgentRuntime.reset(self, scenario=scenario)

  def ReplaceBehaviorModel(self, agent_id=None, behavior=None):
    """Clones the world and replaced the behavior of an agent."""
    cloned_world = self._world.Copy()
    evaluators = self._evaluator._bark_eval_fns
    for eval_key, eval_fn in evaluators.items():
      cloned_world.AddEvaluator(eval_key, eval_fn())
    if behavior is not None:
      cloned_world.agents[agent_id].behavior_model = behavior
    return cloned_world

  def GetAgentIds(self):
    """Returns a list of the other agent's ids."""
    # NOTE: only use nearby agents
    agent_ids = list(self._world.agents.keys())
    # eval_id = self._scenario._eval_agent_ids[0]
    # agent_ids.remove(eval_id)
    return agent_ids

  def GenerateCounterfactualWorlds(self):
    """Generates (len(agents) - 1) x M-behavior counterfactual worlds."""
    cf_worlds = []
    agent_ids = self.GetAgentIds()
    for agent_id in agent_ids:
      for behavior in self._behavior_model_pool:
        return_dict = {}
        return_dict[agent_id] = self.ReplaceBehaviorModel(agent_id, behavior)
        cf_worlds.append(return_dict)
    return cf_worlds

  def SimulateWorld(self, world, local_tracer, N=5, **kwargs):
    """Simulates the world for N steps."""
    self.ml_behavior.set_action_externally = False
    eval_id = self._scenario._eval_agent_ids[0]
    self._world.agents[eval_id].behavior_model = self.ml_behavior
    replaced_agent_id = kwargs.get("replaced_agent", 0)
    if replaced_agent_id not in self._cf_axs and self._visualize_cf_worlds:
      self._cf_axs[replaced_agent_id] = {"ax": plt.subplots(3, 1, constrained_layout=True)[1], "count": 0}
    for i in range(0, N):
      if i == N - 1 and kwargs.get("num_virtual_world", 0) is not None and \
        self._visualize_cf_worlds and replaced_agent_id is not None:
        # NOTE: outsource
        for ftype in [".png", ".pgf"]:
          viewer = MPViewer(
            params=self._params,
            x_range=[-35, 35],
            y_range=[-35, 35],
            follow_agent_id=True,
            axis=self._cf_axs[replaced_agent_id]["ax"][self._cf_axs[replaced_agent_id]["count"]])
          # se
          for agent_id in world.agents.keys():
            viewer.agent_color_map[agent_id] = "gray"
          viewer.agent_color_map[replaced_agent_id] = (127/255, 205/255, 187/255)
          viewer.agent_color_map[eval_id] = (34/255, 94/255, 168/255)
          if replaced_agent_id == 1:
            viewer.drawWorld(
              world,
              eval_agent_ids=self._scenario._eval_agent_ids,
              filename=self._results_folder + "cf_%03d_replaced_" % self._count + str(replaced_agent_id)+ftype,
              debug_text=False)
        self._cf_axs[replaced_agent_id]["count"] += 1
      observed_world = world.Observe([eval_id])[0]
      eval_state = observed_world.Evaluate()
      agent_states = CaptureAgentStates(observed_world)
      eval_state = {**eval_state, **agent_states}
      # TODO: break at collision
      local_tracer.Trace(eval_state, **kwargs)
      if eval_state["collision"] or eval_state["drivable_area"]:
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
    collision_rate = local_tracer.collision_rate
    goal_reached = local_tracer.success_rate
    return {"collision": collision_rate,
            "goal_reached": goal_reached,
            "max_col_rate": self._max_col_rate}

  @staticmethod
  def FilterStates(states, **kwargs):
    states_ = []
    for state in states:
      for kwarg_key, kwarg_val in kwargs.items():
        if kwarg_key in state:
          if state[kwarg_key] == kwarg_val:
            states_.append(state)
    return states_

  @staticmethod
  def ExtractStatesPerWorld(states):
    pure_states_ = {}
    for state in states:
      world_idx = state["num_virtual_world"]
      pure_states_[world_idx] = []
      for key, item in state.items():
        if "state_" in key:
          pure_states_[world_idx].append(item)
      pure_states_[world_idx] = np.array(pure_states_[world_idx])
    return pure_states_

  def GetMeanForAgent(self, local_tracer, agent_id):
    filtered_states = self.FilterStates(local_tracer._states, replaced_agent=agent_id)
    extracted_states = self.ExtractStatesPerWorld(filtered_states)
    # print(extracted_states, agent_id)
    mean = None
    for v in extracted_states.values():
      if mean is None:
        mean = v
      else:
        mean += v
    mean /= len(extracted_states)
    return mean

  def DrawHeatmap(self, local_tracer, filename="./"):
    base_states = self.FilterStates(local_tracer._states, replaced_agent="None")
    extracted_base_states = self.ExtractStatesPerWorld(base_states)
    extracted_base_states_np = extracted_base_states[
      list(extracted_base_states.keys())[0]]

    # loop through all agents
    all_keys = list(local_tracer._states[0].keys())
    all_agent_ids = []
    for i, key in enumerate(all_keys):
      if "state_" in key:
        all_agent_ids.append(int(key.replace("state_", "")))
    # TODO: the ego agent is not replaced, but want influence
    arr = np.zeros(shape=(len(all_agent_ids), len(all_agent_ids)))
    for i, agent_id in enumerate(all_agent_ids):
      print(i, agent_id)
      mean = self.GetMeanForAgent(local_tracer, agent_id)
      row_from = np.sum((extracted_base_states_np - mean)**2, axis=1)
      arr[i, :] = row_from

    np.fill_diagonal(arr, 0.)
    self._axs_heatmap.imshow(arr, cmap=plt.get_cmap('Blues'))
    self._axs_heatmap.set_yticks(np.arange(len(all_agent_ids)))
    self._axs_heatmap.set_xticks(np.arange(len(all_agent_ids)))
    self._axs_heatmap.set_yticklabels(["$W^{v_"+str(agent_id)+"}$" for agent_id in all_agent_ids])
    self._axs_heatmap.set_xticklabels(["$\Delta_{v_"+str(agent_id)+"}$" for agent_id in all_agent_ids])
    self._axs_heatmap.set_rasterized(True)

    self._axs_heatmap.get_figure().savefig(filename+".png")
    self._axs_heatmap.get_figure().savefig(filename+".pgf")

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

    # NOTE: this world would actually have the predicted traj.
    gt_world = self.ReplaceBehaviorModel()
    self.SimulateWorld(
      gt_world, local_tracer, N=self._cf_simulation_steps,
      replaced_agent="None", num_virtual_world="None")
    # NOTE: outsource
    hist = gt_world.agents[eval_id].history
    traj = np.stack([x[0] for x in hist])
    # self._viewer.drawTrajectory(traj, color='blue')

    if self._visualize_heatmap:
      self.DrawHeatmap(
        local_tracer,
        filename=self._results_folder + "cf_%03d" % self._count + "_heatmap")

    # evaluate counterfactual worlds
    trace = self.TraceCounterfactualWorldStats(local_tracer)
    collision_rate = trace['collision']/len(self._behavior_model_pool)
    print(collision_rate)
    self._logger.info(
      f"The counterfactual worlds have a collision" + \
      f"-rate of {collision_rate:.3f}.")

    # choose a policy
    executed_learned_policy = 1
    if collision_rate > self._max_col_rate:
      executed_learned_policy = 0
      self._logger.info(
        f"Executing fallback model.")
      self._world.agents[eval_id].behavior_model = self._ego_rule_based
    trace["executed_learned_policy"] = executed_learned_policy
    self._tracer.Trace(trace)
    self._count += 1
    for fig in self._cf_axs.values():
      for sub_ax in fig["ax"]:
        sub_ax.clear()
    return SingleAgentRuntime.step(self, action)