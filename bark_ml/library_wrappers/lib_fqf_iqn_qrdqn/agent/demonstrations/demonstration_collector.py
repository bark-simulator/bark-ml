# Copyright (c) 2020 fortiss GmbH
#
# Authors: Julian Bernhard, Klemens Esterle, Patrick Hart and
# Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import logging
import pickle
import os
import numpy as np

from bark.runtime.commons.parameters import ParameterServer

from bark.core.world.evaluation import *
from bark.core.world import *

from bark.benchmark.benchmark_result import BenchmarkResult
from bark.benchmark.benchmark_runner import BenchmarkRunner, BehaviorConfig
from bark.benchmark.benchmark_runner_mp import BenchmarkRunnerMP

from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.agent.training_benchmark_database \
       import default_training_evaluators, default_terminal_criteria

def to_pickle(obj, dir, file):
  path = os.path.join(dir, file)
  with open(path, 'wb') as handle:
    pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def from_pickle(dir, file):
  path = os.path.join(dir, file)
  with open(path, 'rb') as handle:
    obj = pickle.load(handle)
  return obj


class DemonstrationEvaluator(BaseEvaluator):
  def __init__(self, nn_observer, reward_evaluator):
    super(DemonstrationEvaluator, self).__init__()
    self._nn_observer = nn_observer
    self._reward_evaluator = reward_evaluator
    self._agent_id = None
    self._last_nn_state = None
    self._episode_experiences = []

  def SetAgentId(self, agent_id):
    self._agent_id = agent_id

  def GetNNInputState(self, observed_world):
    return self._nn_observer.Observe(observed_world)

  def GetAction(self, observed_world):
    return observed_world.agents[self._agent_id].behavior_model.GetLastMacroAction()

  def GetStepEvaluation(self, observed_world, action):
    return self._reward_evaluator.Evaluate(observed_world, action)

  def Evaluate(self, world):
    if isinstance(world, World):
      experience, done, info = self.GetExperience(world)
      self._episode_experiences.append(experience)
      return (self._episode_experiences, done, info)
    else:
      raise NotImplementedError()

  def MakeExperienceTuple(self, nn_state, action, reward, next_nn_state, done):
    demo = True
    return (nn_state, action, reward, next_nn_state, done, demo)

  def GetExperience(self, world):
    if len(self._episode_experiences) == 0:
      self._reward_evaluator.Reset(world)
    observed_world = world.Observe([self._agent_id])[0]
    current_nn_state = self.GetNNInputState(observed_world)
    action = self.GetAction(observed_world)
    reward, done, info = self.GetStepEvaluation(observed_world, action)
    experience = self.MakeExperienceTuple(self._last_nn_state, action, reward, current_nn_state, done)
    self._last_nn_state = current_nn_state
    return experience, done, info

  def __setstate__(self, d):
    self._nn_observer = d["observer"]
    self._reward_evaluator = d["reward_evaluator"]
    self._agent_id = None
    self._last_nn_state = None
    self._episode_experiences = []

  def __getstate__(self):
    return {"observer" : self._nn_observer, \
          "reward_evaluator" : self._reward_evaluator}

class ActionValueEvaluator(BaseEvaluator):
  def __init__(self, nn_observer):
    self._agent_id = None
    self._nn_observer = nn_observer

  def SetAgentId(self, agent_id):
    self._agent_id = agent_id

  def GetExperience(self, world):
    observed_world = world.Observe([self._agent_id])[0]
    current_nn_state = self._nn_observer.Observe(observed_world)
    action_values = self.GetActionValues(observed_world)
    return current_nn_state, action_values

  def AddMissingActionsValues(self, value_dict, num_actions):
    values = []
    assert(num_actions >= len(value_dict))
    for idx in  range(0, num_actions):
      if idx in value_dict:
        values.append(value_dict[idx])
      else:
        values.append(np.nan)
    return values

  def GetActionValues(self, observed_world):
    behavior = observed_world.agents[self._agent_id].behavior_model
    num_actions = len(behavior.ego_behavior.GetMotionPrimitives())
    cost_envelope_values = []
    cost_collision_values = []
    if "envelope" in behavior.last_cost_values:
      cost_envelope_values = \
           self.AddMissingActionsValues(behavior.last_cost_values["envelope"], num_actions)
    cost_collision_values = []
    if "collision" in behavior.last_cost_values:
      cost_collision_values = \
        self.AddMissingActionsValues(behavior.last_cost_values["collision"], num_actions)
    return_values = self.AddMissingActionsValues(behavior.last_return_values, num_actions)
    action_values = []
    action_values.extend(cost_envelope_values)
    action_values.extend(cost_collision_values)
    action_values.extend(return_values)
    return action_values

  def Evaluate(self, world):
    if isinstance(world, World):
      return self.GetExperience(world)
    else:
      raise NotImplementedError()

  def __setstate__(self, d):
    self._nn_observer = d["observer"]
    self._agent_id = None

  def __getstate__(self):
    return {"observer" : self._nn_observer}


class DemonstrationCollector:
  def __init__(self):
    self._collection_result = None
    self._demonstrations = None
    self._directory = None

  def _GetDefaultRunnerInitParams(self):
    return {"log_eval_avg_every" : 5}

  def _GetDefaultRunnerRunParams(self):
    return {"maintain_history" : False, "checkpoint_every" : 10, "viewer" : None}

  def CollectDemonstrations(self, num_episodes, directory,
       env=None, nn_observer=None, reward_evaluator=None, demo_behavior=None, benchmark_configs=None,
       use_mp_runner=True, runner_init_params = None,
      runner_run_params=None):

    if env:
      nn_observer = env._observer
      reward_evaluator = env._evaluator
      scenario_generator = env._scenario_generator
    else:
      scenario_generator = None
    if demo_behavior and env:
      behaviors = {"demo_behavior" : demo_behavior}
    else:
      behaviors = None
    demo_evaluator = self.GetEvaluators(nn_observer, reward_evaluator)
    evaluators = {**default_training_evaluators(), "demo_evaluator" : demo_evaluator}
    terminal_when = self.GetTerminalCriteria()

    runner_init_params_def = self._GetDefaultRunnerInitParams()
    runner_init_params_def.update(runner_init_params or {})
    runner_type = BenchmarkRunnerMP if use_mp_runner else BenchmarkRunner
    runner = runner_type(evaluators=evaluators,
                                  scenario_generation=scenario_generator,
                                  benchmark_configs = benchmark_configs,
                                  terminal_when=terminal_when,
                                  behaviors=behaviors,
                                  num_scenarios = num_episodes,
                                  **runner_init_params_def)
    runner.clear_checkpoint_dir()
    runner_run_params_def = self._GetDefaultRunnerRunParams()
    runner_run_params_def.update(runner_run_params or {})
    self._collection_result = runner.run(**runner_run_params_def)
    self.dump(directory)
    return self._collection_result

  def GetEvaluators(self, nn_observer, reward_evaluator):
    return DemonstrationEvaluator(nn_observer, reward_evaluator)

  def GetTerminalCriteria(self, *args):
    return {"demo_evaluator" : lambda x : x[1] == True} # second index in evaluation result is done

  def dump(self, directory):
    self._directory = directory
    if not os.path.exists(directory):
      os.makedirs(directory)
    if self._collection_result:
      self._collection_result.dump(os.path.join(directory, DemonstrationCollector.collection_result_filename()), dump_histories=False, dump_configs=False)
    if self._demonstrations:
      to_pickle(self._demonstrations, directory, DemonstrationCollector.demonstrations_filename())

  @staticmethod
  def load(directory):
    collector = DemonstrationCollector()
    collection_result_fullname = os.path.join(directory, DemonstrationCollector.collection_result_filename())
    if os.path.exists(collection_result_fullname):
      collector._collection_result = BenchmarkResult.load(collection_result_fullname)
    demonstration_fullname = os.path.join(directory, DemonstrationCollector.demonstrations_filename())
    if os.path.exists(demonstration_fullname):
      collector._demonstrations = from_pickle(directory, DemonstrationCollector.demonstrations_filename())
    collector._directory = directory
    return collector

  @staticmethod
  def collection_result_filename():
    return "collection_result"

  @staticmethod
  def demonstrations_filename():
    return "demonstrations"

  def UseCollectedRow(self, row, eval_criteria):
    demo_eval_result, done, info = row["demo_evaluator"]
    use_scenario = True
    for crit, func in eval_criteria.items():
      use_scenario = use_scenario and func(info[crit])
    return use_scenario

  def GetDemonstrations(self, row):
    demo_eval_result, done, info = row["demo_evaluator"]
    return demo_eval_result[1:]

  def ProcessCollectionResult(self, eval_criteria = None):
    if not self._collection_result:
      logging.error("Collection results not created yet. Call CollectDemonstrations first.")
      return

    data_frame = self._collection_result.get_data_frame()
    self._demonstrations = []
    for index, row in data_frame.iterrows():
      if not eval_criteria or self.UseCollectedRow(row, eval_criteria):
        self._demonstrations.extend(self.GetDemonstrations(row))
    self.dump(self._directory)
    return self._demonstrations

  def GetDemonstrationExperiences(self):
    return self._demonstrations

  def GetCollectionResult(self):
    return self._collection_result
  

class ActionValuesCollector(DemonstrationCollector):
  def __init__(self, terminal_criteria):
    super(ActionValuesCollector, self).__init__()
    self.terminal_criteria = terminal_criteria

  def UseCollectedRow(self, row, eval_criteria):
    use_scenario = True
    for crit, func in eval_criteria.items():
      use_scenario = use_scenario and func(row[crit])
    return use_scenario

  def GetDemonstrations(self, row):
    return row["demo_evaluator"][1:]

  def GetEvaluators(self, nn_observer, reward_evaluator):
    return ActionValueEvaluator(nn_observer)

  def GetTerminalCriteria(self, *args):
    return self.terminal_criteria