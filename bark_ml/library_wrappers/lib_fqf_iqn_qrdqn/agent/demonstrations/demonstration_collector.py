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


class DemonstrationCollector:
  def __init__(self):
    self._collection_result = None
    self._demonstrations = None
    self._directory = None

  def _GetDefaultRunnerInitParams(self):
    return {"log_eval_avg_every" : 5}

  def _GetDefaultRunnerRunParams(self):
    return {"maintain_history" : False, "checkpoint_every" : 10, "viewer" : None}

  def CollectDemonstrations(self, env, demo_behavior, num_episodes, directory, use_mp_runner=True, runner_init_params = None,
      runner_run_params=None):
    demo_evaluator = DemonstrationEvaluator(env._observer, env._evaluator)
    evaluators = {**default_training_evaluators(), "demo_evaluator" : demo_evaluator}
    terminal_when = {"demo_evaluator" : lambda x : x[1] == True} # second index in evaluation result is done

    runner_init_params_def = self._GetDefaultRunnerInitParams()
    runner_init_params_def.update(runner_init_params or {})
    runner_type = BenchmarkRunnerMP if use_mp_runner else BenchmarkRunner
    runner = runner_type(evaluators=evaluators,
                                  scenario_generation=env._scenario_generator,
                                  terminal_when=terminal_when,
                                  behaviors={"demo_behavior" : demo_behavior},
                                  num_scenarios = num_episodes,
                                  **runner_init_params_def)
    runner_run_params_def = self._GetDefaultRunnerRunParams()
    runner_run_params_def.update(runner_run_params or {})
    self._collection_result = runner.run(**runner_run_params_def)
    self.dump(directory)
    return self._collection_result

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

  def ProcessCollectionResult(self, eval_criteria):
    if not self._collection_result:
      logging.error("Collection results not created yet. Call CollectDemonstrations first.")
      return

    data_frame = self._collection_result.get_data_frame()
    self._demonstrations = []
    for index, row in data_frame.iterrows():
      demo_eval_result, done, info = row["demo_evaluator"]
      use_scenario = True
      for crit, func in eval_criteria.items():
        use_scenario = use_scenario and func(info[crit])
      if not use_scenario:
        continue
      self._demonstrations.extend(demo_eval_result[1:])
    self.dump(self._directory)
    return self._demonstrations

  def GetDemonstrationExperiences(self):
    return self._demonstrations

  def GetCollectionResult(self):
    return self._collection_result
  
   

      