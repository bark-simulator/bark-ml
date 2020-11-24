# Copyright (c) 2020 fortiss GmbH
#
# Authors: Julian Bernhard, Klemens Esterle, Patrick Hart and
# Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import logging

from bark.runtime.commons.parameters import ParameterServer

from bark.core.world.evaluation import *
from bark.core.world import *

from bark.benchmark.benchmark_runner import BenchmarkRunner, BehaviorConfig
from bark.benchmark.benchmark_runner_mp import BenchmarkRunnerMP

from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.agent.training_benchmark_database \
       import default_training_evaluators, default_terminal_criteria

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
      experience = self.GetExperience(world)
      self._episode_experiences.append(experience)
      return self._episode_experiences
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
    reward, done, _ = self.GetStepEvaluation(observed_world, action)
    experience = self.MakeExperienceTuple(self._last_nn_state, action, reward, current_nn_state, done)
    self._last_nn_state = current_nn_state
    return experience

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

  @staticmethod
  def load(filename):
    pass

  def _GetDefaultRunnerInitParams(self):
    return {"log_eval_avg_every" : 5}

  def _GetDefaultRunnerRunParams(self):
    return {"maintain_history" : False, "checkpoint_every" : 10, "viewer" : None}

  def CollectDemonstrations(self, env, demo_behavior, num_episodes, max_episode_steps, filename, use_mp_runner=True, runner_init_params = None,
      runner_run_params=None):
    demo_evaluator = DemonstrationEvaluator(env._observer, env._evaluator)
    evaluators = {**default_training_evaluators(), "demo_evaluator" : demo_evaluator}
    terminal_when = default_terminal_criteria(max_episode_steps)

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
    self._collection_result.dump(filename, dump_histories=False, dump_configs=False)
    return self._collection_result

  def ProcessCollectionResult(self):
    if not self._collection_result:
      logging.errir("Collection results not created yet. Call CollectDemonstrations first.")
      return

    data_frame = self._collection_result.get_data_frame()
    data_frame["success"] = data_frame.Terminal.apply(lambda x: "success" in x and (not "collision" in x) and (not "max_steps" in x))
    print(df)



  def GetDemonstrationExperiences(self):
    pass
  
   

      