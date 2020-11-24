# Copyright (c) 2020 fortiss GmbH
#
# Authors: Julian Bernhard, Klemens Esterle, Patrick Hart and
# Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from bark.runtime.commons.parameters import ParameterServer

from bark.core.world.evaluation import *
from bark.core.world import *

from bark.benchmark.benchmark_runner import BenchmarkRunner, BehaviorConfig
from bark.benchmark.benchmark_runner_mp import BenchmarkRunnerMP

from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.agent.training_benchmark_database \
       import default_training_evaluators, default_terminal_criteria


class DemonstrationEvaluator(BaseEvaluator):
  def __init__(self, nn_observer):
    super(DemonstrationEvaluator, self).__init__()
    self._nn_observer = nn_observer
    self._agent_id = None
    self._last_nn_state = None
    self.episode_experiences = []

  def SetAgentId(self, agent_id):
    self._agent_id = agent_id

  def GetNNInputState(self, world):
    observed_world = world.Observe([self._agent_id])[0]
    return self._nn_observer.Observe(observed_world)

  def GetAction(self, world):
    return world.agents[self._agent_id].behavior.GetLastMacroAction()

  def Evaluate(self, world):
    if isinstance(world, World):
      experience = self.GetExperience(world)
      self.episode_experiences.append(experience)
      return self.episode_experiences
    else:
      raise NotImplementedError()

  def MakeExperienceTuple(self, nn_state, action, next_nn_state):
    demo = True
    done = False # we decide after the benchmark running
    return (nn_state, action, next_nn_state, done, demo)

  def GetExperience(self, observed_world):
    current_nn_state = self.GetNNInputState(observed_world)
    action = self.GetAction(observed_world)
    experience = self.MakeExperienceTuple(self._last_nn_state, action, current_nn_state)
    self._last_nn_state = current_nn_state

  def __setstate__(self, d):
    self._nn_observer = d["observer"]

  def __getstate__(self):
    return {"observer" : self._nn_observer}


class DemonstrationCollector:
  def __init__(self):
    self.collection_result = None

  @staticmethod
  def load(filename):
    pass

  def _GetDefaultRunnerInitParams(self):
    return {"log_eval_avg_every" : 5}

  def _GetDefaultRunnerRunParams(sel):
    return {"maintain_history" : False, "checkpoint_every" : 10}

  def CollectDemonstrations(self, env, demo_behavior, num_episodes, max_episode_steps, filename, runner_init_params = None,
      runner_run_params=None):
    demo_evaluator = DemonstrationEvaluator(env._observer)
    evaluators = {**default_training_evaluators(), "demo_evaluator" : demo_evaluator}
    terminal_when = default_terminal_criteria(max_episode_steps)

    runner_init_params = self._GetDefaultRunnerInitParams()
    runner_init_params.update(runner_init_params)
    runner = BenchmarkRunnerMP(evaluators=evaluators,
                                  terminal_when=terminal_when,
                                  behaviors={"demo_behavior" : demo_behavior},
                                  num_scenarios = num_episodes,
                                  **runner_init_params)
    runner_run_params = self._GetDefaultRunnerRunParams()
    runner_run_params.update(runner_run_params)
    self.collection_result = runner.run(runner_run_params)
    self.collection_result.dump(filename, dump_histories=False, dump_configs=False)

  def ProcessCollectionResult(self):
    pass



  def GetDemonstrationExperiences(self):
    pass
  
   

      