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

from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.training_benchmark_database \
       import default_training_evaluators, default_terminal_criteria


class DemonstrationEvaluator(BaseEvaluator):
  def __init__(self, demo_behavior, observer):
    super(DemonstrationEvaluator, self).__init__()
    self._demo_behavior = demo_behavior
    self._observer = observer
    self._agent_id = None
    self._last_nn_state = None
    self.episode_experiences = []

  def SetAgentId(self, agent_id):
    self._agent_id = agent_id

  def GetNNInputState(self, observed_world):
    pass

  def GetAction(self, observed_world):
    pass

  def Evaluate(self, observed_world):
    if isinstance(observed_world, World):

    experience = self.GetExperience(observed_world)
    self.episode_experiences.append(experience)
    return self.episode_experiences

  def MakeExperienceTuple(self, nn_state, action, next_nn_state):
    demo = True
    done = False # we decide after the benchmark running
    return (nn_state, action, next_nn_state, done, demo)

  def GetExperience(self, observed_world):
    current_nn_state = self.GetNNInputState(observed_world)
    action = self.GetAction(observed_world)
    experience = self.MakeExperienceTuple(self._last_nn_state, action, current_nn_state)
    self._last_nn_state = current_nn_state


def CreateDemonstrations(env, num_episodes, max_episode_steps, demo_eval_type, params, **kwargs):
  evaluators = {**default_training_evaluators() ,"demonstrations" : {"type" : demo_eval_type, "params" : params}}
  terminal_when = default_terminal_criteria(max_episode_steps)
  benchmark = BenchmarkRunnerMP(evaluators=evaluators,
                                terminal_when=terminal_when,
                                behaviors=behaviors_tested,
                                log_eval_avg_every=10,
                                num_cpus=4,
                                checkpoint_dir="checkpoints2/",
                                merge_existing=False)


  
   

      