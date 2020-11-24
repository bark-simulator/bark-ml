# Copyright (c) 2020 Julian Bernhard,
# Klemens Esterle, Patrick Hart, Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from bark.benchmark.benchmark_result import BenchmarkConfig
from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.agent import TrainingBenchmark
from bark.benchmark.benchmark_runner import BenchmarkRunner, BehaviorConfig

def default_training_evaluators():
  default_config = {"success" : "EvaluatorGoalReached", "collision_other" : "EvaluatorCollisionEgoAgent",
       "out_of_drivable" : "EvaluatorDrivableArea", "max_steps": "EvaluatorStepCount"}
  return default_config

def default_terminal_criteria(max_episode_steps):
  terminal_when = {"collision_other" : lambda x: x, "out_of_drivable" : lambda x: x, \
        "max_steps": lambda x : x>max_episode_steps, "success" : lambda x: x}
  return terminal_when

class TrainingBenchmarkDatabase(TrainingBenchmark):
  def __init__(self, benchmark_database=None,
                     evaluators=None,
                     terminal_when=None):
    self.database = benchmark_database
    self.evaluators = evaluators
    self.terminal_when = terminal_when

  def reset(self, training_env, num_episodes, max_episode_steps, agent):
    super(TrainingBenchmarkDatabase, self).reset(training_env, num_episodes, \
                                       max_episode_steps, agent)
    evaluators = default_training_evaluators()
    if self.evaluators:
      evaluators = {**self.evaluators, **evaluators}
    terminal_when = default_terminal_criteria(max_episode_steps)
    if self.terminal_when:
      terminal_when = {**self.terminal_when, **terminal_when}
    self.benchmark_runner = BenchmarkRunner(
                                    benchmark_database=self.database, # this has priority over scenario generation
                                    scenario_generation=self.training_env._scenario_generator,
                                    evaluators=evaluators,
                                    terminal_when = terminal_when,
                                    num_scenarios=num_episodes,
                                    log_eval_avg_every = 100000000000,
                                    checkpoint_dir = "checkpoints",
                                    merge_existing = False,
                                    deepcopy=False)

  def run(self):
    mean_return, formatting = super(TrainingBenchmarkDatabase, self).run()
    eval_result = self.benchmark_runner.run()
    data_frame = eval_result.get_data_frame()
    data_frame["max_steps"] = data_frame.Terminal.apply(lambda x: "max_steps" in x and (not "collision" in x))
    data_frame["success"] = data_frame.Terminal.apply(lambda x: "success" in x and (not "collision" in x) and (not "max_steps" in x))
    data_frame = data_frame.drop(columns=["scen_set", "scen_idx", "behavior", "Terminal", "step", "config_idx"])
    mean = data_frame.mean(axis=0)
    eval_result = {**mean.to_dict(), **mean_return}
    return eval_result, f"Benchmark Result: {eval_result}"

  def is_better(self, eval_result1, than_eval_result2):
    pass
