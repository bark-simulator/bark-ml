# test to make sure that the behavior models of bark-ml are runnable with the benchmark runner

# Copyright (c) 2020 fortiss GmbH
#
# Authors: Julian Bernhard, Klemens Esterle, Patrick Hart and
# Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import unittest
import os
import time
import ray

try:
    import debug_settings
except:
    pass

from load.benchmark_database import BenchmarkDatabase
from serialization.database_serializer import DatabaseSerializer
from bark.benchmark.benchmark_runner import BenchmarkRunner, BenchmarkConfig
from bark.benchmark.benchmark_runner_mp import BenchmarkRunnerMP, _BenchmarkRunnerActor, \
  deserialize_benchmark_config, serialize_benchmark_config

from bark.runtime.viewer.matplotlib_viewer import MPViewer

from bark.core.world.evaluation import *
from bark.core.world.evaluation.ltl import ConstantLabelFunction
from bark.runtime.commons.parameters import ParameterServer
from bark.core.models.behavior import BehaviorIDMClassic, BehaviorConstantAcceleration

# bark-ml
from bark_ml.library_wrappers.lib_tf_agents.agents import BehaviorGraphSACAgent
from bark_ml.observers.graph_observer import GraphObserver

try: # bazel run
  os.chdir("../benchmark_database/")
except: # debug
  os.chdir("bazel-bin/bark/benchmark/tests/py_benchmark_runner_tests.runfiles/benchmark_database")

class DatabaseRunnerTests(unittest.TestCase):
  def test_database_runner(self):
    dbs = DatabaseSerializer(test_scenarios=4, test_world_steps=5, num_serialize_scenarios=2)
    dbs.process("data/database1")
    local_release_filename = dbs.release(version="test")

    db = BenchmarkDatabase(database_root=local_release_filename)
    evaluators = {
      "success" : "EvaluatorGoalReached",
      "collision" : "EvaluatorCollisionEgoAgent",
      "max_steps": "EvaluatorStepCount"
    }
    terminal_when = {
      "collision" :lambda x: x,
      "max_steps": lambda x : x>2
    }
    params = ParameterServer() # only for evaluated agents not passed to scenario!
    
    # NOTE: pass ml behavior
    observer = GraphObserver(params=params)
    sac_behavior = BehaviorGraphSACAgent(
      observer=observer,
      params=params)
    
    behaviors_tested = {"IDM": BehaviorIDMClassic(params),
                        "Const" : BehaviorConstantAcceleration(params)}
                                    

    benchmark_runner = BenchmarkRunner(benchmark_database=db,
                                       evaluators=evaluators,
                                       terminal_when=terminal_when,
                                       behaviors=behaviors_tested,
                                       log_eval_avg_every=5)

    result = benchmark_runner.run()
    df = result.get_data_frame()
    print(df)
    self.assertEqual(len(df.index), 2*2*2) # 2 Behaviors * 2 Serialize Scenarios * 1 scenario sets

    groups = result.get_evaluation_groups()
    self.assertEqual(set(groups), set(["behavior", "scen_set"]))


if __name__ == '__main__':
  unittest.main()
