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
import pickle

try:
    import debug_settings
except:
    pass


from bark.runtime.viewer.matplotlib_viewer import MPViewer

from bark.core.world.evaluation import *
from bark.core.world.evaluation.ltl import ConstantLabelFunction
from bark.runtime.commons.parameters import ParameterServer
from bark.core.models.behavior import BehaviorIDMClassic, BehaviorConstantAcceleration

# bark-ml
from bark_ml.library_wrappers.lib_tf_agents.agents import BehaviorGraphSACAgent
from bark_ml.observers.graph_observer import GraphObserver
from bark_ml.environments.single_agent_runtime import SingleAgentRuntime
from bark_ml.environments.blueprints import ContinuousHighwayBlueprint, ContinuousMergingBlueprint


try: # bazel run
  os.chdir("../benchmark_database/")
except: # debug
  os.chdir("bazel-bin/bark/benchmark/tests/py_benchmark_runner_tests.runfiles/benchmark_database")


class PyPicklingTests(unittest.TestCase):
  def test_pickling(self):

    params = ParameterServer() # only for evaluated agents not passed to scenario!
    observer = GraphObserver(params=params)
    pickle.dumps(observer)
    
    bp = ContinuousMergingBlueprint(
      params,
      number_of_senarios=1,
      random_seed=0)
    # pickle.dumps(bp)
    
    env = SingleAgentRuntime(
      blueprint=bp,
      observer=observer,
      render=False)
    # pickle.dumps(env)
    
    sac_behavior = BehaviorGraphSACAgent(
      observer=observer,
      environment=env,
      params=params)
    pickle.dumps(sac_behavior)
    
    # behaviors_tested = {"IDM": BehaviorIDMClassic(params),
    #                     "Const" : BehaviorConstantAcceleration(params),
    #                     "GraphSAC": sac_behavior}
                                    

    # benchmark_runner = BenchmarkRunner(benchmark_database=db,
    #                                    evaluators=evaluators,
    #                                    terminal_when=terminal_when,
    #                                    behaviors=behaviors_tested,
    #                                    log_eval_avg_every=5)

    # result = benchmark_runner.run()
    # df = result.get_data_frame()
    # print(df)
    # self.assertEqual(len(df.index), 2*2*2) # 2 Behaviors * 2 Serialize Scenarios * 1 scenario sets
    # groups = result.get_evaluation_groups()
    # self.assertEqual(set(groups), set(["behavior", "scen_set"]))


if __name__ == '__main__':
  unittest.main()
