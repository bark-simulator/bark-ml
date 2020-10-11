# Copyright (c) 2020 fortiss GmbH
#
# Authors: Patrick Hart, Julian Bernhard, Klemens Esterle, and
# Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


import unittest
import numpy as np
import os
import importlib
import matplotlib
import time
import tensorflow as tf
from graph_nets import utils_tf
from graph_nets.graphs import GraphsTuple
from bark.runtime.commons.parameters import ParameterServer
from bark_ml.environments.blueprints.configurable.configurable_scenario import ConfigurableScenarioBlueprint
from bark_ml.evaluators.goal_reached import GoalReached
from bark_ml.environments.single_agent_runtime import SingleAgentRuntime
from bark_ml.library_wrappers.lib_tf_agents.runners.sac_runner import SACRunner
from bark_ml.library_wrappers.lib_tf_agents.agents.graph_sac_agent import BehaviorGraphSACAgent
from bark_ml.observers.graph_observer import GraphObserver
from bark_ml.behaviors.cont_behavior import BehaviorContinuousML
from experiments.experiment import Experiment


class PyExperimentTests(unittest.TestCase):
  @unittest.skip("...")
  def test_module_creation(self):
    params = ParameterServer(filename="experiments/example_experiment/config.json")
    # Blueprint
    ml_behavior = BehaviorContinuousML(params=params)
    module_name = "ConfigurableScenarioBlueprint"
    further_items = {"num_scenarios": 3500, "viewer": False, "ml_behavior": ml_behavior}
    blueprint = eval("{}(params=params, **further_items)".format(module_name))
    # NOTE: set dummy behavior model for continuous or discerte actions
    
    # Evaluator
    module_name = "GoalReached"
    evaluator = eval("{}(params=params)".format(module_name))
    
    # Observer
    module_name = "GraphObserver"
    observer = eval("{}(params=params)".format(module_name))
    
    # Runtime
    module_name = "SingleAgentRuntime"
    runtime = eval("{}(blueprint=blueprint, evaluator=evaluator, \
                       observer=observer, render=False)".format(module_name))
    
    # SAC-Agent
    module_name = "BehaviorGraphSACAgent"
    further_items = {"init_gnn": 'init_gcn'}
    agent = eval("{}(environment=runtime, observer=observer, \
                     params=params, **further_items)".format(module_name))
    runtime.ml_behavior = agent
    
    # SAC-Runner
    module_name = "SACRunner"
    runner = eval("{}(params=params, environment=runtime, \
                      agent=agent)".format(module_name))

  @unittest.skip("...")
  def test_module_creation_from_json(self):
    params = ParameterServer(filename="experiments/example_experiment/config.json")
    exp_params = params["Experiment"]
    
    def LoadModule(module_name, dict_items):
      return eval("{}(**dict_items)".format(module_name))
      
    # blueprint
    module_name = exp_params["Blueprint"]["ModuleName"]
    items = exp_params["Blueprint"]["Config"].ConvertToDict()
    items["params"] = params
    blueprint = LoadModule(module_name, items)
    blueprint._ml_behavior = BehaviorContinuousML(params=params)
    
    # Evaluator
    module_name = exp_params["Evaluator"]["ModuleName"]
    items = exp_params["Evaluator"]["Config"].ConvertToDict()
    items["params"] = params
    evaluator = LoadModule(module_name, items)
    
    # Observer
    module_name = exp_params["Observer"]["ModuleName"]
    items = exp_params["Observer"]["Config"].ConvertToDict()
    items["params"] = params
    observer = LoadModule(module_name, items)
    
    # Runtime
    module_name = exp_params["Runtime"]["ModuleName"]
    items = exp_params["Runtime"]["Config"].ConvertToDict()
    items["evaluator"] = evaluator
    items["observer"] = observer
    items["blueprint"] = blueprint
    runtime = LoadModule(module_name, items)
    
    # SAC-Agent
    module_name = exp_params["Agent"]["ModuleName"]
    items = exp_params["Agent"]["Config"].ConvertToDict()
    items["environment"] = runtime
    items["observer"] = observer
    items["params"] = params
    agent = LoadModule(module_name, items)
    runtime.ml_behavior = agent
    
    # SAC-Runner
    module_name = exp_params["Runner"]["ModuleName"]
    items = exp_params["Runner"]["Config"].ConvertToDict()
    items["environment"] = runtime
    items["params"] = params
    items["agent"] = agent
    runner = LoadModule(module_name, items)

  def test_experiment_class(self):
    experiment = Experiment("experiments/example_experiment/config.json")
    experiment.runner.Run(num_episodes=5, render=True)
  

if __name__ == '__main__':
  unittest.main()