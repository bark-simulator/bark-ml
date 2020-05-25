# Copyright (c) 2019 The diadem authors 
# 
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# ========================================================


"""
Run the experiment

Important sidemark: the Agent is defined in the parameters, not in the main file!
"""

try:
  import debug_settings
except:
  pass

import tensorflow as tf
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
import sys
import os

from diadem.agents import AgentContext, AgentManager
from diadem.experiment import Experiment
from diadem.experiment.visualizers import OnlineVisualizer
from diadem.summary import PandasSummary, ConsoleSummary
from diadem.common import Params
from diadem.preprocessors import Normalization

from bark_ml.library_wrappers.lib_diadem.diadem_bark_environment import DiademBarkEnvironment

from bark_project.modules.runtime.commons.parameters import ParameterServer
from bark_ml.environments.single_agent_runtime import SingleAgentRuntime
from bark_ml.observers.nearest_state_observer import NearestAgentsObserver
from bark_ml.environments.blueprints import DiscreteHighwayBlueprint

# create scenario
bark_params = ParameterServer()
bp = DiscreteHighwayBlueprint(bark_params,
                              number_of_senarios=10,
                              random_seed=0,
                              viewer=False)

observer = NearestAgentsObserver(bark_params)
runtime = SingleAgentRuntime(blueprint=bp,
                             observer=observer,
                             render=False)


def run_dqn_algorithm(parameter_files):
  exp_dir = "tmp_exp_dir"
  diadem_params = Params(filename=parameter_files)
  environment = DiademBarkEnvironment(runtime=runtime)
  context = AgentContext(
    environment=environment,
    datamanager=None,
    preprocessor=None,
    optimizer=tf.train.AdamOptimizer,
    summary_service=ConsoleSummary()
  )
  agent = AgentManager(
    params=diadem_params,
    context=context
  )
  exp = Experiment(
    params=diadem_params['experiment'],
    main_dir=exp_dir,
    context=context,
    agent=agent,
    visualizer=None)
  exp.run()


# replace second parameter file with other defaults to get categorical, standard dqn or quantile agents
if __name__ == '__main__':
  # basic Double DQN with Prioritized Experience Replay
  # run_dqn_algorithm(parameter_files=["examples/example_params/common_parameters.yaml",
  #                                     "examples/example_params/dqn_distributional_quantile.yaml"])
  run_dqn_algorithm(parameter_files=["examples/example_params/dqn_basic.yaml"])