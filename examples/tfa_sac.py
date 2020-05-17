# Copyright (c) 2020 Patrick Hart, Julian Bernhard,
# Klemens Esterle, Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# TensorFlow Agents (https://github.com/tensorflow/agents) example

# BARK imports
from bark_project.modules.runtime.commons.parameters import ParameterServer

# BARK-ML imports
from bark_ml.environments.blueprints import ContinuousHighwayBlueprint
from bark_ml.environments.single_agent_runtime import SingleAgentRuntime
from bark_ml.library_wrappers.lib_tf_agents.agents.sac_agent import BehaviorSACAgent
from bark_ml.library_wrappers.lib_tf_agents.runners.sac_runner import SACRunner


params = ParameterServer()
bp = ContinuousHighwayBlueprint(params,
                                number_of_senarios=10,
                                random_seed=0)

env = SingleAgentRuntime(blueprint=bp,
                         render=True)
sac_agent = BehaviorSACAgent(environment=env,
                             params=params)

# set sac-agent in environment
env.ml_behavior = sac_agent
runner = SACRunner(params=params,
                   environment=env,
                   agent=sac_agent)

# runner.Train()
# runner.Visualize()
# TODO(@hart): dump (formatted) parameters