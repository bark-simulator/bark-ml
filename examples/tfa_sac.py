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
from bark_ml.library_wrappers.lib_tf_agents.agents.ppo_agent import BehaviorPPOAgent
from bark_ml.library_wrappers.lib_tf_agents.runners.ppo_runner import PPORunner

params = ParameterServer(filename="/Users/hart/2020/bark-ml/examples/sac_params.json")
# params = ParameterServer()

bp = ContinuousHighwayBlueprint(params,
                                number_of_senarios=10,
                                random_seed=0)

env = SingleAgentRuntime(blueprint=bp,
                         render=False)
ppo_agent = BehaviorPPOAgent(environment=env,
                             params=params)

# ppo
env.ml_behavior = ppo_agent
runner = PPORunner(params=params,
                   environment=env,
                   agent=ppo_agent)

# runner.Train()
runner.Visualize()
params.save("/Users/hart/2020/bark-ml/examples/sac_params.json")