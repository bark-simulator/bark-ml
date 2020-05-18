# Copyright (c) 2020 Patrick Hart, Julian Bernhard,
# Klemens Esterle, Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# TensorFlow Agents (https://github.com/tensorflow/agents) example
import gym

# BARK imports
from bark_project.modules.runtime.commons.parameters import ParameterServer

# BARK-ML imports
from bark_ml.environments.blueprints import ContinuousHighwayBlueprint
from bark_ml.environments.single_agent_runtime import SingleAgentRuntime
from bark_ml.library_wrappers.lib_tf_agents.agents import BehaviorSACAgent, BehaviorPPOAgent
from bark_ml.library_wrappers.lib_tf_agents.runners import SACRunner, PPORunner


# params = ParameterServer(filename="/Users/hart/2020/bark-ml/examples/tfa_params.json")
params = ParameterServer()
params["ML"]["BehaviorTFAAgents"]["CheckpointPath"] = "/home/hart/Dokumente/2020/bark-ml/checkpoints/"
params["ML"]["TFARunner"]["SummaryPath"] = "/home/hart/Dokumente/2020/bark-ml/checkpoints/"


bp = ContinuousHighwayBlueprint(params,
                                number_of_senarios=10,
                                random_seed=0)

env = SingleAgentRuntime(blueprint=bp,
                         render=False)

# ppo
# ppo_agent = BehaviorPPOAgent(environment=env,
#                              params=params)
# env.ml_behavior = ppo_agent
# runner = PPORunner(params=params,
#                    environment=env,
#                    agent=ppo_agent)

# does not work either
env = gym.make("Pendulum-v0")

# sac
sac_agent = BehaviorSACAgent(environment=env,
                             params=params)
env.ml_behavior = sac_agent
runner = SACRunner(params=params,
                   environment=env,
                   agent=sac_agent)


runner.Train()
runner.Visualize(5)
# params.save("/Users/hart/2020/bark-ml/examples/tfa_params.json")