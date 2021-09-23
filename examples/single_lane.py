import gym
import numpy as np
# registers bark-ml environments
import bark_ml.environments.gym  # pylint: disable=unused-import
# import agent and runner
from bark_ml.library_wrappers.lib_tf_agents.agents import BehaviorSACAgent
from bark_ml.library_wrappers.lib_tf_agents.runners import SACRunner
# to handle parameters
from bark.runtime.commons.parameters import ParameterServer

# load gym environment
env = gym.make("singlelane-v0")

# params = ParameterServer(filename="tfa_params.json")
params = ParameterServer()
sac_agent = BehaviorSACAgent(environment=env,
                             params=params)
env.ml_behavior = sac_agent
# runner either trains, evaluates or visualized the agent
runner = SACRunner(params=params,
                   environment=env,
                   agent=sac_agent)

runner.Train()
#runner.Run(num_episodes=1, render=True)


for cr in np.arange(0, 1, 0.1):
    runner._environment._max_col_rate = cr
    runner.Run(num_episodes=250, render=False, max_col_rate=cr)
#    runner._environment._tracer.Save(
#      params["ML"]["ResultsFolder"] + "evaluation_results_runtime.pckl")
#    goal_reached = runner._tracer.success_rate
#    runner._tracer.Save(
#      params["ML"]["ResultsFolder"] + "evaluation_results_runner.pckl")
