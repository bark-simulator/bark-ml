import unittest
import numpy as np
import os
import gym
import matplotlib
import time

# BARK imports
from bark_project.modules.runtime.commons.parameters import ParameterServer

# BARK-ML imports
from bark_ml.environments.blueprints import ContinuousHighwayBlueprint, \
  DiscreteHighwayBlueprint, ContinuousMergingBlueprint, DiscreteMergingBlueprint
from bark_ml.environments.single_agent_runtime import SingleAgentRuntime
import bark_ml.environments.gym
from bark_ml.library_wrappers.lib_tf2rl.agents import BehaviorGAILAgent
from bark_ml.library_wrappers.lib_tf2rl.runners import GAILRunner


class PyLibraryWrappersTF2RLTests(unittest.TestCase):
  # make sure the agent works
  def test_agent_wrapping(self):
    params = ParameterServer()
    env = gym.make("Pendulum-v0") 
    env.reset()
    agent = BehaviorGAILAgent(environment=env,
                     params=params)

  # assign as behavior model (to check if trained agent can be used)
  def test_behavior_wrapping(self):
    # create scenario
    params = ParameterServer()
    bp = ContinuousHighwayBlueprint(params,
                                    number_of_senarios=10,
                                    random_seed=0)
    env = SingleAgentRuntime(blueprint=bp,
                             render=True)
    ml_behaviors = []
    ml_behaviors.append(
      BehaviorGAILAgent(environment=env,
                       params=params))
    
    for ml_behavior in ml_behaviors:
      # set agent
      env.ml_behavior = ml_behavior
      env.reset()
      done = False
      while done is False:
        action = np.random.uniform(low=-0.1, high=0.1, size=(2, ))
        observed_next_state, reward, done, info = env.step(action)
        print(f"Observed state: {observed_next_state}, Reward: {reward}, Done: {done}")

  # agent + runner
  def test_agent_and_runner(self):
    params = ParameterServer()
    bp = ContinuousHighwayBlueprint(params,
                                    number_of_senarios=10,
                                    random_seed=0)
    env = SingleAgentRuntime(blueprint=bp,
                             render=False)
    agent = BehaviorGAILAgent(environment=env,
                     params=params)
    
    # set agent
    env.ml_behavior = agent
    runner = GAILRunner(params=params,
                       environment=env,
                       agent=agent)
    # runner.Train()
    runner.Visualize()

class PyLibraryWrappersTF2RLUtilsTests(unittest.TestCase):
  """
  Tests for the tf2rl utils
  """

  def test_load_pkl_file(self):
    """
    Test: Load the pkl file containing the expert trajectories.
    """
    filename = os.path.join(os.path.dirname(__file__), 'data', 'mock_expert_trajectories.pkl')
    print(filename)
    


if __name__ == '__main__':
  unittest.main()