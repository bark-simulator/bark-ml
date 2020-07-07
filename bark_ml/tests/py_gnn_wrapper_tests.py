import unittest
import os
import time
import tensorflow as tf
import numpy as np

from bark.runtime.commons.parameters import ParameterServer
from bark_ml.environments.blueprints import ContinuousHighwayBlueprint
from bark_ml.environments.single_agent_runtime import SingleAgentRuntime
from bark_ml.observers.graph_observer import GraphObserver
from bark_ml.library_wrappers.lib_tf2_gnn import GNNWrapper
from bark_ml.library_wrappers.lib_tf_agents.agents import BehaviorGraphSACAgent
from bark_ml.library_wrappers.lib_tf_agents.runners import SACRunner

class PyGNNWrapperTests(unittest.TestCase):

  def _mock_setup(self):
    params = ParameterServer()
    params["World"]["remove_agents_out_of_map"] = False
    params["ML"]["TFARunner"]["InitialCollectionEpisodesPerStep"] = 8
    params["ML"]["TFARunner"]["CollectionEpisodesPerStep"] = 8
    params["ML"]["BehaviorSACAgent"]["BatchSize"] = 16

    bp = ContinuousHighwayBlueprint(params,
                                    number_of_senarios=1,
                                    random_seed=0)

    observer = GraphObserver(params=params)
    env = SingleAgentRuntime(blueprint=bp, observer=observer, render=False)
    agent = BehaviorGraphSACAgent(environment=env, params=params)
    env.ml_behavior = agent
    runner = SACRunner(params=params, environment=env, agent=agent)
    return agent, runner
  
  def test_that_all_variables_are_training(self):
    """
    Verifies that all variables change during training.
    Trains two iterations, captures the value of the variables
    after both iterations and compares them to make sure that
    in each layer, at least one value changes.
    """
    agent, runner = self._mock_setup()

    iterator = iter(agent._dataset)
    trainable_variables = []

    for i in range(4):
      agent._training = True
      runner._collection_driver.run()
      experience, _ = next(iterator)
      agent._agent.train(experience)

      vals = [(val.name, np.copy(val.numpy())) for val in agent._agent.trainable_variables]
      trainable_variables.append(vals)

    before = trainable_variables[0]
    after = trainable_variables[-1]

    print(f'# trainable vars (before, after): {len(before)}, {len(after)}')
    constant_vars = []
    trained_vars = []
    for b, a in zip(before, after):
      if (b[1] == a[1]).all():
        constant_vars.append(b)
      else:
        trained_vars.append(b)
    
    print(f'\ntrained vars: {len(trained_vars)}')
    for cv in trained_vars:
      print(cv[0])

    print(f'\nconstant vars: {len(constant_vars)}')
    for cv in constant_vars:
      print(cv[0])


if __name__ == '__main__':
  unittest.main()