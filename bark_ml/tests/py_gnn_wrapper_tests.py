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
    params["ML"]["BehaviorGraphSACAgent"]["GNN"]["num_layers"] = 3
    params["ML"]["BehaviorGraphSACAgent"]["GNN"]["hidden_dim"] = 64
    params["ML"]["BehaviorGraphSACAgent"]["GNN"]["global_exchange_every_num_layers"] = 2

    bp = ContinuousHighwayBlueprint(params,
                                    number_of_senarios=1,
                                    random_seed=0)

    observer = GraphObserver(params=params)
    env = SingleAgentRuntime(blueprint=bp, observer=observer, render=False)
    agent = BehaviorGraphSACAgent(environment=env, params=params)
    env.ml_behavior = agent
    runner = SACRunner(params=params, environment=env, agent=agent)
    return agent, runner, observer
  
  # def test_that_all_variables_are_training(self):
  #   """
  #   Verifies that all variables change during training.
  #   Trains two iterations, captures the value of the variables
  #   after both iterations and compares them to make sure that
  #   in each layer, at least one value changes.
  #   """
  #   agent, runner, observer = self._mock_setup()

  #   iterator = iter(agent._dataset)
  #   trainable_variables = []

  #   for i in range(4):
  #     agent._training = True
  #     runner._collection_driver.run()
  #     experience, _ = next(iterator)
  #     agent._agent.train(experience)

  #     vals = [(val.name, np.copy(val.numpy())) for val in agent._agent.trainable_variables]
  #     trainable_variables.append(vals)

  #   before = trainable_variables[0]
  #   after = trainable_variables[-1]

  #   constant_vars = []
  #   trained_vars = []
  #   for b, a in zip(before, after):
  #     if (b[1] == a[1]).all():
  #       constant_vars.append(b)
  #     else:
  #       trained_vars.append(b)

  def test_execution_time(self):
    def _print_stats(self, call_times, iterations, name):
      call_times = np.array(call_times) / iterations
      calls = len(call_times)
      call_duration = np.mean(call_times)
      total = np.sum(call_times)
      
      val_str = f'{total:.4f} s ({calls} x {call_duration:.4f} s)'
      print('{:.<40} {}'.format(name, val_str))

    agent, runner, observer = self._mock_setup()

    t = []
    iterations = 10
    iterator = iter(agent._dataset)

    for i in range(iterations):
      agent._training = True
      runner._collection_driver.run()
      experience, _ = next(iterator)
      
      t0 = time.time()
      agent._agent.train(experience)
      t.append(time.time() - t0)

    execution_time = np.mean(t)
    print(f'\n###########\n')

    _print_stats(observer.observe_times, iterations, "Observe")
    _print_stats(agent._agent._actor_network.call_times, iterations, "Actor")
    _print_stats(agent._agent._actor_network.gnn_call_times, iterations, "  GNN")
    _print_stats(agent._agent._actor_network._gnn.graph_conversion_times, iterations, "    Graph Conversion")
    _print_stats(observer.feature_times, iterations, "      Features")
    _print_stats(observer.edges_times, iterations, "      Edges")
    _print_stats(agent._agent._actor_network._gnn.gnn_call_times, iterations, "    tf2_gnn")
    
    critics = [
      ("Critic 1", agent._agent._critic_network_1),
      ("Critic 2", agent._agent._critic_network_2),
      ("Target Critic 1", agent._agent._target_critic_network_1),
      ("Target Critic 2", agent._agent._target_critic_network_2),
    ]

    for name, critic in critics:
      _print_stats(critic.call_times, iterations, name)
      _print_stats(critic.gnn_call_times, iterations, '  GNN')
      _print_stats(critic.encoder_call_times, iterations, '  Encoder')
      _print_stats(critic.joint_call_times, iterations, '  Joint')

    print(f'----------------------------------------------------------')
    print(f'Total execution time per train cycle: {execution_time:.2f} s')
    
    print(f'\n###########\n')

    self.assertLess(execution_time, 3.0)


if __name__ == '__main__':
  unittest.main()
