import unittest
import os
import time
import tensorflow as tf
import numpy as np
import spektral
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense

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
    
  #   print(f'trainable vars: {len(trained_vars)}')
  #   for var in trained_vars:
  #     print(var[0])


  def test_execution_time(self):
    def _print_stats(call_times, iterations, name):
      call_times = np.array(call_times) / iterations
      calls = len(call_times)
      call_duration = np.mean(call_times)
      total = np.sum(call_times)
      
      val_str = f'{total:.4f} s ({calls} x {call_duration:.4f} s)'
      print('{:.<20} {}'.format(name, val_str))

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
    _print_stats(agent._agent._actor_network.gnn_call_times, iterations, "  GNNWrapper")
    _print_stats(agent._agent._actor_network._gnn.graph_conversion_times, iterations, "    Data Prep")
    _print_stats(agent._agent._actor_network._gnn.gnn_call_times, iterations, "    GNN lib")
    
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

  def graph(self, sparse_links=True):
    node_limit = 5
    num_features = 5
    num_nodes = 5

    # 4 agents + 1 fill-up slot (node_limit = 5)
    agents = [
      np.full(num_features, 0.1),
      np.full(num_features, 0.2),
      np.full(num_features, 0.3),
      np.full(num_features, 0.4),
      np.full(num_features, 0.5)
    ]

    # the empty slot should not be returned
    expected_nodes = agents[:-1]

    # links are already bidirectional, so there
    # are only zeros to the left of the diagonal
    adjacency_list = [
      [0, 1, 1, 1, 0], # 1 connects with 2, 3, 4
      [0, 0, 1, 1, 0], # 2 connects with 3, 4
      [0, 0, 0, 1, 0], # 3 connects with 4
      [0, 0, 0, 0, 0], # 4 has no links
      [0, 0, 0, 0, 0]  # empty slot -> all zeros
    ]

    # the edges encoded in the adjacency list above
    expected_edges = [
      [0, 1], [0, 2], [0, 3],
      [1, 2], [1, 3],
      [2, 3]
    ]

    observation = np.array([node_limit, num_nodes, num_features])
    observation = np.append(observation, agents)
    observation = np.append(observation, adjacency_list)
    observation = observation.reshape(-1)
    
    self.assertEqual(observation.shape, (53,))

    nodes, adj = GraphObserver.graph(observation, sparse_links=True)
    return nodes, adj
    

  def test_spektral(self):
    from spektral.layers import EdgeConditionedConv, GlobalAttnSumPool, GraphAttention
    from spektral.utils import numpy_to_batch
    from tensorflow.keras.layers import Flatten, Dense

    # Parameters
    N = 5       # Number of nodes in the graphs
    F = 5       # Dimension of node features
    S = 0       # Dimension of edge features
    n_out = 128   # Dimension of the target
    
    nodes, adj = self.graph(sparse_links=True)
    X1 = np.array(nodes)
    A1 = np.array(adj)

    nodes, adj = self.graph(sparse_links=True)
    X2 = np.array(nodes)
    A2 = np.array(adj)
    
    E1 = np.random.rand(N, N, S).astype('float32')
    E2 = np.random.rand(N, N, S).astype('float32')

    X = [X1, X2]
    A = [A1, A2]
    E = np.ones(shape=(1, 0, 0, 0))

    X, A, E = numpy_to_batch(X, A, E)

    X = tf.convert_to_tensor(X)
    A = tf.convert_to_tensor(A)
    E = tf.convert_to_tensor(E)
    
    # L1 = EdgeConditionedConv(64, kernel_network=[16, 16], activation='tanh')
    # L2 = EdgeConditionedConv(64, kernel_network=[16, 16], activation='tanh')
    # L3 = EdgeConditionedConv(16, kernel_network=[16, 16], activation='tanh')
    # L4 = GlobalAttnSumPool()
    # L5 = Dense(n_out, activation='tanh')

    X = GraphAttention(16)([X, A, E])
    X = GraphAttention(16)([X, A, E])
    X = GlobalAttnSumPool()(X)
    X = Dense(128, activation='tanh')(X)

    # X = L1([X, A, E])
    # X = L2([X, A, E])
    # X = L3([X, A, E])
    # X = L4(X)
    # X = L5(X)
    
    print(X.shape)


      

if __name__ == '__main__':
  unittest.main()
