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
    params["ML"]["TFARunner"]["SummaryPath"] = '/Users/marco.oliva/Development/bark-ml_logs/summaries/tf2_gnn/'
    params["ML"]["GoalReachedEvaluator"]["MaxSteps"] = 30
    params["ML"]["BehaviorSACAgent"]["DebugSummaries"] = True
    params["ML"]["SACRunner"]["EvaluateEveryNSteps"] = 100
    params["ML"]["BehaviorSACAgent"]["BatchSize"] = 128
    params["ML"]["GraphObserver"]["AgentLimit"] = 4
    params["ML"]["BehaviorGraphSACAgent"]["CriticJointFcLayerParams"] = [256, 128]
    params["ML"]["BehaviorGraphSACAgent"]["ActorFcLayerParams"] = [256, 128]
    params["ML"]["BehaviorGraphSACAgent"]["GNN"]["NumMpLayers"] = 2
    params["ML"]["BehaviorGraphSACAgent"]["GNN"]["MpLayerNumUnits"] = 256
    params["ML"]["BehaviorGraphSACAgent"]["GNN"]["library"] = "tf2_gnn" # "tf2_gnn" or "spektral"
    params["ML"]["SACRunner"]["NumberOfCollections"] = int(1e6)
    params["ML"]["BehaviorGraphSACAgent"]["GNN"]["GraphDimensions"] = (4, 11, 4) # (n_nodes, n_features, n_edge_features)

    # tf2_gnn
    # NOTE: when using the ggnn mp class, MPLayerUnits must match n_features!
    params["ML"]["BehaviorGraphSACAgent"]["GNN"]["message_calculation_class"] = "gnn_edge_mlp"
    params["ML"]["BehaviorGraphSACAgent"]["GNN"]["global_exchange_mode"] = "gru"
    params["ML"]["BehaviorGraphSACAgent"]["GNN"]["dense_every_num_layers"] = 1
    params["ML"]["BehaviorGraphSACAgent"]["GNN"]["global_exchange_every_num_layers"] = 1

    # only considered when "message_calculation_class" = "gnn_edge_mlp"
    params["ML"]["BehaviorGraphSACAgent"]["GNN"]["num_edge_MLP_hidden_layers"] = 2 

    bp = ContinuousHighwayBlueprint(params,
                                    number_of_senarios=1,
                                    random_seed=0)

    observer = GraphObserver(params=params)
    env = SingleAgentRuntime(blueprint=bp, observer=observer, render=False)
    agent = BehaviorGraphSACAgent(environment=env, params=params)
    env.ml_behavior = agent
    runner = SACRunner(params=params, environment=env, agent=agent)
    return agent, runner, observer
  
  def test_that_all_variables_are_training(self):
    """
    Verifies that all variables change during training.
    Trains two iterations, captures the value of the variables
    after both iterations and compares them to make sure that
    in each layer, at least one value changes.
    """
    agent, runner, observer = self._mock_setup()

    iterator = iter(agent._dataset)
    trainable_variables = []

    agent._training = True
    runner._collection_driver.run()
    experience, _ = next(iterator)
    agent._agent.train(experience)

    v = agent._agent.trainable_variables
    shapes = [np.product(p.shape.as_list()) for p in v]
    
    v = agent._agent._actor_network._gnn.trainable_variables
    gnn_shapes = [np.product(p.shape.as_list()) for p in v]
    v = agent._agent._actor_network.trainable_variables
    actor_shapes = [np.product(p.shape.as_list()) for p in v]
    v = agent._agent._critic_network_1.trainable_variables
    critic_shapes = [np.product(p.shape.as_list()) for p in v]

    vals = [(val.name, np.copy(val.numpy())) for val in v]
    trainable_variables.append(vals)

    networks = [
        agent._agent._actor_network,
        agent._agent._critic_network_1,
        agent._agent._critic_network_2,
        agent._agent._target_critic_network_1,
        agent._agent._target_critic_network_2,
    ]

    for net in networks:
        n = np.sum([np.prod(v.get_shape().as_list()) for v in net.trainable_variables])
        print(f"{net.name}:")
        print(f'Parameters: {n}')
        
#analyse_agent(agent)

    print(f'critic params: {np.sum(critic_shapes)}')
    print(f'actor params: {np.sum(actor_shapes)}')
    print(f'actor gnn params: {np.sum(gnn_shapes)}')
    print(f'Total parameters: {np.sum(shapes)}')
    # # return  

    # before = trainable_variables[0]
    # after = trainable_variables[-1]

    # constant_vars = []
    # trained_vars = []
    # for b, a in zip(before, after):
    #   if (b[1] == a[1]).all():
    #     constant_vars.append(b)
    #   else:
    #     trained_vars.append(b)
    
    # print(f'trainable vars: {len(trained_vars)}')
    # for var in trained_vars:
    #   print(var[1].shape)


  # def test_gnn_parameters(self):
  #   params = ParameterServer()
  #   params["ML"]["BehaviorGraphSACAgent"]["GNN"]["NumLayers"] = 2
  #   params["ML"]["BehaviorGraphSACAgent"]["GNN"]["FcLayerParams"] = 11
  #   params["ML"]["BehaviorGraphSACAgent"]["GNN"]["message_calculation_class"] = "gnn_edge_mlp"
  #   params["ML"]["BehaviorGraphSACAgent"]["GNN"]["global_exchange_mode"] = "mean"
    
  #   bp = ContinuousHighwayBlueprint(params, number_of_senarios=2500, random_seed=0)
  #   observer = GraphObserver(params=params)
  #   env = SingleAgentRuntime(blueprint=bp, observer=observer, render=False)
  #   sac_agent = BehaviorGraphSACAgent(environment=env, params=params)

  #   actor_gnn = sac_agent._agent._actor_network._gnn._gnn
  #   critic_gnn = sac_agent._agent._actor_network._gnn._gnn

  #   for gnn in [actor_gnn, critic_gnn]:
  #     self.assertEqual(gnn._params["num_layers"], 2)
  #     self.assertEqual(gnn._params["hidden_dim"], 11)
  #     self.assertEqual(gnn._params["message_calculation_class"], "gnn_edge_mlp")
  #     self.assertEqual(gnn._params["global_exchange_mode"], "mean")

  # def test_execution_time(self):
  #   def _print_stats(call_times, iterations, name):
  #     call_times = np.array(call_times) / iterations
  #     calls = len(call_times)
  #     call_duration = np.mean(call_times)
  #     total = np.sum(call_times)
      
  #     val_str = f'{total:.4f} s ({calls} x {call_duration:.4f} s)'
  #     print('{:.<20} {}'.format(name, val_str))

  #   agent, runner, observer = self._mock_setup()

  #   t = []
  #   iterations = 2
  #   iterator = iter(agent._dataset)

  #   for i in range(iterations):
  #     agent._training = True
  #     runner._collection_driver.run()
  #     experience, _ = next(iterator)
      
  #     t0 = time.time()
  #     agent._agent.train(experience)
  #     t.append(time.time() - t0)

  #   execution_time = np.mean(t)
  #   print(f'\n###########\n')

  #   _print_stats(observer.observe_times, iterations, "Observe")
  #   _print_stats(agent._agent._actor_network.call_times, iterations, "Actor")
  #   _print_stats(agent._agent._actor_network.gnn_call_times, iterations, "  GNN")
  #   _print_stats(agent._agent._actor_network._gnn.graph_conversion_times, iterations, "    Graph Conversion")
  #   _print_stats(agent._agent._actor_network._gnn.gnn_call_times, iterations, "    tf2_gnn")
    
  #   critics = [
  #     ("Critic 1", agent._agent._critic_network_1),
  #     ("Critic 2", agent._agent._critic_network_2),
  #     ("Target Critic 1", agent._agent._target_critic_network_1),
  #     ("Target Critic 2", agent._agent._target_critic_network_2),
  #   ]

  #   for name, critic in critics:
  #       _print_stats(critic.call_times, iterations, name)
  #       _print_stats(critic.gnn_call_times, iterations, '  GNN')
  #       _print_stats(critic.encoder_call_times, iterations, '  Encoder')
  #       _print_stats(critic.joint_call_times, iterations, '  Joint')

  #   print(f'----------------------------------------------------------')
  #   print(f'Total execution time per train cycle: {execution_time:.2f} s')
    
  #   print(f'\n###########\n')


if __name__ == '__main__':
  unittest.main()
