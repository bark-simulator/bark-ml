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
    params["ML"]["GraphObserver"]["AgentLimit"] = 4
    params["ML"]["SACRunner"]["NumberOfCollections"] = 200
    params["ML"]["BehaviorGraphSACAgent"]["CriticJointFcLayerParams"] = [256, 128]
    params["ML"]["BehaviorGraphSACAgent"]["ActorFcLayerParams"] = [256, 128]
    params["ML"]["BehaviorGraphSACAgent"]["GNN"]["NumMpLayers"] = 2
    params["ML"]["BehaviorGraphSACAgent"]["GNN"]["MpLayerNumUnits"] = 256
    params["ML"]["BehaviorGraphSACAgent"]["GNN"]["library"] = "spektral" # "tf2_gnn" or "spektral"
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

  def test_gnn_actor(self):
    agent, runner, _ = self._mock_setup()
    runner.Train()

  def test_gnn_parameters(self):
    params = ParameterServer()
    params["ML"]["BehaviorGraphSACAgent"]["GNN"]["NumMpLayers"] = 4
    params["ML"]["BehaviorGraphSACAgent"]["GNN"]["MpLayerNumUnits"] = 64
    params["ML"]["BehaviorGraphSACAgent"]["GNN"]["message_calculation_class"] = "gnn_edge_mlp"
    params["ML"]["BehaviorGraphSACAgent"]["GNN"]["global_exchange_mode"] = "mean"
    
    bp = ContinuousHighwayBlueprint(params, number_of_senarios=2500, random_seed=0)
    observer = GraphObserver(params=params)
    env = SingleAgentRuntime(blueprint=bp, observer=observer, render=False)
    sac_agent = BehaviorGraphSACAgent(environment=env, params=params)

    actor_gnn = sac_agent._agent._actor_network._gnn._gnn
    critic_gnn = sac_agent._agent._actor_network._gnn._gnn

    for gnn in [actor_gnn, critic_gnn]:
      self.assertEqual(gnn._params["num_layers"], 4)
      self.assertEqual(gnn._params["hidden_dim"], 64)
      self.assertEqual(gnn._params["message_calculation_class"], "gnn_edge_mlp")
      self.assertEqual(gnn._params["global_exchange_mode"], "mean")

if __name__ == '__main__':
  unittest.main()
