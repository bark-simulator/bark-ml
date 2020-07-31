import logging
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import time
import networkx as nx
import tensorflow as tf
import datetime
from collections import OrderedDict
from matplotlib.patches import Ellipse
from tf2_gnn.layers import GNN, GNNInput

# Bark-ML imports
from bark_ml.observers.graph_observer import GraphObserver
from bark_ml.library_wrappers.lib_tf_agents.runners import SACRunner

# Supervised learning imports
from supervised_learning.actor_nets import ConstantActorNet, RandomActorNet, \
    get_GNN_SAC_actor_net, get_SAC_actor_net

def visualize_graph(data_point, ax, visible_distance, normalization_ref):
    # Transform to nx.Graph
    observation = data_point["graph"]
    graph = GraphObserver.graph_from_observation(observation)

    # Get node positions
    pos = dict()
    goal = dict()
    for i in graph.nodes:
        features = graph.nodes[i]
        pos[i] = [features["x"].numpy(), features["y"].numpy()]
        goal[i] = [features["goal_x"].numpy(), features["goal_y"].numpy()]
        
    # Draw ellipse for visibility range of ego agent
    width = 4*visible_distance/normalization_ref["dx"][1]
    height = 4*visible_distance/normalization_ref["dy"][1]
    ellipse = Ellipse(pos[0], width=width,height=height, facecolor='yellow', zorder=-1)#,**kwargs)
    ax.add_patch(ellipse)
    goal_ellipse = Ellipse(goal[0], width= 0.2, height=0.2, facecolor="green", zorder=-2)
    ax.add_patch(goal_ellipse)

    # Change color for ego agent
    node_colors = ["blue" for i in range(len(graph.nodes))]
    node_colors[0] = "red"
    return nx.draw(graph, pos = pos, with_labels=True, ax=ax, node_color=node_colors)


def get_default_gnn_params():
    return GNN.get_default_hyperparameters()

graph_dims = (5, 5, 4)

def get_sample_observations():
    num_nodes = 5
    num_features = 5
    num_edge_features = 4

    agents = np.random.random_sample((num_nodes, num_features))
    edge_features = np.random.random_sample((num_nodes, num_nodes, num_edge_features))

    # note that edges are bidirectional, the
    # the matrix is symmetric
    adjacency_list = [
      [0, 1, 1, 1, 0], # 1 connects with 2, 3, 4
      [1, 0, 1, 1, 0], # 2 connects with 3, 4
      [1, 1, 0, 1, 0], # 3 connects with 4
      [1, 1, 1, 0, 0], # 4 has no links
      [0, 0, 0, 0, 0]  # empty slot -> all zeros
    ]

    observation = np.array([num_nodes, num_nodes, num_features])
    observation = np.append(observation, agents)
    observation = np.append(observation, adjacency_list)
    observation = np.append(observation, edge_features)
    observation = observation.reshape(-1)
    
    return tf.cast(tf.stack([observation, observation]), tf.float32)

def prepare_agent(agent, params, env):
    env.ml_behavior = agent
    runner = SACRunner(params=params, environment=env, agent=agent)
    
    iterator = iter(agent._dataset)
    agent._training = True
    runner._collection_driver.run()
    experience, _ = next(iterator)
    agent._agent.train(experience)

def summarize_agent(agent):
    networks = [
        agent._agent._actor_network,
        agent._agent._critic_network_1,
        agent._agent._critic_network_2,
        agent._agent._target_critic_network_1,
        agent._agent._target_critic_network_2,
    ]
     
    print('{:<30} {}'.format("Network", "Parameters"))
    print("==========================================")
    total_params = 0
    for net in networks:
        n = np.sum([np.prod(v.get_shape().as_list()) for v in net.trainable_variables])
        total_params += n
        print('{:.<34} {:,}'.format(net.name, n).replace(',','.'))
    
    print("------------------------------------------")
    print('{:<32} {:,}'.format("Total parameters", total_params).replace(',','.'))

def print_parameter_overview():
    shared_params = []
    tf2_gnn_params = []
    spektral_params = []

    p = '["ML"]["GraphObserver"]["AgentLimit"] (int)'
    d = "Specifies the maximum number of agents that are included in an observation."
    v = 4
    shared_params.append((p, d, v))
    
    d = "specifies the fully connected layers (number and sizes) of the critic joint network ([int])"
    p = "['ML']['BehaviorGraphSACAgent']['CriticJointFcLayerParams']"
    v = [256, 128]
    shared_params.append((p, d, v))
    
    d = "specifies the fully connected layers (number and sizes) of the actor encoding network ([int])"
    p = "['ML']['BehaviorGraphSACAgent']['ActorFcLayerParams']"
    v = [256, 128]
    shared_params.append((p, d, v))
    
    d = "specifies the number of message passing layers in the GNN (int)"
    p = "['ML']['BehaviorGraphSACAgent']['GNN']['NumMpLayers']"
    v = 2
    shared_params.append((p, d, v))
    
    d = "the number of units in the message passing layers in the GNN (int)"
    p = "['ML']['BehaviorGraphSACAgent']['GNN']['MpLayerNumUnits']"
    v = 128
    shared_params.append((p, d, v))
    
    d = 'which library to use as the GNN implementation, either "tf2_gnn" or "spektral"'
    p = "['ML']['BehaviorGraphSACAgent']['GNN']['library']"
    v = "tf2_gnn"
    shared_params.append((p, d, v))

    # tf2_gnn specific parameters: these only apply when using tf2_gnn
    
    d = "the identifier of the message passing class to be used, here: a recurrent gated convolution network (str)\n# NOTE: when using the 'ggnn' message passing layer, 'MPLayerUnits' must match n_features!"
    p = "['ML']['BehaviorGraphSACAgent']['GNN.message_calculation_class']"
    v = "rgcn"
    tf2_gnn_params.append((p, d, v))
    
    d = "the identifier of the message passing class to be used, here: a gated recurrent unit (str)"
    p = "['ML']['BehaviorGraphSACAgent']['GNN']['global_exchange_mode']"
    v = "gru"
    tf2_gnn_params.append((p, d, v))

    d = "specifies after how many message passing layers a dense layer is inserted (int)"
    p = "['ML']['BehaviorGraphSACAgent']['GNN']['dense_every_num_layers']"
    v = 2
    tf2_gnn_params.append((p, d, v))
    
    d = "specifies after how many message passing layers a global exchange layer is inserted (int)"
    p = "['ML']['BehaviorGraphSACAgent']['GNN.global_exchange_every_num_layers']"
    v = 2
    tf2_gnn_params.append((p, d, v))

    # spektral specific parameters: these only apply when using spektral
    
    d = "the number of channels in the edge conditioned convolution layer (int)"
    p = "['ML']['BehaviorGraphSACAgent']['GNN']['MPChannels']"
    v = 64
    spektral_params.append((p, d, v))
    
    d = "specifies the fully connected layers (number and sizes) in the edge network ([int])"
    p = "['ML']['BehaviorGraphSACAgent']['GNN']['KernelNetUnits']"
    v = [256]
    spektral_params.append((p, d, v))
    
    d = "the activation function of the message passing layer (str)"
    p = "['ML']['BehaviorGraphSACAgent']['GNN']['MPLayerActivation']"
    v = "relu"
    spektral_params.append((p, d, v))
    
    d = "the activation function of the final dense layer (str)"
    p = "['ML']['BehaviorGraphSACAgent']['GNN.DenseActication']"
    v = "tanh"
    spektral_params.append((p, d, v))
    
    for p, d, v in shared_params:
        print(f"\033[1mPath:\033[0m {p}")
        print(f"\033[1mDescription:\033[0m {d}")
        print(f"\033[1mDefault value:\033[0m {v}\n")
    
    print("The following paramater only apply when using tf2_gnn.")
    for p, d, v in tf2_gnn_params:
        print(f"\033[1mPath:\033[0m {p}")
        print(f"\033[1mDescription:\033[0m {d}")
        print(f"\033[1mDefault value:\033[0m {v}\n")
    
    print("The following paramater only apply when using Spektral.")
    for p, d, v in spektral_params:
        print(f"\033[1mPath:\033[0m {p}")
        print(f"\033[1mDescription:\033[0m {d}")
        print(f"\033[1mDefault value:\033[0m {v}\n")
    