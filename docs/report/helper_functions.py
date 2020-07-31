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
    