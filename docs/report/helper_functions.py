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