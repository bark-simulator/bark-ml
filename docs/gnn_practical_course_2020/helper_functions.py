import logging
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import time
import networkx as nx
import tensorflow as tf
import datetime
import logging
import shutil
from collections import OrderedDict
from matplotlib.patches import Ellipse
from tf2_gnn.layers import GNN, GNNInput

# BARK imports
from bark.runtime.commons.parameters import ParameterServer
from bark.runtime.viewer.matplotlib_viewer import MPViewer
from bark.runtime.viewer.video_renderer import VideoRenderer

# BARK-ML imports
from bark_ml.environments.blueprints import ContinuousHighwayBlueprint, \
  ContinuousMergingBlueprint, ContinuousIntersectionBlueprint
from bark_ml.environments.single_agent_runtime import SingleAgentRuntime
from bark_ml.library_wrappers.lib_tf_agents.agents import BehaviorSACAgent,\
  BehaviorPPOAgent, BehaviorGraphSACAgent
from bark_ml.library_wrappers.lib_tf_agents.runners import SACRunner, PPORunner
from bark_ml.observers.graph_observer import GraphObserver
from bark_ml.library_wrappers.lib_tf_agents.networks.gnn_wrapper import GNNWrapper 

# Supervised specific imports
from bark_ml.tests.capability_gnn_actor.data_generation import DataGenerator
from bark_ml.tests.capability_gnn_actor.actor_nets import ConstantActorNet,\
  RandomActorNet
from bark_ml.tests.capability_gnn_actor.data_generation import DataGenerator
from bark_ml.tests.capability_gnn_actor.data_handler import SupervisedData
from bark_ml.tests.capability_gnn_actor.learner import Learner

def clean_log_dir(log_dir):
  if os.path.exists(log_dir):
    [shutil.rmtree(os.path.join(log_dir,log)) for log in os.listdir(log_dir)]
  logging.info("Log dir is clean!")

def explain_observation(observation, graph_dims):
    observation = observation.numpy()
    nodes, n_feats, e_feats = graph_dims
    # node attributes
    end_1 = nodes*n_feats
    node_attributes = observation[:end_1]
    print("Node_attributes(flattened matrix of original shape "+str(nodes)+"x"+str(n_feats)+\
          ") (nodes x node attributes):\n",node_attributes)
    # Adjacency matrix
    end_2 = end_1 + nodes**2
    adjacency_matrix = observation[end_1:end_2]
    print("Adjacency matrix(flattened matrix of original shape "+str(nodes)+"x"+str(nodes)+\
          ") (nodes x nodes):\n",adjacency_matrix)
    # edge features
    edge_attributes = observation[end_2:]
    print("Edge_attributes(flattened matrix of original shape "+str(nodes**2)+"x"+str(e_feats)+\
          ") (number of edges x edge attributes):\n",edge_attributes)

def configurable_setup(params, num_scenarios, graph_sac=True):
  """Configurable GNN setup depending on a given filename

  Args:
    params: ParameterServer instance

  Returns: 
    observer: GraphObserver instance
    actor: ActorNetwork of BehaviorGraphSACAgent
  """
  observer = GraphObserver(params=params)
  bp = ContinuousHighwayBlueprint(params,
                                  number_of_senarios=num_scenarios,
                                  random_seed=0)
  env = SingleAgentRuntime(blueprint=bp, observer=observer,
                            render=False)
  if graph_sac:
    # Get GNN SAC actor net
    sac_agent = BehaviorGraphSACAgent(environment=env, observer=observer,
                                      params=params)
  else:
    sac_agent = BehaviorSACAgent(environment=env, params=params)

  actor = sac_agent._agent._actor_network
  return observer, actor

def benchmark_actor(actor, dataset, epochs, log_dir=None):
  """Benchmarks an actor on a dataset"""
  train_dataset = dataset._train_dataset
  test_dataset = dataset._test_dataset
  learner = Learner(actor, train_dataset, test_dataset,
                    log_dir=log_dir)
  if (actor.__class__==ConstantActorNet) or \
        (actor.__class__==RandomActorNet):
    mode = "Number"
    only_test = True
  else:
    mode = "Distribution"
    only_test = False

  loss = learner.train(epochs=epochs, only_test=only_test, mode=mode)
  return loss

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
  ellipse = Ellipse(pos[0], width=width,height=height, facecolor='yellow',
                    zorder=-1)#,**kwargs)
  ax.add_patch(ellipse)
  goal_ellipse = Ellipse(goal[0], width= 0.2, height=0.2, facecolor="green",
                         zorder=-2)
  ax.add_patch(goal_ellipse)

  # Change color for ego agent
  node_colors = ["blue" for i in range(len(graph.nodes))]
  node_colors[0] = "red"
  return nx.draw(graph, pos = pos, with_labels=True, ax=ax,
                 node_color=node_colors)

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

  observation = np.array([])
  observation = np.append(observation, agents)
  observation = np.append(observation, adjacency_list)
  observation = np.append(observation, edge_features)
  observation = observation.reshape(-1)
    
  return tf.cast(tf.stack([observation, observation]), tf.float32)

def prepare_agent(agent, params, env):
  env.ml_behavior = agent
  runner = SACRunner(params=params, environment=env, agent=agent)
  
  iterator = iter(agent._dataset)
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
    
  print(f"\n\033[1mAGENT SUMMARY\033[0m\n")
  print('{:<30} {}'.format("Network", "Parameters"))
  print("==========================================")
  total_params = 0
  for net in networks:
    n = np.sum([np.prod(v.get_shape().as_list()) for v in net.trainable_variables])
    total_params += n
    print('{:.<34} {:,}'.format(net.name, n).replace(',','.'))
  
  print("------------------------------------------")
  print('{:<32} {:,}'.format("Total parameters", total_params).replace(',','.'))


def run_rl_example(env, agent, params, mode="visualize"):
  env.ml_behavior = agent
  runner = SACRunner(params=params,
                     environment=env,
                     agent=agent)

  if mode == "train":
    runner.SetupSummaryWriter()
    runner.Train()
  elif mode == "visualize":
    runner.Run(num_episodes=10, render=True)
  elif mode == "evaluate":
    runner.Run(num_episodes=10, render=False)
    