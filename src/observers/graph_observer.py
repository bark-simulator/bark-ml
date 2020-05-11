import os, time, json, pickle
from gym import spaces
import numpy as np
import math
import operator
import networkx as nx
from typing import Dict

from bark.models.dynamic import StateDefinition
from bark.world import ObservedWorld
from bark.geometry import Distance
from modules.runtime.commons.parameters import ParameterServer
from src.observers.observer import StateObserver

class Graph(object):
  """
  An abstraction of the observed world in a graph representation.
  """

  def __init__(self, ego_agent_id: str):
    self._graph = nx.Graph(ego_agent_id=ego_agent_id)

  @property
  def nx_graph(self):
    """
    The underlying `nx.Graph` object.
    """
    return self._graph

  @property
  def data(self):
    """
    A dictionary representation of the graph.
    """
    return nx.node_link_data(self._graph) 

  def add_node(self, id: str, attributes: Dict[str, float]):
    """
    Adds a node with the specified identifier and attributes
    to the graph.
    If a node with this id already exists, it will replaced.
    """
    self._graph.add_node(id, **attributes)

  def add_edge(self, source: str, target: str, attributes: Dict[str, float]=None):
    """ 
    Adds an undirected edge between the specified source 
    and target nodes with the given attributes.
    If an edge between these two nodes already exists, 
    it will replaced.
    
    :param source:str: identifier of the source node.
    :param target:str: identifier of the target node.
  
    :param attributes:Dict[str, float]: A dictionary\
      of attributes with string keys and float values.
    """
    assert None not in [source, target], \
      "Specifying a source and target node id is mandatory."
    assert source != target, \
      "Source node id is equal to target node id, which is not allowed."
    self._graph.add_edge(source, target, **attributes)


class GraphObserver(StateObserver):
  
  def __init__(self,
               normalize_observations=True,
               use_edge_attributes=True,
               params=ParameterServer()):
    StateObserver.__init__(self, params)

    self._observation_len = 1 # i.e. one graph
    self._normalize_observations = normalize_observations
    self._use_edge_attributes = use_edge_attributes

  def observe(self, world):
    """see base class
    """
    ego_agent = world.ego_agent
    graph = Graph(ego_agent_id=ego_agent.id)
    actions = {} # generated for now (steering, acceleration)
    
    for (_, agent) in world.agents.items():      
      # create node
      features = self._extract_features(agent)
      graph.add_node(id=str(agent.id), attributes=features)

      # generate actions
      actions[agent.id] = self._generate_actions(features)
      
      # create edges to all other agents
      other_agents = list(filter(lambda a: a.id != agent.id, world.agents.values()))

      for other_agent in other_agents:
        edge_attributes = None
        
        if self._use_edge_attributes: 
          edge_attributes = self._extract_edge_attributes(agent, other_agent)
        
        graph.add_edge(source=str(agent.id), 
                       target=str(other_agent.id), 
                       attributes=edge_attributes)

    return graph, actions

  def _extract_edge_attributes(self, agent1, agent2) -> Dict[str, float]:
    """
    Returns a dict containing attributes for an edge 
    between two agents.
    """
    x_key = int(StateDefinition.X_POSITION)
    y_key = int(StateDefinition.Y_POSITION)
    vel_key = int(StateDefinition.VEL_POSITION)

    attributes = {}
    attributes['dx'] = agent1.state[x_key] - agent2.state[x_key]
    attributes['dy'] = agent1.state[y_key] - agent2.state[y_key]
    attributes['dvel'] = agent1.state[vel_key] - agent2.state[vel_key]
    # maybe include difference in steering angle

    return attributes

  def _extract_features(self, agent) -> Dict[str, float]:
    """Returns dict containing all features of the agent"""
    agent_features = {}
    
    # Get basic information
    state = agent.state
    agent_features["x"] = state[int(StateDefinition.X_POSITION)] # maybe not needed
    agent_features["y"] = state[int(StateDefinition.Y_POSITION)] # maybe not needed
    agent_features["theta"] = state[int(StateDefinition.THETA_POSITION)]
    agent_features["vel"] = state[int(StateDefinition.VEL_POSITION)]

    # Get information related to goal
    goal_center = agent.goal_definition.goal_shape.center[0:2]
    goal_dx = goal_center[0] - agent_features["x"] # Distance to goal in x coord
    goal_dy = goal_center[1] - agent_features["y"] # Distance to goal in y coord
    goal_theta = np.arctan2(goal_dy, goal_dx) # Theta for straight line to goal
    goal_d = np.sqrt(goal_dx**2 + goal_dy**2) # Distance to goal
    agent_features["goal_d"] = goal_d
    agent_features["goal_theta"] = goal_theta
    
    return agent_features

  def _generate_actions(self, features: Dict[str, float]) -> Dict[str, float]:
    labels = dict()
    steering = features["goal_theta"] - features["theta"]
    labels["steering"] = steering
    
    dt = 2 # parameter: time constant in s for planning horizon
    acc = 2. * (features["goal_d"] - features["vel"] * dt) / (dt**2)
    labels["acceleration"] = acc
    return labels

  def _norm(self, agent_state, position, range):
    agent_state[int(position)] = \
      (agent_state[int(position)] - range[0])/(range[1]-range[0])
    return agent_state

  def _normalize(self, agent_state):
    agent_state = \
      self._norm(agent_state,
                 StateDefinition.X_POSITION,
                 self._world_x_range)
    agent_state = \
      self._norm(agent_state,
                 StateDefinition.Y_POSITION,
                 self._world_y_range)
    agent_state = \
      self._norm(agent_state,
                 StateDefinition.THETA_POSITION,
                 self._theta_range)
    agent_state = \
      self._norm(agent_state,
                 StateDefinition.VEL_POSITION,
                 self._velocity_range)
    return agent_state

  def reset(self, world):
    return world

  @property
  def observation_space(self):
    return spaces.Box(
      low=np.zeros(self._observation_len),
      high=np.ones(self._observation_len))

  @property
  def _len_state(self):
    return 1