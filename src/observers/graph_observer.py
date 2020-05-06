
from gym import spaces
import numpy as np
import math
import operator
import networkx as nx

from bark.models.dynamic import StateDefinition
from bark.world import ObservedWorld
from modules.runtime.commons.parameters import ParameterServer
from src.observers.observer import StateObserver


class GraphObserver(StateObserver):
  def __init__(self,
               normalize_observations=True,
               params=ParameterServer()):
    StateObserver.__init__(self, params)

    self._state_definition = [int(StateDefinition.X_POSITION),
                              int(StateDefinition.Y_POSITION),
                              int(StateDefinition.THETA_POSITION),
                              int(StateDefinition.VEL_POSITION)]
    self._observation_len = \
      self._max_num_vehicles*self._len_state
    self._normalize_observations = normalize_observations

  def observe(self, world):
    """see base class
    """
    # Init empty graph
    G = nx.Graph()

    # Get info from ego vehicle
    ego = world.ego_agent # seems not to work for two controlled/ego vehicles properly
    ego_id = ego.id
    G.graph['ego_id'] = ego_id
    ego_features = self._extract_features(ego)
    G.add_node(str(ego_id), **ego_features)
    #print('{:2.2f}s: Ego: id {}, goal {}'.format(world.time, id_, goal_center))
    
    # Check data of other agents
    for agent_id in world.other_agents:
      agent = world.other_agents[agent_id]
      agent_features = self._extract_features(agent)
      G.add_node(str(agent_id), **agent_features) #Uses dict key:datum pairs as needed

    # Add edge information
    for agent in G.nodes:
      #print(G.nodes[agent])
      x1, y1 = G.nodes[agent]["x"], G.nodes[agent]["y"]
      for agent_2 in G.nodes:
          if agent_2 != agent:
              x2, y2 = G.nodes[agent_2]["x"], G.nodes[agent_2]["y"]
              dist = self._calc_distance(x1,y1,x2,y2)
              if agent == str(G.graph['ego_id']) or agent_2 == str(G.graph['ego_id']):
                  G.add_edge(agent, agent_2, distance=dist, category=1)
              else:
                  G.add_edge(agent, agent_2, distance=dist, category=2)
    
    # for now: calc labels
    # Option1: only for ego vehicle
    """graph_labels = self._calc_labels(ego_features)
    graph_labels = {ego_id: graph_labels}"""
    # Option2: Calc labels for every agent
    graph_labels = dict()
    for agent in G.nodes:
      agent_labels = self._calc_labels(G.nodes[agent])
      graph_labels[agent] = agent_labels
    graph_data = nx.node_link_data(G) # Convert to dict format
    #print("graph_data:", graph_data.__class__)
    #print("graph_labels: ", graph_labels.__class__)
    return graph_data, graph_labels

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

  def _extract_features(self, agent):
    """Returns dict containing all features of the agent"""
    agent_features = dict()
    # Get basic information
    state = agent.state
    agent_features["x"] = state[int(StateDefinition.X_POSITION)]
    agent_features["y"] = state[int(StateDefinition.Y_POSITION)]
    agent_features["theta"] = state[int(StateDefinition.THETA_POSITION)]
    agent_features["v"] = state[int(StateDefinition.VEL_POSITION)]

    # Get information related to goal
    goal_center = agent.goal_definition.goal_shape.center[0:2]
    goal_dx = goal_center[0] - agent_features["x"] # Distance to goal in x coord
    goal_dy = goal_center[1] - agent_features["y"] # Distance to goal in y coord
    goal_d = np.sqrt(goal_dx**2 + goal_dy**2) # Distance to goal
    goal_theta = np.arctan2(goal_dy, goal_dx) # Theta for straight line to goal
    agent_features["goal_dx"] = goal_dx
    agent_features["goal_dy"] = goal_dy
    agent_features["goal_d"] = goal_d
    agent_features["goal_theta"] = goal_theta
    return agent_features

  def _calc_labels(self, agent_features):
    labels = dict()
    steering = agent_features["goal_theta"] - agent_features["theta"]
    labels["steering"] = steering
    #########################################
    dt = 2 # parameter: time constant in s for planning horizon
    #########################################
    acc = 2.*(agent_features["goal_d"]- agent_features["v"]*dt)/(dt**2)
    labels["acceleration"] = acc
    return labels
  
  def _calc_distance(self, x1, y1, x2, y2):
    return np.sqrt((x1-x2)**2+(y1-y2)**2)

  def reset(self, world):
    return world

  @property
  def observation_space(self):
    return spaces.Box(
      low=np.zeros(self._observation_len),
      high=np.ones(self._observation_len))

  @property
  def _len_state(self):
    return len(self._state_definition)


