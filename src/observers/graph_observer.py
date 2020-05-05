
from gym import spaces
import numpy as np
import math
import operator

from bark.models.dynamic import StateDefinition
from bark.world import ObservedWorld
from modules.runtime.commons.parameters import ParameterServer
from src.observers.observer import StateObserver

class Graph(object):
  """
  Abstraction of the observed world in a graph representation.
  This class could integrate networkx to build the graph, or only use some features
  to visualize it.
  """

  def __init__(self):
    self.nodes = {}
    self.edges = [] # list of two-element tuples

  def add_node(self, id, features):
    # assert id is str
    # assert features is list

    self.nodes[id] = features

  def add_edge(self, source: str, target: str):
    self.edges.append((source, target))

#####

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
    graph = Graph()
    ego_agent = world.ego_agent

    for (_, agent) in world.agents.items():      
      features = self._extract_features(agent)
      # normalize the features here?
      graph.add_node(id=str(agent.id), features=features)

      for nearby_agent in world.GetNearestAgents(world.ego_position, 3):
        graph.add_edge(source=agent.id, target=nearby_agent.id)

    return graph

  def _extract_features(self, agent):
    state = agent.state
    if self._normalize_observations: 
      self._normalize(agent.state)

    goal_center = agent.goal_definition.goal_shape.center[0:2]
    agent_position = (
      agent.state[int(StateDefinition.X_POSITION)], 
      agent.state[int(StateDefinition.Y_POSITION)]
    )

    distance_to_goal = np.linalg.norm(agent_position - goal_center)

    # TODO: add more features here

    return [
      state[int(StateDefinition.X_POSITION)],
      state[int(StateDefinition.Y_POSITION)],
      state[int(StateDefinition.THETA_POSITION)],
      state[int(StateDefinition.VEL_POSITION)],    
      distance_to_goal,
    ]

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
    return len(self._state_definition)


