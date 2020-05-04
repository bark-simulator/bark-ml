
from gym import spaces
import numpy as np
import math
import operator

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
    # Get info from ego vehicle
    ego = world.ego_agent # seems not to work for two controlled/ego vehicles properly
    state = ego.state #equal to world.ego_state
    if self._normalize_observations:
      state = self._normalize(state)
    id_ = ego.id
    goal = ego.goal_definition
    road_corridor = world.road_corridor
    lane_corridor = world.lane_corridor
    other_agents = world.other_agents
    #print(road_corridor)
    #print(lane_corridor)
    #print(other_agents)
    #print(help(corridor))
    # Get middle point of goal Polygon as reference point(assumption: rectangular)
    goal_center = goal.goal_shape.ToArray()[1:].mean(axis=0)
    at_goal = goal.AtGoal(ego) # Seems not to work as intuitivly expected, always false -> unsure about correct usage
    print('{:2.2f}s: Ego: id {}, goal {}'.format(world.time, id_, goal_center))
    
    # Check data of other agents
    for agent_id in other_agents:
      agent = other_agents[agent_id]
      state = agent.state
      assert(agent_id is agent.id)
      if self._normalize_observations:
        state = self._normalize(state)
        #print("Normed state: ", state)
    
    ## build graph here
    graph = ego.state

    return graph

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


