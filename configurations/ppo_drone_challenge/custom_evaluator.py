import numpy as np
import logging

from bark.world.evaluation import \
  EvaluatorGoalReached, EvaluatorCollisionEgoAgent, \
  EvaluatorCollisionDrivingCorridor, EvaluatorStepCount
from modules.runtime.commons.parameters import ParameterServer
from bark.geometry import *

from src.evaluators.goal_reached import GoalReached
logger = logging.getLogger()

class CustomEvaluator(GoalReached):
  """Shows the capability of custom elements inside
     a configuration.
  """
  def __init__(self,
               params=ParameterServer(),
               eval_agent=None):
    GoalReached.__init__(self,
                         params,
                         eval_agent)
    self._next_goal_definition = -1
    self._last_distance = None
    self._goal_number = 0

  def _add_evaluators(self):
    self._evaluators["goal_reached"] = EvaluatorGoalReached(self._eval_agent)
    self._evaluators["ego_collision"] = \
      EvaluatorCollisionEgoAgent(self._eval_agent)
    self._evaluators["step_count"] = EvaluatorStepCount()

  def _distance_to_next_goal(self, world):
    if self._next_goal_definition is not None:
      agent = world.agents[self._eval_agent]
      agent_state = agent.state
      goal_poly = self._next_goal_definition.goal_shape
      agent_pt = Point2d(agent_state[6], agent_state[9])
      return distance(goal_poly, agent_pt)

  def _evaluate(self, world, eval_results):
    """Returns information about the current world state
    """
    # should read parameter that has been set in the observer
    # print(self._params["ML"]["Maneuver"]["lane_change"])
    agent_state = world.agents[self._eval_agent].state

    done = False
    success = eval_results["goal_reached"]
    collision = eval_results["ego_collision"]
    step_count = eval_results["step_count"]

    next_goal = world.agents[self._eval_agent].goal_definition. \
      GetNextGoal(world.agents[self._eval_agent])
    if self._next_goal_definition is -1:
      self._next_goal_definition = next_goal

    # intermediate goals
    reward = 0.
    if self._next_goal_definition is not next_goal:
      logger.info("Intermediate goal reached.")
      # reward += 1.0
      self._goal_number += 1
      self._next_goal_definition = next_goal
    
    # determine whether the simulation should terminate
    if success or collision or step_count > self._max_steps:
      done = True
    distance = self._distance_to_next_goal(world)
    
    if distance is None:
      distance = 10.
    reward = 0.001*(10.0*self._goal_number - distance)
    reward += collision * self._collision_penalty + \
      success * self._goal_reward
    logger.info("Distance: {}m and reward: {}".format(str(distance), str(reward)))    
    return reward, done, eval_results
    
  def reset(self, world, agents_to_evaluate):
    world = super(CustomEvaluator, self).reset(world, agents_to_evaluate)
    self._next_goal_definition = -1
    self._goal_number = 0
    return world
    
