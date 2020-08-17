# Copyright (c) 2020 Patrick Hart, Julian Bernhard,
# Klemens Esterle, Tobias Kessler
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import numpy as np
# BARK
from bark.core.world.evaluation import \
  EvaluatorGoalReached, EvaluatorCollisionEgoAgent, \
  EvaluatorStepCount, EvaluatorDrivableArea
from bark.runtime.commons.parameters import ParameterServer
from bark.core.geometry import *
# BARK-ML
from bark_ml.evaluators.evaluator import StateEvaluator


class GoalReachedGuiding(StateEvaluator):
  def __init__(self,
               params=ParameterServer(),
               eval_agent=None):
    StateEvaluator.__init__(self, params)
    self._goal_reward = \
      self._params["ML"]["GoalReachedGuiding"]["GoalReward",
        "Reward for reaching the goal.",
        1.]
    self._col_penalty = \
      self._params["ML"]["GoalReachedGuiding"]["CollisionPenalty",
        "Reward given for a collisions.",
        -1.]
    self._max_steps = \
      self._params["ML"]["GoalReachedGuiding"]["MaxSteps",
        "Maximum steps per episode.",
        50]
    self._act_penalty = \
      self._params["ML"]["GoalReachedGuiding"]["ActionPenalty",
        "Weight factor for penalizing actions",
        0.01]
    self._goal_dist = \
      self._params["ML"]["GoalReachedGuiding"]["GoalDistance",
        "Weight factor for distance to goal",
        0.01]
    self._eval_agent = eval_agent
    self._goal_lane_corr = None

  def _add_evaluators(self):
    """Evaluators that will be set in the BARK world"""
    self._evaluators["goal_reached"] = EvaluatorGoalReached()
    self._evaluators["collision"] = EvaluatorCollisionEgoAgent()
    self._evaluators["step_count"] = EvaluatorStepCount()
    self._evaluators["drivable_area"] = EvaluatorDrivableArea()

  def GetGoalLaneCorridorForGoal(self, observed_world):
    """Returns the lanecorridor the goal is in"""
    if self._goal_lane_corr is not None:
      return self._goal_lane_corr
    ego_agent = observed_world.ego_agent
    goal_def = ego_agent.goal_definition
    goal_shape = goal_def.goal_shape
    rc = observed_world.ego_agent.road_corridor
    lane_corr = None
    for lc in rc.lane_corridors:
      if Collide(lc.polygon, goal_shape):
        lane_corr = lc
    return lane_corr

  def CalculateDistanceToGoal(self, observed_world, goal_lane_corr):
    """Calculates the distance to the goal of the ego_agent"""
    goal_center_line = goal_lane_corr.center_line
    ego_agent = observed_world.ego_agent
    ego_agent_state = ego_agent.state
    distance_to_gaol = Distance(
      goal_center_line,
      Point2d(ego_agent_state[1], ego_agent_state[2]))
    return distance_to_gaol

  def CalculateGuidingReward(self, observed_world, action):
    """Returns a guiding reward using the dist. to goal and penalized acts."""
    guiding_reward =  0.
    goal_lane_corr = self.GetGoalLaneCorridorForGoal(observed_world)
    distance_to_goal = self.CalculateDistanceToGoal(observed_world, goal_lane_corr)
    guiding_reward -= self._goal_dist*distance_to_goal
    # NOTE: this will only work for continious actions
    if action is not None and type(action) is not int:
      accs = action[0]
      delta = action[1]
      guiding_reward -= self._act_penalty*(accs**2 + delta*+2)
    return guiding_reward

  def _evaluate(self, observed_world, eval_results, action):
    """Returns information about the current world state
    """
    done = False
    success = eval_results["goal_reached"]
    collision = eval_results["collision"] or eval_results["drivable_area"]
    step_count = eval_results["step_count"]
    # determine whether the simulation should terminate
    if success or collision or step_count > self._max_steps:
      done = True
    guiding_reward = self.CalculateGuidingReward(observed_world, action)
    # calculate reward
    reward = collision * self._col_penalty + \
      success * self._goal_reward + guiding_reward
    return reward, done, eval_results
    
  def Reset(self, world):
    self._goal_lane_corr = None
    return super(GoalReachedGuiding, self).Reset(world)