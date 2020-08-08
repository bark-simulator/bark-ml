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
      self._params["ML"]["GoalReachedEvaluator"]["GoalReward",
        "Reward for reaching the goal.",
        1.]
    self._col_penalty = \
      self._params["ML"]["GoalReachedEvaluator"]["CollisionPenalty",
        "Reward given for a collisions.",
        -1.]
    self._max_steps = \
      self._params["ML"]["GoalReachedEvaluator"]["MaxSteps",
        "Maximum steps per episode.",
        50]
    self._eval_agent = eval_agent

  def _add_evaluators(self):
    evaluators = {}
    evaluators["goal_reached"] = EvaluatorGoalReached()
    evaluators["collision"] = EvaluatorCollisionEgoAgent()
    evaluators["step_count"] = EvaluatorStepCount()
    evaluators["drivable_area"] = EvaluatorDrivableArea()
    return evaluators

  def CalculateGuidingReward(self, observed_world, action):
    ego_agent = observed_world.ego_agent
    goal_def = ego_agent.goal_definition
    goal_shape = goal_def.goal_shape
    rc = observed_world.ego_agent.road_corridor
    lane_corr = None
    for lc in rc.lane_corridors:
      if Collide(lc.polygon, goal_shape):
        lane_corr = lc
    
    total_reward =  0.
    goal_center_line = lane_corr.center_line
    ego_agent_state = ego_agent.state
    lateral_offset = Distance(goal_center_line,
                              Point2d(ego_agent_state[1], ego_agent_state[2]))
    total_reward -= 0.01*lateral_offset

    if action is not None:
      accs = action[0]
      delta = action[1]
      total_reward -= 0.01*(accs**2 + delta*+2)
    return total_reward

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
    return super(GoalReachedGuiding, self).Reset(world)