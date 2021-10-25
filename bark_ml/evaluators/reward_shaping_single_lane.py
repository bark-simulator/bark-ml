# Copyright (c) 2020 Patrick Hart, Julian Bernhard,
# Klemens Esterle, Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import numpy as np
from bark.core.world.evaluation import \
  EvaluatorGoalReached, EvaluatorCollisionEgoAgent, \
  EvaluatorStepCount, EvaluatorDrivableArea
from bark.runtime.commons.parameters import ParameterServer
from bark.core.models.dynamic import StateDefinition
from bark.core.geometry import *

from bark_ml.evaluators.evaluator import BaseEvaluator
from bark_ml.evaluators.commons import *



class SingleLaneEvaluator(BaseEvaluator):
  def __init__(self,
               params=ParameterServer(),
               eval_agent=None):
    BaseEvaluator.__init__(self, params)
    self._goal_reward = \
      self._params["ML"]["SingleLaneEvaluator"]["GoalReward",
        "Reward for reaching the goal.",
        1.]
    self._col_penalty = \
      self._params["ML"]["SingleLaneEvaluator"]["CollisionPenalty",
        "Reward given for a collisions.",
        -1.]
    self._max_steps = \
      self._params["ML"]["SingleLaneEvaluator"]["MaxSteps",
        "Maximum steps per episode.",
        200]
    self._desired_vel = \
      self._params["ML"]["SingleLaneEvaluator"]["DesiredVelocity",
        "Maximum steps per episode.",
        5.]
    self._desired_vel_weight = \
      self._params["ML"]["SingleLaneEvaluator"]["DesiredVelocityWeight",
        "Weight.",
        0.01]
    self._eval_agent = eval_agent

  @staticmethod
  def _add_evaluators():
    evaluators = {}
    evaluators["goal_reached"] = EvaluatorGoalReached()
    evaluators["collision"] = EvaluatorCollisionEgoAgent()
    evaluators["step_count"] = EvaluatorStepCount()
    evaluators["drivable_area"] = EvaluatorDrivableArea()
    return evaluators


  def _evaluate(self, observed_world, eval_results, action):
    """Returns information about the current world state."""
    done = False
    success = eval_results["goal_reached"]
    step_count = eval_results["step_count"]
    collision = eval_results["collision"] or eval_results["drivable_area"]

    # TERMINATE WITH NEGATIVE VEL.
    ego_state = observed_world.ego_agent.state
    ego_vel = ego_state[int(StateDefinition.VEL_POSITION)]
    ego_vel_is_neg = ego_state[int(StateDefinition.VEL_POSITION)] < 0.

    if collision:
      success = 0
      eval_results["goal_reached"] = False

    if success or collision or step_count > self._max_steps or ego_vel:
      done = True

    # for now it is only collision and success
    reward = collision * self._col_penalty + \
      success * self._goal_reward + \
      ego_vel_is_neg * self._col_penalty - \
      self._desired_vel_weight * np.sqrt((ego_vel-self._desired_vel)**2)
    return reward, done, eval_results

  def Reset(self, world):
    return super(SingleLaneEvaluator, self).Reset(world)