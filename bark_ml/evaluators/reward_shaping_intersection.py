# Copyright (c) 2020 Patrick Hart, Julian Bernhard,
# Klemens Esterle, Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import numpy as np
import matplotlib.pyplot as plt
from bark.core.world.evaluation import \
  EvaluatorGoalReached, EvaluatorCollisionEgoAgent, \
  EvaluatorStepCount, EvaluatorDrivableArea
from bark.runtime.commons.parameters import ParameterServer
from bark.core.geometry import *

from bark_ml.evaluators.reward_shaping import RewardShapingEvaluator
from bark_ml.evaluators.commons import *


def DistancePotential(d, d_max, b=0.4):
  return 1. - (d/d_max)**b


class RewardShapingEvaluatorIntersection(RewardShapingEvaluator):
  """Reward shaping evaluator using potential functions.

  Implemented are potential functions fot the distance to
  the goal $\phi(d)$, velocity $\phi(v)$, and distance to
  other agents $\phi(d_i)$.

  Reward signal: $r(s_t, a_t, s_{t+1}) + F(s_t, a_t, s_{t+1})$

  with the shaping function $F = \gamma*\phi(s_{t+1}) - \phi(s_t)$
  """

  def __init__(self,
               params=ParameterServer(),
               eval_agent=None):
    RewardShapingEvaluator.__init__(self, params, eval_agent)


  def RewardShapingFunction(self, observed_world, gamma=0.99):
    ego_agent = observed_world.ego_agent
    state_history = [state_action[0] for state_action in ego_agent.history[-2:]]
    last_state, current_state = state_history
    params_goal = self._active_shaping_functions["DistancePotential"]
    positive_potentials = []

    rc = observed_world.ego_agent.road_corridor
    goal_corr_lc = rc.lane_corridors[0]
    goal_center_line = goal_corr_lc.center_line
    if Distance(
        goal_center_line,
        Point2d(current_state[1], current_state[2])) < params_goal["d_max"]:
      goal_potentials = [DistancePotential(
        Distance(
          goal_center_line,
          Point2d(state[1], state[2])),
        d_max=params_goal["d_max"],
        b=params_goal["exponent"]) for state in [last_state, current_state]]
      positive_potentials.append(goal_potentials)

    # reward shaping function gamma*p_{t+1} - p_t
    reward_shaping_value = 0.
    for potential_values in positive_potentials:
      reward_shaping_value += gamma*potential_values[1] - potential_values[0]
    return reward_shaping_value

  def _evaluate(self, observed_world, eval_results, action):
    """Returns information about the current world state."""
    done = False
    success = eval_results["goal_reached"]
    step_count = eval_results["step_count"]
    collision = eval_results["collision"] or eval_results["drivable_area"]

    reward_shaping_signal = self.RewardShapingFunction(observed_world)
    if success or collision or step_count > self._max_steps:
      done = True

    if collision:
      success = 0
      # TODO: needs to be for logging
      eval_results["goal_reached"] = False

    # for now it is only collision and success
    reward = collision * self._col_penalty + \
      success * self._goal_reward + reward_shaping_signal
    return reward, done, eval_results

  def Reset(self, world):
    return super(RewardShapingEvaluator, self).Reset(world)