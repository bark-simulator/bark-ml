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

def VelocityPotential(v, v_des, v_dev_max=10., a=0.4):
  return 1. - (np.sqrt((v-v_des)**2)/v_dev_max)**a

def DistancePotential(d, d_max, b=0.4):
  return 1. - (d/d_max)**b

def ObjectPotential(d, d_max, c=0.4):
  return -1. + (d/d_max)**c


class RewardShapingEvaluator(BaseEvaluator):
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
    BaseEvaluator.__init__(self, params)
    self._goal_reward = \
      self._params["ML"]["RewardShapingEvaluator"]["GoalReward",
        "Reward for reaching the goal.",
        1.]
    self._col_penalty = \
      self._params["ML"]["RewardShapingEvaluator"]["CollisionPenalty",
        "Reward given for a collisions.",
        -1.]
    self._max_steps = \
      self._params["ML"]["RewardShapingEvaluator"]["MaxSteps",
        "Maximum steps per episode.",
        60]
    self._max_vel = \
      self._params["ML"]["RewardShapingEvaluator"]["MaxVelocity",
        "Maximum velocity in episode.",
        10]
    self._active_shaping_functions = \
      self._params["ML"]["RewardShapingEvaluator"]["RewardShapingPotentials",
        "Reward shaping functions.", {
          "VelocityPotential" : {
            "desired_vel": 10., "vel_dev_max": 10., "exponent": 0.4, "type": "positive"
          },
          "DistancePotential": {
            "exponent": 0.4, "d_max": 10., "type": "positive"
          },
          "DistanceOtherPotential": {
            "exponent": 0.4, "d_max": 20., "type": "negative"
          }
        }]
    self._eval_agent = eval_agent

  @staticmethod
  def _add_evaluators():
    evaluators = {}
    evaluators["goal_reached"] = EvaluatorGoalReached()
    evaluators["collision"] = EvaluatorCollisionEgoAgent()
    evaluators["step_count"] = EvaluatorStepCount()
    evaluators["drivable_area"] = EvaluatorDrivableArea()
    return evaluators

  def RewardShapingFunction(self, observed_world, gamma=0.99):
    ego_agent = observed_world.ego_agent
    state_history = [state_action[0] for state_action in ego_agent.history[-2:]]
    last_state, current_state = state_history
    negative_potentials, positive_potentials = [], []
    if "VelocityPotential" in self._active_shaping_functions:
      params_vp = self._active_shaping_functions["VelocityPotential"]
      vel_potentials = [VelocityPotential(
        state[int(StateDefinition.VEL_POSITION)],
        params_vp["desired_vel"],
        v_dev_max=params_vp["vel_dev_max"],
        a=params_vp["exponent"]) for state in [last_state, current_state]]
      positive_potentials.append(vel_potentials)
    if "DistancePotential" in self._active_shaping_functions:
      params_goal = self._active_shaping_functions["DistancePotential"]
      goal_center_line = ego_agent.goal_definition.center_line
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
    if "DistanceOtherPotential" in self._active_shaping_functions:
      params_obj = self._active_shaping_functions["DistanceOtherPotential"]
      # TODO: integrate lon lat offsets
      for agent in observed_world.other_agents.values():
        other_state_history = [state_action[0] for state_action in agent.history[-2:]]
        other_last_state, other_current_state = other_state_history
        if Distance(
          Point2d(other_current_state[1], other_current_state[2]),
          Point2d(current_state[1], current_state[2])) > params_obj["d_max"]:
          continue
        goal_potentials = [ObjectPotential(
          Distance(
            Point2d(state[0][1], state[0][2]),
            Point2d(state[1][1], state[1][2])),
          d_max=params_obj["d_max"],
          c=params_obj["exponent"]) for state in [
            (last_state, other_last_state), (current_state, other_current_state)]]
        negative_potentials.append(goal_potentials)

    number_pos_potentials = len(positive_potentials)
    number_neg_potentials = len(negative_potentials)

    # normalize
    for i in range(0, number_pos_potentials):
      positive_potentials[i][0] /= number_pos_potentials
      positive_potentials[i][1] /= number_pos_potentials
    for i in range(0, number_neg_potentials):
      negative_potentials[i][0] /= number_neg_potentials
      negative_potentials[i][1] /= number_neg_potentials

    # reward shaping function gamma*p_{t+1} - p_t
    reward_shaping_value = 0.
    for potential_values in positive_potentials:
      reward_shaping_value += gamma*potential_values[1] - potential_values[0]
    for potential_values in negative_potentials:
      reward_shaping_value += gamma*potential_values[1] - potential_values[0]
    return reward_shaping_value

  def _evaluate(self, observed_world, eval_results, action):
    """Returns information about the current world state."""
    done = False
    success = eval_results["goal_reached"]
    step_count = eval_results["step_count"]
    collision = eval_results["collision"] or eval_results["drivable_area"]

    # TERMINATE WITH NEGATIVE VEL. or if larger than max. vel
    ego_state = observed_world.ego_agent.state
    ego_vel = ego_state[int(StateDefinition.VEL_POSITION)] < 0 or \
      ego_state[int(StateDefinition.VEL_POSITION)] > self._max_vel

    reward_shaping_signal = self.RewardShapingFunction(observed_world)
    if success or collision or step_count > self._max_steps or ego_vel:
      done = True

    if collision:
      success = 0
      # TODO: needs to be for logging
      eval_results["goal_reached"] = False

    # for now it is only collision and success
    reward = collision * self._col_penalty + \
      success * self._goal_reward + reward_shaping_signal + \
      ego_vel * self._col_penalty
    return reward, done, eval_results

  def Reset(self, world):
    return super(RewardShapingEvaluator, self).Reset(world)