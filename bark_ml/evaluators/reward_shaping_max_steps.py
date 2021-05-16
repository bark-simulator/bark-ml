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

from bark_ml.evaluators.reward_shaping import RewardShapingEvaluator
from bark_ml.evaluators.commons import *

def VelocityPotential(v, v_des, v_dev_max=10., a=0.4):
  return 1. - (np.sqrt((v-v_des)**2)/v_dev_max)**a

def DistancePotential(d, d_max, b=0.4):
  return 1. - (d/d_max)**b

def ObjectPotential(d, d_max, c=0.4):
  return -1. + (d/d_max)**c


class RewardShapingEvaluatorMaxSteps(RewardShapingEvaluator):
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

  def _evaluate(self, observed_world, eval_results, action):
    """Returns information about the current world state."""
    done = False
    step_count = eval_results["step_count"]
    success = True if (step_count == self._max_steps) else False
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