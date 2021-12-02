# Copyright (c) 2020 Patrick Hart, Julian Bernhard,
# Klemens Esterle, Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from bark.core.world.evaluation import \
  EvaluatorGoalReached, EvaluatorCollisionEgoAgent, \
  EvaluatorStepCount, EvaluatorDrivableArea
from bark.runtime.commons.parameters import ParameterServer

from bark_ml.evaluators.evaluator import BaseEvaluator


class GeneralEvaluator:
  """Sparse reward evaluator returning +1 for reaching the goal,
  -1 for having a collision or leaving the drivable area."""

  def __init__(self,
               params=ParameterServer(),
               eval_agent=None):
    self._eval_agent = eval_agent
    self._params = params
    self._bark_eval_fns = {
      "goal_reached" : EvaluatorGoalReached(),
      "collision" : EvaluatorCollisionEgoAgent(),
      "step_count" : EvaluatorStepCount(),
      "drivable_area" : EvaluatorDrivableArea()
    }
    self._bark_ml_eval_fns = {
      "collision" : collision_fn,
      "terminal" : terminal_fn,
      "success" : sucess_fn,
      "desired_velocity" : desired_vel_fn,
      "distance_to_centerline" : distanec_to_centerline
    }

  def Evaluate(self, observed_world, action):
    """Returns information about the current world state."""
    eval_results = observed_world.Evaluate()
    done = False
    reward = 0.
    # collision_fn -> terminal, reward, additional_info
    # TODO: XOR the done result
    return reward, done, eval_results

  def Reset(self, world):
    world.ClearEvaluators()
    for eval_name, eval_fn in self._bark_eval_fns:
      world.AddEvaluator(eval_name, eval_fn)
    return world

  def SetViewer(self, viewer):
    self._viewer = viewer