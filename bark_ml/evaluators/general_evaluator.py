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


class CollisionFunctor:
  def __init__(self, params):
    self._params = params["CollisionFunctor"]

  def __call__(self, observed_world, action, eval_results):
    if eval_results["collision"]:
      return True, self._params["CollisionReward"], {}


class GoalFunctor:
  def __init__(self, params):
    self._params = params["GoalFunctor"]

  def __call__(self, observed_world, action, eval_results):
    if eval_results["goal_reached"]:
      return True, self._params["GoalReward"], {}


class DrivableAreaFunctor:
  def __init__(self, params):
    self._params = params["DrivableAreaFunctor"]

  def __call__(self, observed_world, action, eval_results):
    if eval_results["drivable_area"]:
      return True, self._params["DrivableAreaReward"], {}


class StepCountFunctor:
  def __init__(self, params):
    self._params = params["StepCountFunctor"]

  def __call__(self, observed_world, action, eval_results):
    if eval_results["step_count"] > self._params["MaxStepCount"]:
      return True, self._params["StepCountReward"], {}


class MaxVelFunctor:
  def __init__(self, params):
    self._params = params["MaxVelFunctor"]

  def __call__(self, observed_world, action, eval_results):
    ego_agent = observed_world.ego_agent
    ego_vel = ego_agent.state[int(StateDefinition.VEL_POSITION)]
    if ego_vel > self._params["MaxVel"]:
      return True, self._params["MaxVelViolationReward"], {}


class SmoothnessFunctor:
  def __init__(self, params):
    self._params = params["SmoothnessFunctor"]

  def __call__(self, observed_world, action, eval_results):
    ego_agent = observed_world.ego_agent
    if len(ego_agent.history) >= 2:
      actions = np.array([s_a[1] for s_a in ego_agent.history[-2:]])
      action_diff = abs(actions[1] - actions[0])/self._params["Dt"]
      if action_diff[0] > self._params["MaxAccRate"] or \
        action_diff[1] > self._params["MaxSteeringRate"]:
          return True, self._params["InputRateViolation"], {}


class PotentialBasedFunctor:
  def __init__(self, params):
    self._params = params["PotentialBasedFunctor"]

  def GetPrevAndCurState(self, observed_world):
    ego_agent = observed_world.ego_agent
    state_history = [state_action[0] for state_action in ego_agent.history[-2:]]
    prev_state, cur_state = state_history
    return prev_state, cur_state


class PotentialCenterlineFunctor(PotentialBasedFunctor):
  def __init__(self, params):
    super().__init__(params=params)
    self._params = params["PotentialCenterlineFunctor"]

  @staticmethod
  def DistancePotential(d, d_max, b=0.2):
    return 1. - (d/d_max)**b

  def DistanceToCenterline(self, observed_world, state):
    ego_agent = observed_world.ego_agent
    lane_center_line = ego_agent.road_corridor.lane_corridors[0].center_line
    dist = Distance(lane_center_line,
      Point2d(state[int(StateDefinition.X_POSITION)],
              state[int(StateDefinition.Y_POSITION)]))
    return dist

  def __call__(self, observed_world, action, eval_results):
    prev_state, cur_state = self.GetPrevAndCurState(observed_world)
    prev_dist = self.DistanceToCenterline(observed_world, prev_state)
    cur_dist = self.DistanceToCenterline(observed_world, cur_state)
    prev_pot = self.DistancePotential(
      prev_dist, self._params["MaxDist"], self._params["DistExponent"])
    cur_pot = self.DistancePotential(
      cur_dist, self._params["MaxDist"], self._params["DistExponent"])
    return self._params["Gamma"]*cur_pot - prev_pot


class PotentialVelocityFunctor(PotentialBasedFunctor):
  def __init__(self, params):
    super().__init__(params=params)
    self._params = params["PotentialVelocityFunctor"]

  @staticmethod
  def VelocityPotential(d, d_max, b=0.2):
    return 1. - (d/d_max)**b

  def __call__(self, observed_world, action, eval_results):
    prev_state, cur_state = self.GetPrevAndCurState(observed_world)
    prev_v = prev_state[int(StateDefinition.VEL_POSITION)]
    cur_v = cur_state[int(StateDefinition.VEL_POSITION)]
    prev_pot = self.VelocityPotential(
      prev_v, self._params["MaxVel"], self._params["VelExponent"])
    cur_pot = self.VelocityPotential(
      cur_v, self._params["MaxVel"], self._params["VelExponent"])
    return self._params["Gamma"]*cur_pot - prev_pot


class GeneralEvaluator:
  """Evaluator using Functors"""

  def __init__(self,
               params=ParameterServer(),
               eval_agent=None,
               bark_eval_fns=None,
               bark_ml_eval_fns=None):
    self._eval_agent = eval_agent
    self._params = params
    self._bark_eval_fns = bark_eval_fns or {
      "goal_reached" : EvaluatorGoalReached(),
      "collision" : EvaluatorCollisionEgoAgent(),
      "step_count" : EvaluatorStepCount(),
      "drivable_area" : EvaluatorDrivableArea()
    }
    self._bark_ml_eval_fns = bark_ml_eval_fns or {
      "collision_functor" : CollisionFunctor(params),
      "goal_reached_functor" : GoalFunctor(params),
      "drivable_area_functor" : DrivableAreaFunctor(params),
      "step_count_functor" : StepCountFunctor(params),
      "smoothness_functor" : SmoothnessFunctor(params),
      "max_vel_functor" : MaxVelFunctor(params),
      "pot_center_functor": PotentialCenterlineFunctor(params),
      "pot_vel_functor": PotentialVelocityFunctor(params),
    }

  def Evaluate(self, observed_world, action):
    """Returns information about the current world state."""
    eval_results = observed_world.Evaluate()
    reward = 0.
    scheduleTerminate = False

    for _, eval_fn in self._bark_ml_eval_fns.items():
      t, r, i = eval_fn(observed_world, action, eval_results)
      eval_results = {**eval_results, **i} # merge info
      reward += r # accumulate reward
      if t: # if any of the t are True -> terminal
        scheduleTerminate = True

    return reward, scheduleTerminate, eval_results

  def Reset(self, world):
    world.ClearEvaluators()
    for eval_name, eval_fn in self._bark_eval_fns.items():
      world.AddEvaluator(eval_name, eval_fn)
    return world

  def SetViewer(self, viewer):
    self._viewer = viewer