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


class Functor:

  def Reset(self):
    pass

class CollisionFunctor(Functor):
  def __init__(self, params):
    self._params = params["CollisionFunctor"]

  def __call__(self, observed_world, action, eval_results):
    if eval_results["collision"]:
      return True, self._params["CollisionReward", "", -1.], {}
    return False, 0, {}


class GoalFunctor(Functor):
  def __init__(self, params):
    self._params = params["GoalFunctor"]

  def __call__(self, observed_world, action, eval_results):
    if eval_results["goal_reached"]:
      return True, self._params["GoalReward", "", 1.], {}
    return False, 0, {}

class DrivableAreaFunctor(Functor):
  def __init__(self, params):
    self._params = params["DrivableAreaFunctor"]

  def __call__(self, observed_world, action, eval_results):
    if eval_results["drivable_area"]:
      return True, self._params["DrivableAreaReward", "", -1.], {}
    return False, 0, {}


class StepCountFunctor(Functor):
  def __init__(self, params):
    self._params = params["StepCountFunctor"]

  def __call__(self, observed_world, action, eval_results):
    if eval_results["step_count"] > self._params[
      "MaxStepCount", "", 220]:
      return True, self._params["StepCountReward", "", 0.], {}
    return False, 0, {}


class MinMaxVelFunctor(Functor):
  def __init__(self, params):
    self._params = params["MinMaxVelFunctor"]

  def __call__(self, observed_world, action, eval_results):
    ego_agent = observed_world.ego_agent
    ego_vel = ego_agent.state[int(StateDefinition.VEL_POSITION)]
    if ego_vel > self._params["MaxVel", "", 25.] or \
      ego_vel < self._params["MinVel", "", 0.]:
      return True, self._params["MaxVelViolationReward", "", -1.], {}
    return False, 0, {}


class SmoothnessFunctor(Functor):
  def __init__(self, params):
    self._params = params["SmoothnessFunctor"]

  def __call__(self, observed_world, action, eval_results):
    acc = action[0]
    delta_dot = action[1]
    reward = 0.
    reward += self._params["AccWeight", "", 0.5]*acc*acc
    reward += self._params["SteeringRateWeight", "", 0.5]*delta_dot*delta_dot
    return False, 0, {}


class PotentialBasedFunctor(Functor):
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
  def DistancePotential(d, d_max, b):
    return 1. - (d/d_max)**b

  def DistanceToCenterline(self, observed_world, state):
    ego_agent = observed_world.ego_agent
    lane_center_line = ego_agent.road_corridor.lane_corridors[0].center_line
    dist = Distance(lane_center_line,
      Point2d(state[int(StateDefinition.X_POSITION)],
              state[int(StateDefinition.Y_POSITION)]))
    return dist

  def __call__(self, observed_world, action, eval_results):
    hist = observed_world.ego_agent.history
    if len(hist) >= 2:
      prev_state, cur_state = self.GetPrevAndCurState(observed_world)
      prev_dist = self.DistanceToCenterline(observed_world, prev_state)
      cur_dist = self.DistanceToCenterline(observed_world, cur_state)
      prev_pot = self.DistancePotential(
        prev_dist, self._params["MaxDist", "", 100.], self._params["DistExponent", "", 0.2])
      cur_pot = self.DistancePotential(
        cur_dist, self._params["MaxDist", "", 100.], self._params["DistExponent", "", 0.2])
      return False, self._params["Gamma", "", 0.99]*cur_pot - prev_pot, {}
    return False, 0, {}


class PotentialVelocityFunctor(PotentialBasedFunctor):
  def __init__(self, params):
    super().__init__(params=params)
    self._params = params["PotentialVelocityFunctor"]

  @staticmethod
  def VelocityPotential(v, v_des, v_dev_max, a):
    return 1. - (np.sqrt((v-v_des)**2)/v_dev_max)**a

  def __call__(self, observed_world, action, eval_results):
    hist = observed_world.ego_agent.history
    if len(hist) >= 2:
      prev_state, cur_state = self.GetPrevAndCurState(observed_world)
      prev_v = prev_state[int(StateDefinition.VEL_POSITION)]
      cur_v = cur_state[int(StateDefinition.VEL_POSITION)]
      prev_pot = self.VelocityPotential(
        prev_v, self._params["DesiredVel", "", 4.],
        self._params["MaxVel", "", 20.], self._params["VelExponent", "", 0.2])
      cur_pot = self.VelocityPotential(
        cur_v,  self._params["DesiredVel", "", 4.],
        self._params["MaxVel", "", 100.], self._params["VelExponent", "", 0.2])
      return False, self._params["Gamma", "", 0.99]*cur_pot - prev_pot, {}
    return False, 0, {}

class PotentialGoalSwitchVelocityFunctor(PotentialBasedFunctor):
  def __init__(self, params):
    super().__init__(params=params)
    self._params = params["PotentialGoalSwitchVelocityFunctor"]

  @staticmethod
  def VelocityPotential(v, v_des, v_dev_max, a):
    return 1. - (np.sqrt((v-v_des)**2)/v_dev_max)**a

  def __call__(self, observed_world, action, eval_results):
    hist = observed_world.ego_agent.history
    desired_vel = self._params["DesiredVel", "", 4.]
    if eval_results["goal_reached"]:
      desired_vel = 0.
    if len(hist) >= 2:
      prev_state, cur_state = self.GetPrevAndCurState(observed_world)
      prev_v = prev_state[int(StateDefinition.VEL_POSITION)]
      cur_v = cur_state[int(StateDefinition.VEL_POSITION)]
      prev_pot = self.VelocityPotential(
        prev_v, desired_vel, self._params["MaxVel", "", 100.],
        self._params["VelExponent", "", 0.2])
      cur_pot = self.VelocityPotential(
        cur_v,  desired_vel, self._params["MaxVel", "", 100.],
        self._params["VelExponent", "", 0.2])
      return False, self._params["Gamma", "", 0.99]*cur_pot - prev_pot, {}
    return False, 0, {}


class LowSpeedGoalFunctor(Functor):
  def __init__(self, params):
    self._params = params["LowSpeedGoalFunctor"]

  def __call__(self, observed_world, action, eval_results):
    ego_agent = observed_world.ego_agent
    ego_vel = ego_agent.state[int(StateDefinition.VEL_POSITION)]
    if eval_results["goal_reached"] and \
      ego_vel < self._params["MaxSpeed", "", 1.]:
      return True, self._params["GoalReward", "", 1.], {"low_speed_goal_reached": True}
    return False, 0, {"low_speed_goal_reached": False}


class GeneralEvaluator:
  """Evaluator using Functors"""

  def __init__(self,
               params=ParameterServer(),
               eval_agent=None,
               bark_eval_fns=None,
               bark_ml_eval_fns=None):
    self._eval_agent = eval_agent
    self._params = params["ML"]["GeneralEvaluator"]
    self._bark_eval_fns = bark_eval_fns or {
      "goal_reached" : lambda: EvaluatorGoalReached(),
      "collision" : lambda: EvaluatorCollisionEgoAgent(),
      "step_count" : lambda: EvaluatorStepCount(),
      "drivable_area" : lambda: EvaluatorDrivableArea()
    }
    self._bark_ml_eval_fns = bark_ml_eval_fns or {
      "collision_functor" : CollisionFunctor(self._params),
      # "goal_reached_functor" : GoalFunctor(self._params),
      "low_speed_goal_reached_functor" : LowSpeedGoalFunctor(params),
      "drivable_area_functor" : DrivableAreaFunctor(self._params),
      "step_count_functor" : StepCountFunctor(self._params),
      # "smoothness_functor" : SmoothnessFunctor(params),
      "min_max_vel_functor" : MinMaxVelFunctor(self._params),
      "pot_center_functor": PotentialCenterlineFunctor(self._params),
      "pot_goal_switch_vel_functor": PotentialVelocityFunctor(self._params)
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
      world.AddEvaluator(eval_name, eval_fn())
    for _, eval_func in self._bark_ml_eval_fns.items():
      eval_func.Reset()
    return world

  def SetViewer(self, viewer):
    self._viewer = viewer
