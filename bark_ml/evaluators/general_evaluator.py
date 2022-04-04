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
from bark.core.geometry import Point2d, Within, Distance


class Functor:
  def __init__(self, params):
    self._weight = params["RewardWeight","weight for reward calculation", 1.0]
  def Reset(self):
    pass

  @property
  def weight(self):
   return self._weight

  def WeightedReward(self, reward_):
    return self._weight * reward_
    
  @staticmethod
  def in_goal_area(observed_world):
    ego_agent = observed_world.ego_agent
    goal_shape_ = ego_agent.goal_definition.goal_shape
    ego_pos_ = Point2d(ego_agent.state[int(StateDefinition.X_POSITION)],ego_agent.state[int(StateDefinition.Y_POSITION)])
    return Within(ego_pos_,goal_shape_)

class CollisionFunctor(Functor):
  def __init__(self, params):
    self._params = params["CollisionFunctor"]
    super().__init__(params=self._params)

  def __call__(self, observed_world, action, eval_results):
    if eval_results["collision"]:
      return True, self.WeightedReward(self._params["CollisionReward", "", -1.]), {}
    return False, 0, {}


class GoalFunctor(Functor):
  def __init__(self, params):
    self._params = params["GoalFunctor"]
    super().__init__(params=self._params)

  def __call__(self, observed_world, action, eval_results):
    goal_terminate= False
    if eval_results["goal_reached"]:
      goal_terminate = True
      if eval_results["drivable_area"] or eval_results["collision"]:
        return goal_terminate, 0, {}
      return goal_terminate, self.WeightedReward(self._params["GoalReward", "", 1.]), {}
    return False, 0, {}

class DrivableAreaFunctor(Functor):
  def __init__(self, params):
    self._params = params["DrivableAreaFunctor"]
    super().__init__(params=self._params)

  def __call__(self, observed_world, action, eval_results):
    if eval_results["drivable_area"]:
      return True, self.WeightedReward(self._params["DrivableAreaReward", "", -1.]), {}
    return False, 0, {}

class CollisionDrivableAreaFunctor(Functor):
  def __init__(self, params):
    
    self._params = params["CollisionDrivableAreaFunctor"]
    super().__init__(params=self._params)

  def __call__(self, observed_world, action, eval_results):
    if eval_results["drivable_area"] or eval_results["collision"]:
      return True, self.WeightedReward(self._params["FailReward", "", -1.]), {}
    return False, 0, {}

class StepCountFunctor(Functor):
  def __init__(self, params):
    
    self._params = params["StepCountFunctor"]
    super().__init__(params=self._params)

  def __call__(self, observed_world, action, eval_results):
    if eval_results["step_count"] > self._params[
      "MaxStepCount", "", 220]:
      return True, self.WeightedReward(self._params["StepCountReward", "", 0.]), {}
    return False, 0, {}


class MinMaxVelFunctor(Functor):
  def __init__(self, params):
    
    self._params = params["MinMaxVelFunctor"]
    super().__init__(params=self._params)

  def __call__(self, observed_world, action, eval_results):
    ego_agent = observed_world.ego_agent
    ego_vel = ego_agent.state[int(StateDefinition.VEL_POSITION)]
    if ego_vel > self._params["MaxVel", "", 25.] or \
      ego_vel < self._params["MinVel", "", 0.]:
      return False, self.WeightedReward(self._params["MaxVelViolationReward", "", -1.]), {}
    return False, 0, {}


class SmoothnessFunctor(Functor):
  def __init__(self, params):
    
    self._params = params["SmoothnessFunctor"]
    super().__init__(params=self._params)

  def __call__(self, observed_world, action, eval_results):
    acc = action[0]
    delta_dot = action[1]
    reward = 0.
    reward -= self._params["AccWeight", "", 0.1]*acc*acc
    reward -= self._params["SteeringRateWeight", "", 0.05]*delta_dot*delta_dot
    return False, self.WeightedReward(reward), {}


class PotentialBasedFunctor(Functor):
  def __init__(self, params):
    self._params = params["PotentialBasedFunctor"]
    super().__init__(params=params)

  def GetPrevAndCurState(self, observed_world):
    ego_agent = observed_world.ego_agent
    state_history = [state_action[0] for state_action in ego_agent.history[-2:]]
    prev_state, cur_state = state_history
    return prev_state, cur_state


class PotentialCenterlineFunctor(PotentialBasedFunctor):
  def __init__(self, params):
    self._params = params["PotentialCenterlineFunctor"]
    super().__init__(params=self._params)
    

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
        prev_dist, self._params["MaxDist", "", 100.],
        self._params["DistExponent", "", 0.2])
      cur_pot = self.DistancePotential(
        cur_dist, self._params["MaxDist", "", 100.],
        self._params["DistExponent", "", 0.2])
      return False, self.WeightedReward(self._params["Gamma", "", 0.99]*cur_pot - prev_pot), {}
    return False, 0, {}

class PotentialGoalCenterlineFunctor(PotentialBasedFunctor):
  def __init__(self, params):
    self._params = params["PotentialGoalCenterlineFunctor"]
    super().__init__(params=self._params)
    

  @staticmethod
  def DistancePotential(d, d_max, b):
    return 1. - (d/d_max)**b

  def DistanceToCenterline(self, observed_world, state):
    ego_agent = observed_world.ego_agent
    goal_center_line = ego_agent.goal_definition.center_line
    dist = Distance(goal_center_line,
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
        prev_dist, self._params["MaxDist", "", 100.],
        self._params["DistExponent", "", 0.2])
      cur_pot = self.DistancePotential(
        cur_dist, self._params["MaxDist", "", 100.],
        self._params["DistExponent", "", 0.2])
      return False, self.WeightedReward(self._params["Gamma", "", 0.99]*cur_pot - prev_pot), {}
    return False, 0, {}

class PotentialVelocityFunctor(PotentialBasedFunctor):
  def __init__(self, params):
    self._params = params["PotentialVelocityFunctor"]
    super().__init__(params=self._params)
    

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
        self._params["MaxVel", "", 100.], self._params["VelExponent", "", 0.2])
      cur_pot = self.VelocityPotential(
        cur_v,  self._params["DesiredVel", "", 4.],
        self._params["MaxVel", "", 100.], self._params["VelExponent", "", 0.2])
      return False, self.WeightedReward(self._params["Gamma", "", 0.99]*cur_pot - prev_pot), {}
    return False, 0, {}

class PotentialGoalSwitchVelocityFunctor(PotentialBasedFunctor):
  def __init__(self, params):
    self._params = params["PotentialGoalSwitchVelocityFunctor"]
    super().__init__(params=self._params)
    

  @staticmethod
  def VelocityPotential(v, v_des, v_dev_max, a):
    return 1. - (np.sqrt((v-v_des)**2)/v_dev_max)**a

  def __call__(self, observed_world, action, eval_results):
    hist = observed_world.ego_agent.history
    desired_vel = self._params["DesiredVel", "", 4.]
    if self.in_goal_area(observed_world):
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
      return False, self.WeightedReward(self._params["Gamma", "", 0.99]*cur_pot - prev_pot), {}
    return False, 0, {}

class PotentialGoalReachedVelocityFunctor(PotentialBasedFunctor):
  def __init__(self, params):
    self._params = params["PotentialGoalReachedVelocityFunctor"]
    super().__init__(params=self._params)
    
  @staticmethod
  def VelocityPotential(v, v_des, v_dev_max, a):
    return 1. - (np.sqrt((v-v_des)**2)/v_dev_max)**a

  def __call__(self, observed_world, action, eval_results):
    desired_vel = self._params["DesiredVel", "", 0.]
    if self.in_goal_area(observed_world):
      hist = observed_world.ego_agent.history
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
    super().__init__(params=self._params) 
    self._in_goal_area = False

  # @staticmethod
  # def VelocityPotential(v, v_des, v_dev_max, a):
  #   return -(np.sqrt((v-v_des)**2)/v_dev_max)**a

  # def __call__(self, observed_world, action, eval_results):
  #   cur_v = observed_world.ego_agent.history[-1][0][int(StateDefinition.VEL_POSITION)]
  #   desired_vel = self._params["DesiredVel", "", 0.]
  #   cur_pot = self.VelocityPotential(
  #       cur_v,  desired_vel, self._params["MaxVel", "", 100.],
  #       self._params["VelExponent", "", 0.8])
  #   if eval_results["goal_reached"] and (not(eval_results["drivable_area"] or eval_results["collision"])):
  #     return True, self._params["GoalReward", "", 1.]+ cur_pot, {"low_speed_goal_reached": True}
  #   return False, 0, {"low_speed_goal_reached": False}
      
  def __call__(self, observed_world, action, eval_results):
    ego_agent = observed_world.ego_agent
    ego_vel = ego_agent.state[int(StateDefinition.VEL_POSITION)]
    if eval_results["goal_reached"]:
      self._in_goal_area = True
      if ego_vel < self._params["MaxSpeed", "", 1.0] and \
        (not(eval_results["drivable_area"] or eval_results["collision"])):
        self._in_goal_area = False
        return True, self.WeightedReward(self._params["GoalReward", "", 1.]), {"goal_reached": True}

    if self._in_goal_area and not eval_results["goal_reached"]:
      self._in_goal_area = False
      return True,0, {"goal_reached": False}

    return False, 0, {"goal_reached": False}

  # def __call__(self, observed_world, action, eval_results):
  #   ego_agent = observed_world.ego_agent
  #   ego_vel = ego_agent.state[int(StateDefinition.VEL_POSITION)]
  #   if eval_results["goal_reached"] and \
  #     (not(eval_results["drivable_area"] or eval_results["collision"])) and \
  #     ego_vel < self._params["MaxSpeed", "", 1.0]:
  #     return True, self.WeightedReward(self._params["GoalReward", "", 1.]), {"goal_reached": True}
  #   return False, 0, {"goal_reached": False}


class StateActionLoggingFunctor(Functor):
  def __init__(self, params):
    self._params = params["StateActionLoggingFunctor"]
    super().__init__(params=self._params)
    

  def __call__(self, observed_world, action, eval_results):
    ego_agent = observed_world.ego_agent
    t = ego_agent.state[int(StateDefinition.TIME_POSITION)]
    x = ego_agent.state[int(StateDefinition.X_POSITION)]
    y = ego_agent.state[int(StateDefinition.Y_POSITION)]
    theta = ego_agent.state[int(StateDefinition.THETA_POSITION)]
    vel = ego_agent.state[int(StateDefinition.VEL_POSITION)]
    acc = action[0]
    delta = action[1]

    return False, 0, {"time": t, "x": x, "y": y, "theta": theta,
                      "vel": vel, "acc": acc, "delta": delta}

# TODO: extract t -> (x, y), t -> v && min/max acc, delta, v, theta
# TODO: MIN/MAX functor for defined state value
# TODO: Deviation functor for state-difference (desired vel. and x,y)

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
      "goal_functor" : GoalFunctor(self._params),
      "low_speed_goal_reached_functor" : LowSpeedGoalFunctor(self._params),
      "drivable_area_functor" : DrivableAreaFunctor(self._params),
      "step_count_functor" : StepCountFunctor(self._params),
      "smoothness_functor" : SmoothnessFunctor(self._params),
      "min_max_vel_functor" : MinMaxVelFunctor(self._params),
      # "pot_center_functor": PotentialCenterlineFunctor(self._params),
      # "pot_vel_functor": PotentialVelocityFunctor(self._params),
      "pot_goal_center_functor": PotentialGoalCenterlineFunctor(self._params),
      # "pot_goal_switch_vel_functor": PotentialGoalSwitchVelocityFunctor(self._params)
      # "state_action_logging_functor": StateActionLoggingFunctor(self._params)
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
