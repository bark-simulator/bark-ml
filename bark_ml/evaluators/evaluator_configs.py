# Copyright (c) 2020 Patrick Hart, Julian Bernhard,
# Klemens Esterle, Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from bark_ml.evaluators.general_evaluator import *

class GoalReached(GeneralEvaluator):
  def __init__(self, params):
    self._params = params["ML"]["GoalReachedEvaluator"]
    super().__init__(
      params=self._params,
      bark_ml_eval_fns={
        "collision_functor" : CollisionFunctor(self._params),
        "drivable_area_functor" : DrivableAreaFunctor(self._params),
        "step_count_functor" : StepCountFunctor(self._params),
        "goal_functor": GoalFunctor(self._params)
      })


class RewardShapingGoalDistEvaluator(GeneralEvaluator):
  def __init__(self, params):
    self._params = params["ML"]["RewardShapingGoalDistEvaluator"]
    super().__init__(
      params=self._params,
      bark_ml_eval_fns={
        "collision_functor" : CollisionFunctor(self._params),
        "drivable_area_functor" : DrivableAreaFunctor(self._params),
        "step_count_functor" : StepCountFunctor(self._params),
        "pot_goal_center_functor": PotentialGoalCenterlineFunctor(self._params),
        "min_max_vel_functor" : MinMaxVelFunctor(self._params),
        "goal_functor": GoalFunctor(self._params)
      })
class TestRewardShapingGoalDistEvaluator(GeneralEvaluator):
  def __init__(self, params):
    self._params = params["ML"]["TestRewardShapingGoalDistEvaluator"]
    super().__init__(
      params=self._params,
      bark_ml_eval_fns={
        "collision_drivable_area_functor" : CollisionDrivableAreaFunctor(self._params),
        "step_count_functor" : StepCountFunctor(self._params),
        "pot_center_functor": PotentialGoalCenterlineFunctor(self._params),
        "min_max_vel_functor" : MinMaxVelFunctor(self._params),
        "goal_functor": GoalFunctor(self._params)
      })
class RewardShapingEvaluator(GeneralEvaluator):
  def __init__(self, params):
    self._params = params["ML"]["RewardShapingEvaluator"]
    super().__init__(
      params=self._params,
      bark_ml_eval_fns={
        "collision_functor" : CollisionFunctor(self._params),
        "drivable_area_functor" : DrivableAreaFunctor(self._params),
        "step_count_functor" : StepCountFunctor(self._params),
        "pot_center_functor": PotentialCenterlineFunctor(self._params),
        "pot_vel_functor": PotentialVelocityFunctor(self._params),
        "goal_functor": GoalFunctor(self._params)
      })

class SimpleSingleLaneEvaluator(GeneralEvaluator):
  def __init__(self, params):
    self._params = params["ML"]["SimpleSingleLaneEvaluator"]
    super().__init__(
      params=self._params,
      bark_ml_eval_fns={
        "collision_functor" : CollisionFunctor(self._params),
        "low_speed_goal_reached_functor" : LowSpeedGoalFunctor(self._params),
        "drivable_area_functor" : DrivableAreaFunctor(self._params),
        "step_count_functor" : StepCountFunctor(self._params),
        "pot_goal_center_functor": PotentialGoalCenterlineFunctor(self._params),
        "min_max_vel_functor" : MinMaxVelFunctor(self._params)
      })
class TestSimpleSingleLaneEvaluator(GeneralEvaluator):
  def __init__(self, params):
    self._params = params["ML"]["TestSimpleSingleLaneEvaluator"]
    super().__init__(
      params=self._params,
      bark_ml_eval_fns={
        "collision_drivable_area_functor" : CollisionDrivableAreaFunctor(self._params),
        "low_speed_goal_reached_functor" : LowSpeedGoalFunctor(self._params),
        "step_count_functor" : StepCountFunctor(self._params),
        "pot_center_functor": PotentialGoalCenterlineFunctor(self._params),
        "min_max_vel_functor" : MinMaxVelFunctor(self._params),
        "pot_goal_vel_functor" : PotentialGoalReachedVelocityFunctor(self._params)
      })
class SingleLaneEvaluator(GeneralEvaluator):
  def __init__(self, params):
    self._params = params["ML"]["SingleLaneEvaluator"]
    super().__init__(
      params=self._params,
      bark_ml_eval_fns={
        "collision_functor" : CollisionFunctor(self._params),
        "low_speed_goal_reached_functor" : LowSpeedGoalFunctor(self._params),
        "drivable_area_functor" : DrivableAreaFunctor(self._params),
        "step_count_functor" : StepCountFunctor(self._params),
        "min_max_vel_functor" : MinMaxVelFunctor(self._params),
        "pot_center_functor": PotentialCenterlineFunctor(self._params),
        "pot_vel_functor": PotentialVelocityFunctor(self._params)
      })

class SmoothnessSingleLaneEvaluator(GeneralEvaluator):
  def __init__(self, params):
    self._params = params["ML"]["SmoothnessSingleLaneEvaluator"]
    super().__init__(
      params=self._params,
      bark_ml_eval_fns={
        "collision_functor" : CollisionFunctor(self._params),
        "low_speed_goal_reached_functor" : LowSpeedGoalFunctor(self._params),
        "drivable_area_functor" : DrivableAreaFunctor(self._params),
        "smoothness_functor": SmoothnessFunctor(self._params),
        "step_count_functor" : StepCountFunctor(self._params),
        "min_max_vel_functor" : MinMaxVelFunctor(self._params),
        "pot_center_functor": PotentialCenterlineFunctor(self._params),
        "pot_vel_functor": PotentialVelocityFunctor(self._params)
      })
class EvaluatorConfigurator(GeneralEvaluator):
  def __init__(self, params):
    # add mapping of functors to keys
    self._fn_key_map = {
      "CollisionFunctor" : "collision_functor",
      "GoalFunctor" : "goal_functor",
      "LowSpeedGoalFunctor" : "low_speed_goal_reached_functor",
      "DrivableAreaFunctor" : "drivable_area_functor",
      "StepCountFunctor" : "step_count_functor",
      "SmoothnessFunctor" : "smoothness_functor",
      "MinMaxVelFunctor" : "min_max_vel_functor",
      "PotentialCenterlineFunctor": "pot_center_functor",
      "PotentialVelocityFunctor": "pot_vel_functor",
      "PotentialGoalSwitchVelocityFunctor": "pot_goal_switch_vel_functor",
      "PotentialGoalCenterlineFunctor": "pot_goal_center_functor",
      "StateActionLoggingFunctor": "state_action_logging_functor",
      "CollisionDrivableAreaFunctor" : "collision_drivable_area_functor",
      "PotentialGoalReachedVelocityFunctor": "pot_goal_vel_functor"
    }
    self._params = params["ML"]["EvaluatorConfigurator"]["EvaluatorConfigs"]["FunctorConfigs"]
    config_params_dict = self._params.ConvertToDict()
    # initialize functor and functorweights dicts
    eval_fns = {}
    # get values for each item
    for key in config_params_dict.keys():
      matched_functor_key = self._fn_key_map[key]
      eval_fns[matched_functor_key]= eval("{}(self._params)".format(key))
    super().__init__(params=self._params, bark_ml_eval_fns=eval_fns)

  def addKeyFunctorPair(self,functor_name,key_name):
    self._fn_key_map[functor_name] = key_name