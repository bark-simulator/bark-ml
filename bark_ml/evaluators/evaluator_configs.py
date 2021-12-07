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
        "pot_vel_functor": PotentialVelocityFunctor(self._params)
      })

class SimpleSingleLaneEvaluator(GeneralEvaluator):
  def __init__(self, params):
    self._params = params["ML"]["SimpleSingleLaneEvaluator"]
    super().__init__(
      params=self._params,
      bark_ml_eval_fns={
        "collision_functor" : CollisionFunctor(self._params),
        "low_speed_goal_reached_functor" : LowSpeedGoalFunctor(params),
        "drivable_area_functor" : DrivableAreaFunctor(self._params),
        "step_count_functor" : StepCountFunctor(self._params),
        "min_max_vel_functor" : MinMaxVelFunctor(self._params),
      })

class SingleLaneEvaluator(GeneralEvaluator):
  def __init__(self, params):
    self._params = params["ML"]["SingleLaneEvaluator"]
    super().__init__(
      params=self._params,
      bark_ml_eval_fns={
        "collision_functor" : CollisionFunctor(self._params),
        "low_speed_goal_reached_functor" : LowSpeedGoalFunctor(params),
        "drivable_area_functor" : DrivableAreaFunctor(self._params),
        "step_count_functor" : StepCountFunctor(self._params),
        "min_max_vel_functor" : MinMaxVelFunctor(self._params),
        "pot_center_functor": PotentialCenterlineFunctor(self._params),
        "pot_goal_switch_vel_functor": PotentialVelocityFunctor(self._params)
      })

class SmoothnessSingleLaneEvaluator(GeneralEvaluator):
  def __init__(self, params):
    self._params = params["ML"]["SmoothnessSingleLaneEvaluator"]
    super().__init__(
      params=self._params,
      bark_ml_eval_fns={
        "collision_functor" : CollisionFunctor(self._params),
        "low_speed_goal_reached_functor" : LowSpeedGoalFunctor(params),
        "drivable_area_functor" : DrivableAreaFunctor(self._params),
        "smoothness_functor": SmoothnessFunctor(self._params),
        "step_count_functor" : StepCountFunctor(self._params),
        "min_max_vel_functor" : MinMaxVelFunctor(self._params),
        "pot_center_functor": PotentialCenterlineFunctor(self._params),
        "pot_goal_switch_vel_functor": PotentialVelocityFunctor(self._params)
      })