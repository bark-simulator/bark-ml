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
