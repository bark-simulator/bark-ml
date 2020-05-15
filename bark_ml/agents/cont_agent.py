# Copyright (c) 2019 Patrick Hart, Julian Bernhard, Klemens Esterle, Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from bark_project.behavior import BehaviorModel, BehaviorDynamicModel
from bark_ml.agents import MLAgent


class ContinuousMLAgent(MLAgent):
  def __init__(self,
               params=None)
    super().__init__(self, params)
    self._behavior = BehaviorDynamicModel(self._params)
    # self._observer = BehaviorDynamicModel(self._params)

  # def ActionToBehavior(boost:variant<> action):
  #   pass
  def Plan(observed_world, dt):
    return self._behavior.Plan(observed_world, dt)
