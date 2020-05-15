# Copyright (c) 2019 Patrick Hart, Julian Bernhard,
# Klemens Esterle, Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from bark_project.behavior import BehaviorModel


class MLBehavior(BehaviorModel):
  def __init__(self,
               params=None,
               behavior=None)
    self._params = None
    self._behavior = None

  def Plan(self, observed_world, dt):
    return self._behavior.Plan(observed_world, dt)

  def Reset(self):
    pass

  @property
  def action_space(self):
    pass