# Copyright (c) 2019 Patrick Hart, Julian Bernhard, Klemens Esterle, Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


from bark_project.behavior import BehaviorModel

class MLAgent(BehaviorModel):
  def __init__(self,
               params=None,
               behavior=None,
               observer=None)
    self._params = None
    self._observer = None
    self._behavior = None

  # def ActionToBehavior(boost:variant<> action):
  #   pass

  def Plan(observed_world, dt):
    return self._behavior.Plan(observed_world, dt)
