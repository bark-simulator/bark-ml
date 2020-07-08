# Copyright (c) 2020 Patrick Hart, Julian Bernhard,
# Klemens Esterle, Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


class Blueprint:
  def __init__(self,
               scenario_generation=None,
               viewer=None,
               dt=None,
               evaluator=None,
               observer=None,
               ml_behavior=None):
    self._scenario_generation = scenario_generation
    self._viewer = viewer
    self._dt = dt
    self._evaluator = evaluator
    self._observer = observer
    self._ml_behavior = ml_behavior