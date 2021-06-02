# Copyright (c) 2020 fortiss GmbH
#
# Authors: Patrick Hart, Julian Bernhard, Klemens Esterle, and
# Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from abc import ABC
from bark.runtime.commons.parameters import ParameterServer


class BaseEvaluator(ABC):
  """Evaluates the state of the environment, e.g., if
  a collision has occured."""

  def __init__(self,
               params=ParameterServer()):
    self._params = params
    self._viewer = None

  def Evaluate(self, observed_world, action):
    """Evaluates the observed world."""
    eval_results, reward, done = None, 0., False
    eval_results = observed_world.Evaluate()
    reward, done, eval_results = self._evaluate(
      observed_world, eval_results, action)
    return reward, done, eval_results

  def Reset(self, world):
    world.ClearEvaluators()
    evaluators = self._add_evaluators()
    for key, evaluator in evaluators.items():
      world.AddEvaluator(key, evaluator)
    return world

  def SetViewer(self, viewer):
    self._viewer = viewer