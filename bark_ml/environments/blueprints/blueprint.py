# Copyright (c) 2020 fortiss GmbH
#
# Authors: Patrick Hart, Julian Bernhard, Klemens Esterle, and
# Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


class Blueprint:
  """ Blueprints define the environment.

  The `scenario_generation` generates the scenarios.

  The `evaluator` is used to generate the reward signal and to
  determine when the episode is terminal.

  The `observer` converts the semantic environment into a
  machine learnign friendly observation.

  The `viewer` is used for rendering purposes.

  The `ml_behavior` is the machine learning agent.
  """

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
