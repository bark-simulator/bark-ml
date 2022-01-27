# Copyright (c) 2020 fortiss GmbH
#
# Authors: Patrick Hart, Julian Bernhard, Klemens Esterle, and
# Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from bark.runtime.viewer.matplotlib_viewer import MPViewer
from bark.runtime.scenario.scenario_generation.configurable_scenario_generation import \
  ConfigurableScenarioGeneration

from bark_ml.environments.blueprints.blueprint import Blueprint
from bark_ml.evaluators.evaluator_configs import GoalReached
from bark_ml.behaviors.discrete_behavior import *
from bark_ml.observers.nearest_state_observer import NearestAgentsObserver


class ConfigurableScenarioBlueprint(Blueprint):
  """Blueprint using the configurable scenario generation of BARK.

     Can be an arbitrary scenario.
  """
  def __init__(self,
               params=None,
               ml_behavior=None,
               num_scenarios=100,
               viewer=True):
    # NOTE: scneario number is wrong
    scenario_generation = ConfigurableScenarioGeneration(
      num_scenarios=num_scenarios,
      params=params)
    if viewer:
      viewer = MPViewer(params=params,
                        x_range=[-35, 35],
                        y_range=[-35, 35],
                        follow_agent_id=True)
    dt = 0.2
    # NOTE: evaluator and observer could be overwritten
    evaluator = GoalReached(params)
    observer = NearestAgentsObserver(params)
    ml_behavior = ml_behavior
    super().__init__(
      scenario_generation=scenario_generation,
      viewer=viewer,
      dt=dt,
      evaluator=evaluator,
      observer=observer,
      ml_behavior=ml_behavior)