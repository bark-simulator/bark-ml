# Copyright (c) 2020 fortiss GmbH
#
# Authors: Patrick Hart, Julian Bernhard, Klemens Esterle, and
# Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import os
from bark.runtime.commons.parameters import ParameterServer
from bark.runtime.viewer.matplotlib_viewer import MPViewer
from bark.runtime.scenario.scenario_generation.configurable_scenario_generation import \
  ConfigurableScenarioGeneration
from bark.core.models.dynamic import SingleTrackModel
from bark.core.world.opendrive import XodrDrivingDirection
from bark.core.world.goal_definition import GoalDefinitionPolygon
from bark.core.world.goal_definition import GoalDefinitionStateLimitsFrenet
from bark.core.models.behavior import BehaviorLaneChangeRuleBased, BehaviorIDMClassic, \
  BehaviorMobilRuleBased

from bark_ml.environments.blueprints.blueprint import Blueprint
from bark_ml.evaluators.goal_reached import GoalReached
from bark_ml.evaluators.goal_reached_guiding import GoalReachedGuiding
from bark_ml.observers.nearest_state_observer import NearestAgentsObserver
from bark_ml.behaviors.cont_behavior import BehaviorContinuousML
from bark_ml.behaviors.discrete_behavior import BehaviorDiscreteML
from bark_ml.core.observers import NearestObserver



class ConfigurableScenarioBlueprint(Blueprint):
  def __init__(self,
               params=None,
               ml_behavior=None,
               viewer=True):
    # NOTE: scneario number is wrong
    scenario_generation = ConfigurableScenarioGeneration(
      num_scenarios=2500,
      params=params)
    if viewer:
      viewer = MPViewer(params=params,
                        x_range=[-35, 35],
                        y_range=[-35, 35],
                        follow_agent_id=True)
    dt = 0.2
    # NOTE: evaluator and observer could be overwritten
    evaluator = GoalReached(params)
    observer = NearestObserver(params)
    ml_behavior = ml_behavior
    super().__init__(
      scenario_generation=scenario_generation,
      viewer=viewer,
      dt=dt,
      evaluator=evaluator,
      observer=observer,
      ml_behavior=ml_behavior)