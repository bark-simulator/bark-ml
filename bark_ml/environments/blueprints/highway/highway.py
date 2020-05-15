# Copyright (c) 2019 Patrick Hart, Julian Bernhard,
# Klemens Esterle, Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from bark_project.modules.runtime.commons.parameters import ParameterServer
from bark_project.modules.runtime.viewer.matplotlib_viewer import MPViewer
from bark_project.modules.runtime.scenario.scenario_generation.config_with_ease import \
  LaneCorridorConfig, ConfigWithEase

from bark_ml.environments.blueprint import Blueprint
from bark_ml.evaluators.goal_reached import GoalReached
from bark_ml.observers.nearest_state_observer import GoalReached
from bark_ml.behaviors.cont_behavior import ContinuousMLBehavior


class ContinuousHighwayBlueprint:
  def __init__(self,
               params=None
               number_of_senarios=25,
               random_seed=0):
    right_lane = LaneCorridorConfig(params=params,
                                    road_ids=[16],
                                    lane_corridor_id=1,
                                    controlled_ids=True)
    scenario_generation = \
      ConfigWithEase(
        num_scenarios=number_of_senarios,
        map_file_name="bark_ml/environments/blueprints/highway/city_highway_straight.xodr",  # NOLINT
        random_seed=random_seed,
        params=params,
        lane_corridor_configs=[right_lane])
    viewer = MPViewer(params=params,
                      x_range=[-35, 35],
                      y_range=[-35, 35],
                      follow_agent_id=True)
    dt = 0.2
    evaluator = GoalReached(params)
    observer = NearestAgentsObserver(params)
    ml_behavior = ContinuousMLBehavior(params)
    super().__init__(
      scenario_generation,
      viewer,
      dt,
      evaluator,
      ml_behavior)
