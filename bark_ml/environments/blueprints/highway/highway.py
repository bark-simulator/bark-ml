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
from bark.runtime.scenario.scenario_generation.config_with_ease import \
  LaneCorridorConfig, ConfigWithEase
from bark.core.models.dynamic import SingleTrackModel
from bark.core.world.opendrive import XodrDrivingDirection
from bark.core.world.goal_definition import GoalDefinitionStateLimitsFrenet

from bark_ml.environments.blueprints.blueprint import Blueprint
from bark_ml.evaluators.goal_reached import GoalReached
from bark_ml.evaluators.goal_reached_guiding import GoalReachedGuiding
from bark_ml.behaviors.cont_behavior import BehaviorContinuousML
from bark_ml.behaviors.discrete_behavior import BehaviorDiscreteMotionPrimitivesML, \
        BehaviorDiscreteMacroActionsML

from bark_ml.behaviors.discrete_behavior import BehaviorDiscreteMotionPrimitivesML, \
        BehaviorDiscreteMacroActionsML
from bark_ml.core.observers import NearestObserver


class HighwayLaneCorridorConfig(LaneCorridorConfig):
  def __init__(self,
               params=None,
               **kwargs):
    super(HighwayLaneCorridorConfig, self).__init__(params, **kwargs)
  
  def goal(self, world):
    road_corr = world.map.GetRoadCorridor(
      self._road_ids, XodrDrivingDirection.forward)
    lane_corr = self._road_corridor.lane_corridors[0]
    return GoalDefinitionStateLimitsFrenet(lane_corr.center_line,
                                           (0.4, 0.4),
                                           (0.1, 0.1),
                                           (12.5, 17.5))


class HighwayBlueprint(Blueprint):
  def __init__(self,
               params=None,
               num_scenarios=250,
               random_seed=0,
               ml_behavior=None,
               viewer=True,
               mode="dense"):
    if mode == "dense":
      ds_min = 10.
      ds_max = 20.
    if mode == "medium":
      ds_min = 20.
      ds_max = 35.
    params["BehaviorIDMClassic"]["DesiredVelocity"] = 15.
    params["World"]["remove_agents_out_of_map"] = False
    left_lane = HighwayLaneCorridorConfig(params=params,
                                          road_ids=[16],
                                          lane_corridor_id=0,
                                          min_vel=12.5,
                                          max_vel=17.5,
                                          ds_min=ds_min,
                                          ds_max=ds_max,
                                          s_min=5.,
                                          s_max=200.,
                                          controlled_ids=None)
    right_lane = HighwayLaneCorridorConfig(params=params,
                                           road_ids=[16],
                                           lane_corridor_id=1,
                                           min_vel=12.5,
                                           max_vel=17.5,
                                           s_min=5.,
                                           s_max=200.,
                                           ds_min=ds_min,
                                           ds_max=ds_max,
                                           controlled_ids=True)
    scenario_generation = \
      ConfigWithEase(
        num_scenarios=num_scenarios,
        map_file_name=os.path.join(os.path.dirname(__file__), "../../../environments/blueprints/highway/city_highway_straight.xodr"),  # NOLINT
        random_seed=random_seed,
        params=params,
        lane_corridor_configs=[left_lane, right_lane])
    if viewer:
      viewer = MPViewer(params=params,
                        x_range=[-50, 50],
                        y_range=[-50, 50],
                        follow_agent_id=True)
    dt = 0.2
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


class ContinuousHighwayBlueprint(HighwayBlueprint):
  def __init__(self,
               params=None,
               num_scenarios=25,
               random_seed=0,
               viewer=True,
               mode="dense"):
    ml_behavior = BehaviorContinuousML(params)
    HighwayBlueprint.__init__(self,
                              params=params,
                              num_scenarios=num_scenarios,
                              random_seed=random_seed,
                              ml_behavior=ml_behavior,
                              viewer=True,
                              mode=mode)


class DiscreteHighwayBlueprint(HighwayBlueprint):
  def __init__(self,
               params=None,
               num_scenarios=25,
               random_seed=0,
               mode="dense"):
    ml_behavior = BehaviorDiscreteMacroActionsML(params)
    HighwayBlueprint.__init__(self,
                              params=params,
                              num_scenarios=num_scenarios,
                              random_seed=random_seed,
                              ml_behavior=ml_behavior,
                              viewer=True,
                              mode=mode)