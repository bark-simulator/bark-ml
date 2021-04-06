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
from bark.core.world.goal_definition import GoalDefinitionPolygon
from bark.core.world.goal_definition import GoalDefinitionStateLimitsFrenet
from bark.core.models.behavior import BehaviorLaneChangeRuleBased, BehaviorIDMClassic, \
  BehaviorMobilRuleBased

from bark_ml.environments.blueprints.blueprint import Blueprint
from bark_ml.evaluators.goal_reached import GoalReached

from bark_ml.evaluators.goal_reached_guiding import GoalReachedGuiding
from bark_ml.observers.nearest_state_observer import NearestAgentsObserver
from bark_ml.behaviors.cont_behavior import BehaviorContinuousML
from bark_ml.behaviors.discrete_behavior import BehaviorDiscreteMacroActionsML, \
    BehaviorDiscreteMotionPrimitivesML
from bark_ml.core.observers import NearestObserver



class MergingLaneCorridorConfig(LaneCorridorConfig):
  def __init__(self,
               params=None,
               **kwargs):
    super(MergingLaneCorridorConfig, self).__init__(params, **kwargs)

  def goal(self, world):
    road_corr = world.map.GetRoadCorridor(
      self._road_ids, XodrDrivingDirection.forward)
    lane_corr = self._road_corridor.lane_corridors[0]
    lane_polygon = lane_corr.polygon
    return GoalDefinitionStateLimitsFrenet(lane_corr.center_line,
                                           (0.9, 0.9),
                                           (0.15, 0.15),
                                           (5., 15.))


class MergingBlueprint(Blueprint):
  def __init__(self,
               params=None,
               num_scenarios=250,
               random_seed=0,
               ml_behavior=None,
               viewer=True,
               mode="dense"):
    params["BehaviorIDMClassic"]["BrakeForLaneEnd"] = True
    params["BehaviorIDMClassic"]["BrakeForLaneEndEnabledDistance"] = 100.
    params["BehaviorIDMClassic"]["BrakeForLaneEndDistanceOffset"] = 30.
    params["BehaviorIDMClassic"]["DesiredVelocity"] = 10.
    params["World"]["remove_agents_out_of_map"] = False
    # TODO: modify dense
    # ds_min and ds_max
    if mode == "dense":
      ds_min = 7.
      ds_max = 12.
    if mode == "medium":
      ds_min = 12.
      ds_max = 17.
    left_lane = MergingLaneCorridorConfig(
      params=params,
      road_ids=[0, 1],
      ds_min=ds_min,
      ds_max=ds_max,
      min_vel=9.,
      max_vel=11.,
      s_min=5.,
      s_max=45.,
      lane_corridor_id=0,
      controlled_ids=None)
    right_lane = MergingLaneCorridorConfig(
      params=params,
      road_ids=[0, 1],
      lane_corridor_id=1,
      ds_min=ds_min,
      ds_max=ds_max,
      s_min=5.,
      s_max=25.,
      min_vel=9.,
      max_vel=11.,
      behavior_model=BehaviorMobilRuleBased(params),
      controlled_ids=True)
    scenario_generation = \
      ConfigWithEase(
        num_scenarios=num_scenarios,
        map_file_name=os.path.join(os.path.dirname(__file__), "../../../environments/blueprints/merging/DR_DEU_Merging_MT_v01_centered.xodr"),  # NOLINT
        random_seed=random_seed,
        params=params,
        lane_corridor_configs=[left_lane, right_lane])
    if viewer:
      viewer = MPViewer(params=params,
                        x_range=[-20, 20],
                        y_range=[-20, 20],
                        follow_agent_id=True)
    dt = 0.2
    # evaluator = GoalReachedGuiding(params)
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


class ContinuousMergingBlueprint(MergingBlueprint):
  def __init__(self,
               params=None,
               num_scenarios=25,
               random_seed=0,
               viewer=True,
               mode="dense"):
    ml_behavior = BehaviorContinuousML(params)
    MergingBlueprint.__init__(self,
                              params=params,
                              num_scenarios=num_scenarios,
                              random_seed=random_seed,
                              ml_behavior=ml_behavior,
                              viewer=viewer,
                              mode=mode)


class DiscreteMergingBlueprint(MergingBlueprint):
  def __init__(self,
               params=None,
               num_scenarios=25,
               random_seed=0,
               viewer=True,
               mode="dense"):
    ml_behavior = BehaviorDiscreteMacroActionsML(params)
    MergingBlueprint.__init__(self,
                              params=params,
                              num_scenarios=num_scenarios,
                              random_seed=random_seed,
                              ml_behavior=ml_behavior,
                              viewer=viewer,
                              mode=mode)