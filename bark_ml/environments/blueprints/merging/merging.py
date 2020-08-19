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
# from bark_ml.observers.nearest_state_observer import NearestAgentsObserver
from bark_ml.behaviors.cont_behavior import BehaviorContinuousML
from bark_ml.behaviors.discrete_behavior import BehaviorDiscreteML
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
    return GoalDefinitionStateLimitsFrenet(lane_corr.center_line,
                                           (0.4, 0.4),
                                           (0.1, 0.1),
                                           (10., 15.))


class MergingBlueprint(Blueprint):
  def __init__(self,
               params=None,
               number_of_senarios=250,
               random_seed=0,
               ml_behavior=None,
               viewer=True):
    params["BehaviorIDMClassic"]["BrakeForLaneEnd"] = True
    params["BehaviorIDMClassic"]["BrakeForLaneEndEnabledDistance"] = 100.
    params["BehaviorIDMClassic"]["BrakeForLaneEndDistanceOffset"] = 25.
    params["BehaviorIDMClassic"]["DesiredVelocity"] = 12.5
    params["World"]["remove_agents_out_of_map"] = False
    left_lane = MergingLaneCorridorConfig(
      params=params,
      road_ids=[0, 1],
      min_vel=10.,
      max_vel=15.,
      s_min=5.,
      s_max=25.,
      lane_corridor_id=0,
      controlled_ids=None)
    right_lane = MergingLaneCorridorConfig(
      params=params,
      road_ids=[0, 1],
      lane_corridor_id=1,
      s_min=5.,
      s_max=25.,
      min_vel=8.,
      max_vel=12.,
      behavior_model=BehaviorIDMClassic(params),
      controlled_ids=True)
    scenario_generation = \
      ConfigWithEase(
        num_scenarios=number_of_senarios,
        map_file_name=os.path.join(os.path.dirname(__file__), "../../../environments/blueprints/merging/DR_DEU_Merging_MT_v01_shifted.xodr"),  # NOLINT
        random_seed=random_seed,
        params=params,
        lane_corridor_configs=[left_lane, right_lane])
    if viewer:
      viewer = MPViewer(params=params,
                        x_range=[-35, 35],
                        y_range=[-35, 35],
                        follow_agent_id=True)
    dt = 0.2
    evaluator = GoalReachedGuiding(params)
    # evaluator = GoalReached(params)
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
               number_of_senarios=25,
               random_seed=0,
               viewer=True):
    ml_behavior = BehaviorContinuousML(params)
    MergingBlueprint.__init__(self,
                              params=params,
                              number_of_senarios=number_of_senarios,
                              random_seed=random_seed,
                              ml_behavior=ml_behavior,
                              viewer=True)


class DiscreteMergingBlueprint(MergingBlueprint):
  def __init__(self,
               params=None,
               number_of_senarios=25,
               random_seed=0,
               viewer=True):
    ml_behavior = BehaviorDiscreteML(params)
    MergingBlueprint.__init__(self,
                              params=params,
                              number_of_senarios=number_of_senarios,
                              random_seed=random_seed,
                              ml_behavior=ml_behavior,
                              viewer=True)