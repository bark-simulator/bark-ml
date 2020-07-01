# Copyright (c) 2020 Patrick Hart, Julian Bernhard,
# Klemens Esterle, Tobias Kessler
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
from bark.core.models.behavior import BehaviorIntersectionRuleBased
from bark.core.world.goal_definition import GoalDefinitionPolygon

from bark_ml.environments.blueprints.blueprint import Blueprint
from bark_ml.evaluators.goal_reached import GoalReached
# from bark_ml.observers.nearest_state_observer import NearestAgentsObserver
from bark_ml.behaviors.cont_behavior import BehaviorContinuousML
from bark_ml.behaviors.discrete_behavior import BehaviorDiscreteML
from bark_ml.core.observers import NearestObserver


class IntersectionLaneCorridorConfig(LaneCorridorConfig):
  def __init__(self,
               params=None,
               **kwargs):
    super(IntersectionLaneCorridorConfig, self).__init__(params, **kwargs)

  def controlled_goal(self, world):
    road_corr = world.map.GetRoadCorridor(
      self._road_ids, XodrDrivingDirection.forward)
    lane_corr = self._road_corridor.lane_corridors[0]
    # lanes are sorted by their s-value
    lanes = lane_corr.lanes
    s_val = max(lanes.keys())
    return GoalDefinitionPolygon(lanes[s_val].polygon)


class IntersectionBlueprint(Blueprint):
  def __init__(self,
               params=None,
               number_of_senarios=250,
               random_seed=0,
               ml_behavior=None,
               viewer=True):
    lane_corridors = []
    lane_corridors.append(
      IntersectionLaneCorridorConfig(params=params,
                                     source_pos=[-40, -3],
                                     sink_pos=[40, -3],
                                     behavior_model=BehaviorIntersectionRuleBased(params),
                                     min_vel=10.,
                                     max_vel=15.,
                                     ds_min=10.,
                                     ds_max=15.,
                                     s_min=5.,
                                     s_max=50.,
                                     controlled_ids=None))
    lane_corridors.append(
      IntersectionLaneCorridorConfig(params=params,
                                     source_pos=[40, 3],
                                     sink_pos=[-40, 3],
                                     behavior_model=BehaviorIntersectionRuleBased(params),
                                     min_vel=10.,
                                     max_vel=15.,
                                     ds_min=15.,
                                     ds_max=15.,
                                     s_min=5.,
                                     s_max=50.,
                                     controlled_ids=None))
    lane_corridors.append(
      IntersectionLaneCorridorConfig(params=params,
                                     source_pos=[3, -30],
                                     sink_pos=[-40, 3],
                                     behavior_model=BehaviorIntersectionRuleBased(params),
                                     min_vel=5.,
                                     max_vel=10.,
                                     ds_min=10.,
                                     ds_max=15.,
                                     s_min=10.,
                                     s_max=25.,
                                     controlled_ids=True))

    scenario_generation = \
      ConfigWithEase(
        num_scenarios=number_of_senarios,
        map_file_name=os.path.join(os.path.dirname(__file__), "../../../environments/blueprints/intersection/4way_intersection.xodr"),  # NOLINT
        random_seed=random_seed,
        params=params,
        lane_corridor_configs=lane_corridors)
    if viewer:
      viewer = MPViewer(params=params,
                        x_range=[-35, 35],
                        y_range=[-35, 35],
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


class ContinuousIntersectionBlueprint(IntersectionBlueprint):
  def __init__(self,
               params=None,
               number_of_senarios=25,
               random_seed=0,
               viewer=True):
    ml_behavior = BehaviorContinuousML(params)
    IntersectionBlueprint.__init__(self,
                                   params=params,
                                   number_of_senarios=number_of_senarios,
                                   random_seed=random_seed,
                                   ml_behavior=ml_behavior,
                                   viewer=True)


class DiscreteIntersectionBlueprint(IntersectionBlueprint):
  def __init__(self,
               params=None,
               number_of_senarios=25,
               random_seed=0,
               viewer=True):
    ml_behavior = BehaviorDiscreteML(params)
    IntersectionBlueprint.__init__(self,
                                   params=params,
                                   number_of_senarios=number_of_senarios,
                                   random_seed=random_seed,
                                   ml_behavior=ml_behavior,
                                   viewer=True)