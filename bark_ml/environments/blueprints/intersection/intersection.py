# Copyright (c) 2020 Patrick Hart, Julian Bernhard,
# Klemens Esterle, Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from bark_project.modules.runtime.commons.parameters import ParameterServer
from bark_project.modules.runtime.viewer.matplotlib_viewer import MPViewer
from bark_project.modules.runtime.scenario.scenario_generation.config_with_ease import \
  LaneCorridorConfig, ConfigWithEase
from bark.models.dynamic import SingleTrackModel
from bark.world.opendrive import XodrDrivingDirection
from bark.world.goal_definition import GoalDefinitionPolygon
from bark.models.behavior import BehaviorIntersectionRuleBased

from bark_ml.environments.blueprints.blueprint import Blueprint
from bark_ml.evaluators.goal_reached import GoalReached
from bark_ml.observers.nearest_state_observer import NearestAgentsObserver
from bark_ml.behaviors.cont_behavior import BehaviorContinuousML
from bark_ml.behaviors.discrete_behavior import BehaviorDiscreteML


class IntersectionLaneCorridorConfig(LaneCorridorConfig):
  def __init__(self,
               params=None,
               **kwargs):
    super(IntersectionLaneCorridorConfig, self).__init__(params, **kwargs)


class IntersectionBlueprint(Blueprint):
  def __init__(self,
               params=None,
               number_of_senarios=250,
               random_seed=0,
               ml_behavior=None):
    lane_corridors = []
    lane_corridors.append(
      LaneCorridorConfig(params=params,
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
      LaneCorridorConfig(params=params,
                        source_pos=[40, 3],
                        sink_pos=[-40, 3],
                        behavior_model=BehaviorIntersectionRuleBased(params),
                        min_vel=10.,
                        max_vel=15.,
                        ds_min=5.,
                        ds_max=10.,
                        s_min=5.,
                        s_max=50.,
                        controlled_ids=None))
    lane_corridors.append(
      LaneCorridorConfig(params=params,
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
        map_file_name="bark_ml/environments/blueprints/intersection/4way_intersection.xodr",  # NOLINT
        random_seed=random_seed,
        params=params,
        lane_corridor_configs=lane_corridors)
    viewer = MPViewer(params=params,
                      x_range=[-35, 35],
                      y_range=[-35, 35],
                      follow_agent_id=True)
    dt = 0.2
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


class ContinuousIntersectionBlueprint(IntersectionBlueprint):
  def __init__(self,
               params=None,
               number_of_senarios=25,
               random_seed=0):
    ml_behavior = BehaviorContinuousML(params)
    IntersectionBlueprint.__init__(self,
                                   params=params,
                                   number_of_senarios=number_of_senarios,
                                   random_seed=random_seed,
                                   ml_behavior=ml_behavior)


class DiscreteIntersectionBlueprint(IntersectionBlueprint):
  def __init__(self,
               params=None,
               number_of_senarios=25,
               random_seed=0):
    dynamic_model = SingleTrackModel(params)
    ml_behavior = BehaviorDiscreteML(dynamic_model, params)
    IntersectionBlueprint.__init__(self,
                                   params=params,
                                   number_of_senarios=number_of_senarios,
                                   random_seed=random_seed,
                                   ml_behavior=ml_behavior)