# Copyright (c) 2020 fortiss GmbH
#
# Authors: Patrick Hart, Julian Bernhard, Klemens Esterle, and
# Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import os
from bark.runtime.viewer.buffered_mp_viewer import BufferedMPViewer
from bark.runtime.scenario.scenario_generation.config_with_ease import \
  LaneCorridorConfig, ConfigWithEase
from bark.core.world.opendrive import XodrDrivingDirection
from bark.core.world.goal_definition import GoalDefinitionPolygon
from bark.core.models.behavior import BehaviorMobilRuleBased

from bark_ml.environments.blueprints.blueprint import Blueprint
from bark_ml.behaviors.cont_behavior import BehaviorContinuousML
from bark_ml.behaviors.discrete_behavior import BehaviorDiscreteMacroActionsML
from bark_ml.observers.nearest_state_observer import NearestAgentsObserver
from bark_ml.evaluators.evaluator_configs import RewardShapingEvaluator

class IntersectionLaneCorridorConfig(LaneCorridorConfig):
  """Configures the a single lane, e.g., the goal.
  """

  def __init__(self,
               params=None,
               **kwargs):
    super(IntersectionLaneCorridorConfig, self).__init__(params, **kwargs)

  def controlled_goal(self, world):
    world.map.GetRoadCorridor(
      self._road_ids, XodrDrivingDirection.forward)
    lane_corr = self._road_corridor.lane_corridors[0]
    # lanes are sorted by their s-value
    lanes = lane_corr.lanes
    s_val = max(lanes.keys())
    return GoalDefinitionPolygon(lanes[s_val].polygon)


class IntersectionBlueprint(Blueprint):
  """The intersection blueprint sets up a merging scenario with initial
  conditions.

  The ego vehicle start on the right lane on the road in the bottom.
  The polygonal goal is placed on the upper left road on the right lane.
  """

  def __init__(self,
               params=None,
               num_scenarios=250,
               random_seed=0,
               ml_behavior=None,
               viewer=True):
    lane_corridors = []
    lane_corridors.append(
      IntersectionLaneCorridorConfig(params=params,
                                     source_pos=[-40, -3],
                                     sink_pos=[40, -3],
                                     behavior_model=BehaviorMobilRuleBased(params),
                                     min_vel=10.,
                                     max_vel=12.,
                                     ds_min=10.,
                                     ds_max=25.,
                                     s_min=5.,
                                     s_max=50.,
                                     controlled_ids=None))
    lane_corridors.append(
      IntersectionLaneCorridorConfig(params=params,
                                     source_pos=[40, 3],
                                     sink_pos=[-40, 3],
                                     behavior_model=BehaviorMobilRuleBased(params),
                                     min_vel=10.,
                                     max_vel=12.,
                                     ds_min=15.,
                                     ds_max=25.,
                                     s_min=5.,
                                     s_max=50.,
                                     controlled_ids=None))
    lane_corridors.append(
      IntersectionLaneCorridorConfig(params=params,
                                     source_pos=[3, -30],
                                     sink_pos=[-40, 3],
                                     behavior_model=BehaviorMobilRuleBased(params),
                                     min_vel=5.,
                                     max_vel=10.,
                                     ds_min=40.,
                                     ds_max=40.,
                                     s_min=30.,
                                     s_max=40.,
                                     controlled_ids=True))

    scenario_generation = \
      ConfigWithEase(
        num_scenarios=num_scenarios,
        map_file_name=os.path.join(os.path.dirname(__file__), "../../../environments/blueprints/intersection/4way_intersection.xodr"),  # pylint: disable=unused-import
        random_seed=random_seed,
        params=params,
        lane_corridor_configs=lane_corridors)
    if viewer:
      viewer = BufferedMPViewer(
        params=params,
        x_range=[-30, 30],
        y_range=[-30, 30],
        follow_agent_id=True)
    dt = 0.2

    params["ML"]["RewardShapingEvaluator"]["PotentialVelocityFunctor"][
      "DesiredVel", "Desired velocity for the ego agent.", 6]
    evaluator = RewardShapingEvaluator(params)
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
               num_scenarios=25,
               random_seed=0,
               viewer=True):
    ml_behavior = BehaviorContinuousML(params)
    IntersectionBlueprint.__init__(self,
                                   params=params,
                                   num_scenarios=num_scenarios,
                                   random_seed=random_seed,
                                   ml_behavior=ml_behavior,
                                   viewer=True)


class DiscreteIntersectionBlueprint(IntersectionBlueprint):
  def __init__(self,
               params=None,
               num_scenarios=25,
               random_seed=0):
    ml_behavior = BehaviorDiscreteMacroActionsML(params)
    IntersectionBlueprint.__init__(self,
                                   params=params,
                                   num_scenarios=num_scenarios,
                                   random_seed=random_seed,
                                   ml_behavior=ml_behavior,
                                   viewer=True)