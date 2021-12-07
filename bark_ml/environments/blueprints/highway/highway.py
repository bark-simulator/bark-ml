# Copyright (c) 2020 fortiss GmbH
#
# Authors: Patrick Hart, Julian Bernhard, Klemens Esterle, and
# Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import os
import numpy as np
from bark.runtime.viewer.buffered_mp_viewer import BufferedMPViewer
from bark.runtime.scenario.scenario_generation.config_with_ease import \
  LaneCorridorConfig, ConfigWithEase
from bark.core.world.goal_definition import GoalDefinitionPolygon
from bark.core.geometry import Polygon2d, Point2d

from bark_ml.environments.blueprints.blueprint import Blueprint
from bark_ml.evaluators.evaluator_configs import RewardShapingEvaluator
from bark_ml.behaviors.cont_behavior import BehaviorContinuousML
from bark_ml.behaviors.discrete_behavior import BehaviorDiscreteMacroActionsML
from bark_ml.core.observers import NearestObserver


class HighwayLaneCorridorConfig(LaneCorridorConfig):

  """
  Configures the a single lane, e.g., the goal.
  """

  def __init__(self,
               params=None,
               **kwargs):
    super(HighwayLaneCorridorConfig, self).__init__(params, **kwargs)

  def goal(self, world):
    goal_polygon = Polygon2d(
      [-1000, -1000, -1000],
      [Point2d(-1, -1), Point2d(-1, 1), Point2d(1, 1), Point2d(1, -1)])
    return GoalDefinitionPolygon(goal_polygon)


class HighwayBlueprint(Blueprint):
  """The highway blueprint sets up a merging scenario with initial
  conditions.

  Two lane highway, with the ego vehicle being placed on the right lane
  and the ego vehicle's goal on the left lane.
  """

  def __init__(self,
               params=None,
               num_scenarios=250,
               random_seed=0,
               ml_behavior=None,
               viewer=True,
               mode="medium"):
    if mode == "dense":
      ds_min = 15.
      ds_max = 30.
    if mode == "medium":
      ds_min = 20.
      ds_max = 35.
    params["World"]["remove_agents_out_of_map"] = False

    ego_lane_id = 2
    lane_configs = []
    for i in range(0, 4):
      is_controlled = True if (ego_lane_id == i) else None
      s_min = 0
      s_max = 250
      if is_controlled == True:
        s_min = 40.
        s_max = 80.
      local_params = params.clone()
      local_params["BehaviorIDMClassic"]["DesiredVelocity"] = np.random.uniform(12, 17)
      lane_conf = HighwayLaneCorridorConfig(params=local_params,
                                            road_ids=[16],
                                            lane_corridor_id=i,
                                            min_vel=12.5,
                                            max_vel=17.5,
                                            ds_min=ds_min,
                                            ds_max=ds_max,
                                            s_min=s_min,
                                            s_max=s_max,
                                            controlled_ids=is_controlled)
      lane_configs.append(lane_conf)

    scenario_generation = \
      ConfigWithEase(
        num_scenarios=num_scenarios,
        map_file_name=os.path.join(os.path.dirname(__file__), "../../../environments/blueprints/highway/round_highway.xodr"),  # pylint: disable=unused-import
        random_seed=random_seed,
        params=params,
        lane_corridor_configs=lane_configs)
    if viewer:
      # viewer = MPViewer(params=params,
      #                   use_world_bounds=True)
      viewer = BufferedMPViewer(
        params=params,
        x_range=[-55, 55],
        y_range=[-55, 55],
        follow_agent_id=True)
    dt = 0.2
    params["ML"]["RewardShapingEvaluator"]["PotentialVelocityFunctor"][
          "DesiredVel", "Desired velocity for the ego agent.", 20]
    evaluator = RewardShapingEvaluator(params)
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