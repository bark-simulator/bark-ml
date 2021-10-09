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
from bark.runtime.viewer.matplotlib_viewer import MPViewer
from bark.runtime.scenario.scenario_generation.config_with_ease import \
  LaneCorridorConfig, ConfigWithEase
from bark.core.world.goal_definition import GoalDefinitionPolygon
from bark.core.geometry import Polygon2d, Point2d
from bark.core.world.opendrive import *
from bark.core.geometry import *

from bark_ml.environments.blueprints.blueprint import Blueprint
from bark_ml.evaluators.reward_shaping_max_steps import RewardShapingEvaluatorMaxSteps
from bark_ml.behaviors.cont_behavior import BehaviorContinuousML
from bark_ml.behaviors.discrete_behavior import BehaviorDiscreteMacroActionsML

# from bark_ml.observers.nearest_state_observer import NearestAgentsObserver
from bark_ml.core.observers import NearestObserver


class SingleLaneLaneCorridorConfig(LaneCorridorConfig):

  """
  Configures the a single lane, e.g., the goal.
  """

  def __init__(self,
               params=None,
               **kwargs):
    super(SingleLaneLaneCorridorConfig, self).__init__(
      params, **dict(kwargs, min_vel=0., max_vel=0.2))
    self._hasBeenCalled = False

  def goal(self, world):
    goal_polygon = Polygon2d(
      [0, 0, 0],
      [Point2d(95, -4), Point2d(95, 0), Point2d(100, 0), Point2d(100, -4)])
    return GoalDefinitionPolygon(goal_polygon)

  def position(self, world):
    if self._road_corridor == None:
      world.map.GenerateRoadCorridor(
        self._road_ids, XodrDrivingDirection.forward)
      self._road_corridor = world.map.GetRoadCorridor(
        self._road_ids, XodrDrivingDirection.forward)
    if self._hasBeenCalled:
      self._hasBeenCalled = False
      return None
    lane_corr = self._road_corridor.lane_corridors[self._lane_corridor_id]
    centerline = lane_corr.center_line
    samplingRange = np.random.uniform(0, 10)
    xy_point =  GetPointAtS(centerline, samplingRange)
    angle = GetTangentAngleAtS(centerline, samplingRange)
    self._hasBeenCalled = True
    return (xy_point.x(), xy_point.y(), angle)

class SingleLaneBlueprint(Blueprint):
  """The SingleLane blueprint sets up a merging scenario with initial
  conditions.
  Two lane SingleLane, with the ego vehicle being placed on the right lane
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
      ds_min = 10.
      ds_max = 15.
    if mode == "medium":
      ds_min = 20.
      ds_max = 35.
    params["World"]["remove_agents_out_of_map"] = False

    lane_configs = []
    s_min = 0
    s_max = 100
    local_params = params.clone()
    lane_conf = SingleLaneLaneCorridorConfig(params=local_params,
                                             road_ids=[16],
                                             lane_corridor_id=0,
                                             min_vel=0.,
                                             max_vel=0.01,
                                             ds_min=ds_min,
                                             ds_max=ds_max,
                                             s_min=s_min,
                                             s_max=s_max,
                                             controlled_ids=True)
    lane_configs.append(lane_conf)

    scenario_generation = \
      ConfigWithEase(
        num_scenarios=num_scenarios,
        map_file_name=os.path.join(
          os.path.dirname(__file__),
          "../../../environments/blueprints/single_lane/single_lane.xodr"),  # pylint: disable=unused-import
        random_seed=random_seed,
        params=params,
        lane_corridor_configs=lane_configs)
    if viewer:
      viewer = MPViewer(params=params,
                        use_world_bounds=True)
      # viewer = BufferedMPViewer(
      #   params=params,
      #   follow_agent_id=True)
    dt = 0.2
    params["ML"]["RewardShapingEvaluator"]["RewardShapingPotentials",
      "Reward shaping functions.", {
        "VelocityPotential" : {
          "desired_vel": 20., "vel_dev_max": 20., "exponent": 0.2, "type": "positive"
        }
    }]
    evaluator = RewardShapingEvaluatorMaxSteps(params)
    observer = NearestObserver(params)
    ml_behavior = ml_behavior

    super().__init__(
      scenario_generation=scenario_generation,
      viewer=viewer,
      dt=dt,
      evaluator=evaluator,
      observer=observer,
      ml_behavior=ml_behavior)


class ContinuousSingleLaneBlueprint(SingleLaneBlueprint):
  def __init__(self,
               params=None,
               num_scenarios=25,
               random_seed=0,
               viewer=True,
               mode="dense"):
    ml_behavior = BehaviorContinuousML(params)
    SingleLaneBlueprint.__init__(self,
                              params=params,
                              num_scenarios=num_scenarios,
                              random_seed=random_seed,
                              ml_behavior=ml_behavior,
                              viewer=True,
                              mode=mode)


class DiscreteSingleLaneBlueprint(SingleLaneBlueprint):
  def __init__(self,
               params=None,
               num_scenarios=25,
               random_seed=0,
               mode="dense"):
    ml_behavior = BehaviorDiscreteMacroActionsML(params)
    SingleLaneBlueprint.__init__(self,
                              params=params,
                              num_scenarios=num_scenarios,
                              random_seed=random_seed,
                              ml_behavior=ml_behavior,
                              viewer=True,
                              mode=mode)