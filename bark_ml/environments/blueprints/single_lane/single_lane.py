# Copyright (c) 2020 fortiss GmbH
#
# Authors: Patrick Hart, Julian Bernhard, Klemens Esterle, and
# Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import os
import numpy as np
from bark.runtime.viewer.matplotlib_viewer import MPViewer
from bark.runtime.scenario.scenario_generation.config_with_ease import \
  LaneCorridorConfig, ConfigWithEase
from bark.core.world.opendrive import *
from bark.core.geometry import *
from bark.core.models.behavior import BehaviorDynamicModel
from bark.core.world.map import MapInterface
from bark.core.world.opendrive import XodrDrivingDirection
from bark.core.world.goal_definition import GoalDefinitionStateLimitsFrenet

from bark_ml.environments.blueprints.blueprint import Blueprint
from bark_ml.evaluators.reward_shaping import RewardShapingEvaluator
from bark_ml.behaviors.cont_behavior import BehaviorContinuousML
from bark_ml.behaviors.discrete_behavior import BehaviorDiscreteMacroActionsML

from bark_ml.observers.nearest_state_observer import NearestAgentsObserver
# from bark_ml.core.observers.frenet_observer import FrenetObserver


class SingleLaneLaneCorridorConfig(LaneCorridorConfig):
  """
  Configures the a single lane, e.g., the goal.
  """

  def __init__(self,
               params=None,
               samplingRange=[1., 10.],
               yOffset=[[0., 0.]],
               xOffset=[[0., 0.]],
               **kwargs):
    super(SingleLaneLaneCorridorConfig, self).__init__(
      params, **dict(kwargs, min_vel=0., max_vel=0.2))
    self._hasBeenCalled = False
    self._samplingRange = samplingRange
    self._yOffset = yOffset
    self._xOffset = xOffset

  def goal(self, world):
    world.map.GetRoadCorridor(
      self._road_ids, XodrDrivingDirection.forward)
    lane_corr = self._road_corridor.lane_corridors[0]
    points = lane_corr.center_line.ToArray()[75:100]
    new_line = Line2d(points)
    return GoalDefinitionStateLimitsFrenet(new_line,
                                           (2.5, 2.),
                                           (0.15, 0.15),
                                           (3., 7.))

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
    samplingRange = np.random.uniform(
      self._samplingRange[0], self._samplingRange[1])
    xy_point =  GetPointAtS(centerline, samplingRange)
    angle = GetTangentAngleAtS(centerline, samplingRange)
    self._hasBeenCalled = True
    yOffsetBounds = self._yOffset[np.random.randint(0, len(self._yOffset))]
    yOffset = np.random.uniform(yOffsetBounds[0], yOffsetBounds[1])
    xOffsetBounds = self._xOffset[np.random.randint(0, len(self._xOffset))]
    xOffset = np.random.uniform(xOffsetBounds[0], xOffsetBounds[1])
    return (xy_point.x() + xOffset,
            xy_point.y() + yOffset,
            angle)

  def behavior_model(self, world):
    behavior_model = BehaviorDynamicModel(self._params)
    return behavior_model

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
    # ego vehicle
    lane_conf = SingleLaneLaneCorridorConfig(params=local_params,
                                             road_ids=[0],
                                             lane_corridor_id=0,
                                             min_vel=0.,
                                             max_vel=0.01,
                                             ds_min=ds_min,
                                             ds_max=ds_max,
                                             s_min=s_min,
                                             s_max=s_max,
                                             controlled_ids=True,
                                             yOffset=[[-0.1, 0.1]])
    lane_configs.append(lane_conf)

    # other vehicle
    if params["World"]["other_vehicle", "Other Vehicle", True]:
      lane_conf_other = SingleLaneLaneCorridorConfig(
        params=local_params,
        road_ids=[0],
        lane_corridor_id=0,
        min_vel=0.,
        max_vel=0.,
        ds_min=ds_min,
        ds_max=ds_max,
        s_min=s_min,
        s_max=s_max,
        controlled_ids=None,
        yOffset=[[2, 2.1], [-2.4, -2.5]],
        samplingRange=[30, 50])
      lane_configs.append(lane_conf_other)

    # Map Definition
    csvfile = os.path.join(
      os.path.dirname(__file__),
      "../../../environments/blueprints/single_lane/base_map_lanes_guerickestr_assymetric_48.csv")
    map_interface = MapInterface()
    map_interface.SetCsvMap(csvfile, 692000, 5.339e+06)

    scenario_generation = \
      ConfigWithEase(
        num_scenarios=num_scenarios,
        random_seed=random_seed,
        params=params,
        map_interface=map_interface,
        lane_corridor_configs=lane_configs)
    # HACK: change
    # scenario_generation._map_interface = map_interface

    if viewer:
      viewer = MPViewer(params=params,
                        x_range=[-100, 100],
                        y_range=[-100, 100],
                        follow_agent_id=True)
      # viewer = BufferedMPViewer(
      #   params=params,
      #   follow_agent_id=True)
    dt = 0.2
    params["ML"]["RewardShapingEvaluator"]["RewardShapingPotentials",
      "Reward shaping functions.", {
        "VelocityPotential" : {
          "desired_vel": 5., "vel_dev_max": 10., "exponent": 0.2, "type": "positive"
        }
    }]
    params["ML"]["RewardShapingEvaluator"]["MaxSteps", "max. number of steps", 100]
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