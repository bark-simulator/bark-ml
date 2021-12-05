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
from bark.core.models.dynamic import *

from bark_ml.environments.blueprints.blueprint import Blueprint
from bark_ml.evaluators.general_evaluator import GeneralEvaluator
from bark_ml.behaviors.cont_behavior import BehaviorContinuousML
from bark_ml.behaviors.discrete_behavior import BehaviorDiscreteMacroActionsML

from bark_ml.observers.nearest_state_observer import NearestAgentsObserver
from bark_ml.core.observers import StaticObserver


class SingleLaneLaneCorridorConfig(LaneCorridorConfig):
  """
  Configures the a single lane, e.g., the goal.
  """

  def __init__(self,
               params=None,
               samplingRange=[1., 8.],
               distanceRange=[2.0, 10.],
               lateralOffset=[[0., 0.]],
               longitudinalOffset=[[0., 0.]],
               **kwargs):
    super(SingleLaneLaneCorridorConfig, self).__init__(
      params, **dict(kwargs, min_vel=0., max_vel=0.2))
    self._samplingRange = samplingRange
    self._lateralOffset = lateralOffset
    self._longitudinalOffset = longitudinalOffset
    self._current_s = None
    self._distanceRange = distanceRange
    self._hasVehicles = True

  def goal(self, world):
    world.map.GetRoadCorridor(
      self._road_ids, XodrDrivingDirection.forward)
    lane_corr = self._road_corridor.lane_corridors[0]
    points = lane_corr.center_line.ToArray()[75:]
    new_line = Line2d(points)
    return GoalDefinitionStateLimitsFrenet(new_line,
                                           (2.5, 2.),
                                           (0.15, 0.15),
                                           (0., 10.))

  def position(self, world):
    if self._road_corridor == None:
      world.map.GenerateRoadCorridor(
        self._road_ids, XodrDrivingDirection.forward)
      self._road_corridor = world.map.GetRoadCorridor(
        self._road_ids, XodrDrivingDirection.forward)
    if self._current_s and (self._current_s >= self._distanceRange[1] or self._controlled_ids):
      return None
    lane_corr = self._road_corridor.lane_corridors[self._lane_corridor_id]
    centerline = lane_corr.center_line

    if not self._current_s:
      self._current_s = self._distanceRange[0]

    self._current_s += np.random.uniform(
      self._samplingRange[0], self._samplingRange[1])

    xy_point =  GetPointAtS(centerline, self._current_s)
    angle = GetTangentAngleAtS(centerline, self._current_s)


    lateralOffsetBounds = self._lateralOffset[np.random.randint(0, len(self._lateralOffset))]
    lateralOffset = np.random.uniform(lateralOffsetBounds[0], lateralOffsetBounds[1])
    longitudinalOffsetBounds = self._longitudinalOffset[
      np.random.randint(0, len(self._longitudinalOffset))]
    longitudinalOffset = np.random.uniform(
      longitudinalOffsetBounds[0], longitudinalOffsetBounds[1])

    return (xy_point.x() + longitudinalOffset,
            xy_point.y() + lateralOffset,
            angle)

  def behavior_model(self, world):
    behavior_model = BehaviorDynamicModel(self._params)
    return behavior_model

  @property
  def dynamic_model(self):
    """Returns dyn. model
    """
    if self._controlled_ids is not None:
      return SingleTrackSteeringRateModel(self._params)
    return SingleTrackModel(self._params)


  def state(self, world):
    """Returns a state of the agent

    Arguments:
        world {bark.core.world}

    Returns:
        np.array -- time, x, y, theta, velocity
    """
    if not self._hasVehicles:
      return None

    if self._controlled_ids is not None:
      pose = self.position(world)
      if pose is None:
        return None
      velocity = self.velocity()
      return np.array([0, pose[0], pose[1], pose[2], velocity, 0.])
    else:
      return super().state(world)

  def reset(self):
    """Resets the LaneCorridorConfig
    """
    if self._controlled_ids is None:
      hasVehicles = np.random.randint(0, 2)
      self._hasVehicles = hasVehicles
    self._current_s = None


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
               mode="medium",
               csv_path=None):
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
                                             lateralOffset=[[-0.1, 0.1]],
                                             wb=2.786,
                                             crad=1.1)
    lane_configs.append(lane_conf)

    # other vehicle
    if params["World"]["other_vehicle", "Other Vehicle", True]:
      lane_conf_other_left = SingleLaneLaneCorridorConfig(
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
        lateralOffset=[[1.8, 2.4]],
        samplingRange=[20., 22.5],
        distanceRange=[8, 70],
        wb=2.786,
        crad=1.)
      lane_configs.append(lane_conf_other_left)
      lane_conf_other_right = SingleLaneLaneCorridorConfig(
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
        lateralOffset=[[-2.2, -3.2]],
        samplingRange=[20., 22.5],
        distanceRange=[15., 70])
      lane_configs.append(lane_conf_other_right)

    # Map Definition
    if csv_path is None:
      csv_path = os.path.join(
        os.path.dirname(__file__),
        "../../../environments/blueprints/single_lane/base_map_lanes_guerickestr_short_assymetric_48.csv")
    print(f"CSV map file path is: {csv_path}.")
    map_interface = MapInterface()
    map_interface.SetCsvMap(
      csv_path,
      params["SingleLaneBluePrint"]["MapOffsetX", "", 692000],
      params["SingleLaneBluePrint"]["MapOffsetY", "", 5.339e+06])
    scenario_generation = \
      ConfigWithEase(
        num_scenarios=num_scenarios,
        random_seed=random_seed,
        params=params,
        map_interface=map_interface,
        lane_corridor_configs=lane_configs)
    if viewer:
      viewer = MPViewer(params=params,
                        x_range=[-100, 100],
                        y_range=[-100, 100],
                        follow_agent_id=True)
    evaluator = GeneralEvaluator(params)
    observer = NearestAgentsObserver(params)
    ml_behavior = ml_behavior
    dt = 0.2
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
               mode="dense",
               csv_path=None):
    ml_behavior = BehaviorContinuousML(params)
    SingleLaneBlueprint.__init__(self,
                              params=params,
                              num_scenarios=num_scenarios,
                              random_seed=random_seed,
                              ml_behavior=ml_behavior,
                              viewer=True,
                              mode=mode,
                              csv_path=csv_path)


class DiscreteSingleLaneBlueprint(SingleLaneBlueprint):
  def __init__(self,
               params=None,
               num_scenarios=25,
               random_seed=0,
               mode="dense",
               csv_path=None):
    ml_behavior = BehaviorDiscreteMacroActionsML(params)
    SingleLaneBlueprint.__init__(self,
                              params=params,
                              num_scenarios=num_scenarios,
                              random_seed=random_seed,
                              ml_behavior=ml_behavior,
                              viewer=True,
                              mode=mode,
                              csv_path=csv_path)