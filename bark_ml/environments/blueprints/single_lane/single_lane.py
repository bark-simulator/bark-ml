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
from bark.runtime.viewer.video_renderer import VideoRenderer
from bark.runtime.scenario.scenario_generation.config_with_ease import \
  LaneCorridorConfig, ConfigWithEase
from bark.runtime.commons.xodr_parser import XodrParser
from bark.core.world.opendrive import *
from bark.core.geometry import Line2d, GetPointAtS, GetTangentAngleAtS, Polygon2d, Point2d
from bark.core.models.behavior import BehaviorDynamicModel, BehaviorMacroActionsFromParamServer
from bark.core.world.map import MapInterface
from bark.core.world.opendrive import XodrDrivingDirection
from bark.core.world.goal_definition import GoalDefinitionStateLimitsFrenet, GoalDefinitionPolygon
from bark.core.models.dynamic import SingleTrackSteeringRateModel, SingleTrackModel
from bark.core.models.observer import ObserverModelParametric

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
               goalConfigs= None,
               **kwargs):
    super(SingleLaneLaneCorridorConfig, self).__init__(
      params, **dict(kwargs, min_vel=0., max_vel=0.2))
    self._samplingRange = samplingRange
    self._lateralOffset = lateralOffset
    self._longitudinalOffset = longitudinalOffset
    self._current_s = None
    self._distanceRange = distanceRange
    self._hasVehicles = True
    self._goalConfigs = goalConfigs

  def goal(self, world):
    world.map.GetRoadCorridor(
      self._road_ids, XodrDrivingDirection.forward)
    lane_corr = self._road_corridor.lane_corridors[0]
    points = lane_corr.center_line.ToArray()
    len_cent_line_ = len(points)
    st_idx = self._goalConfigs.get("first_pt_index_range",[0.5,0.9])
    idx = np.random.randint(st_idx[0]*len_cent_line_, st_idx[1]*len_cent_line_)
    pt_num = self._goalConfigs.get("length_pt_portion",None)
    if pt_num is None:
      new_line = Line2d(points[idx:])
    else:
      pt_num = int(pt_num * len_cent_line_)
      new_line = Line2d(points[idx:idx+pt_num])
    max_lateral_dist_= tuple(self._goalConfigs.get("max_lateral_dist",[2.5,2.0]))
    max_orient_diff_= tuple(self._goalConfigs.get("max_orient_diff",[0.15, 0.15]))
    velocity_range_= tuple(self._goalConfigs.get("velocity_range",[0.0, 20.0]))
    return GoalDefinitionStateLimitsFrenet(new_line,
                                           max_lateral_dist_,
                                           max_orient_diff_,
                                           velocity_range_)

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
    behavior_model.ActionToBehavior(np.array([0., 0.]))
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

  def velocity(self):
    return 0.0

class NaiveGoalSingleLaneLaneCorridorConfig(LaneCorridorConfig):

  """
  Configures the a single lane, e.g., with a simple goal.
  """
  def __init__(self,
               params=None,
               **kwargs):
    super(NaiveGoalSingleLaneLaneCorridorConfig, self).__init__(params,
                                                                **kwargs)
  def goal(self, world):
        lane_corr = self._road_corridor.lane_corridors[0]
        goal_polygon = Polygon2d([0, 0, 0], [Point2d(-1, -1), Point2d(-1, 1), Point2d(1, 1), Point2d(1, -1)])
        goal_polygon = goal_polygon.Translate(Point2d(lane_corr.center_line.ToArray()[-1, 0], lane_corr.center_line.ToArray()[-1, 1]))
        return GoalDefinitionPolygon(goal_polygon)
  

class NaiveSingleLaneBlueprint(Blueprint):
  """The NaiveSingleLaneBlueprint blueprint sets up a single lane scenario with initial
  conditions.
  One lane SingleLane, with the ego vehicle following leading vehicle.
  """

  def __init__(self,
               params=None,
               num_scenarios=250,
               random_seed=0,
               ml_behavior=None,
               viewer=True,
               dt=0.2,
               xodr_path=None,
               goalConfigs=None,
               mode="medium"):
    ds_min = 35. # [m]
    ds_max = 50.
    if mode == "dense":
      ds_min = 20. # [m]
      ds_max = 35.
    if mode == "medium":
      ds_min = 30. # [m]
      ds_max = 40.
    params["World"]["remove_agents_out_of_map"] = False
    lane_configs = []
    s_min = 20.
    s_max = 80.
    local_params = params.clone()
    local_params["BehaviorIDMClassic"]["DesiredVelocity"] = np.random.uniform(12, 17)
    lane_conf = NaiveGoalSingleLaneLaneCorridorConfig(params=local_params,
                                                      road_ids=[16],
                                                      lane_corridor_id=0,
                                                      min_vel=10,
                                                      max_vel=17,
                                                      ds_min=ds_min,
                                                      ds_max=ds_max,
                                                      s_min=s_min,
                                                      s_max=s_max,
                                                      controlled_ids=True)
    lane_configs.append(lane_conf)

    # Map Definition
    if xodr_path is None:
      xodr_path = os.path.join(
        os.path.dirname(__file__),
        "../../../environments/blueprints/single_lane/single_lane.xodr")
    print(f"xodr map file path is: {xodr_path}.")
    map_interface = MapInterface()
    xodr_parser = XodrParser(xodr_path)
    map_interface.SetOpenDriveMap(xodr_parser.map)
    
    scenario_generation = \
      ConfigWithEase(
        num_scenarios=num_scenarios,
        random_seed=random_seed,
        params=params,
        map_interface=map_interface,
        lane_corridor_configs=lane_configs)
    
    if viewer:
      viewer = MPViewer(params=params,
                        x_range=[-150, 150],
                        y_range=[-150, 150],
                        follow_agent_id=True)
      if params["Experiment"]["ExportVideos"]:
        viewer = VideoRenderer(renderer=viewer, world_step_time=dt)

    evaluator = GeneralEvaluator(params)
    observer = NearestAgentsObserver(params)

    super().__init__(
      scenario_generation=scenario_generation,
      viewer=viewer,
      dt=dt,
      evaluator=evaluator,
      observer=observer,
      ml_behavior=ml_behavior)
    
  
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
               dt=0.2,
               csv_path=None,
               laneCorrConfigs={
                 "corr0" :  {
                   "samplingRange": [10., 20.],
                   "distanceRange": [15., 70.],
                   "lateralOffset": [[2.2, 2.4]]
                 },
                 "corr1" :  {
                   "samplingRange": [10., 20.],
                   "distanceRange": [15., 70.],
                   "lateralOffset": [[-2.2, -3.2]]
                  }},
                  goalConfigs= None):
    params["World"]["remove_agents_out_of_map"] = False
    lane_configs = []
    local_params = params.clone()
    # ego vehicle
    lane_conf = SingleLaneLaneCorridorConfig(params=local_params,
                                             road_ids=[0],
                                             lane_corridor_id=0,
                                             min_vel=0.,
                                             max_vel=0.01,
                                             controlled_ids=True,
                                             lateralOffset=[[-0.1, 0.1]],
                                             wb=2.786,
                                             crad=1.1,
                                             samplingRange=[4., 10.],
                                             distanceRange=[0., 20.],
                                             goalConfigs=goalConfigs)
    lane_configs.append(lane_conf)


    # other vehicle
    if params["World"]["other_vehicle", "Other Vehicle", True]:
      for conf in laneCorrConfigs.values():
        lane_conf = SingleLaneLaneCorridorConfig(
          params=local_params,
          road_ids=[0],
          lane_corridor_id=0,
          min_vel=0.,
          max_vel=0.,
          controlled_ids=None,
          lateralOffset=conf["lateralOffset"],
          samplingRange=conf["samplingRange"],
          distanceRange=conf["distanceRange"],
          wb=2.786,
          crad=1.,
          goalConfigs=goalConfigs)
        lane_configs.append(lane_conf)

    # Map Definition
    if csv_path is None:
      csv_path = os.path.join(
        os.path.dirname(__file__),
        "../../../environments/blueprints/single_lane/base_map_lanes_guerickestr_short_assymetric_48.csv")
    print(f"CSV map file path is: {csv_path}.")
    map_interface = MapInterface()
    map_interface.SetCsvMap(
      csv_path,
      params["Experiment"]["Blueprint"]["MapOffsetX", "", 692000],
      params["Experiment"]["Blueprint"]["MapOffsetY", "", 5.339e+06])
    observer_model = None
    if params["Experiment"]["Blueprint"]["UseObserveModel", "", False]:
      params["ObserverModelParametric"] \
            ["EgoStateDeviationDist"]["Covariance", "", [[0.05, 0.0, 0.0, 0.0],
                                                      [0.0, 0.01, 0.0, 0.0],
                                                      [0.0, 0.00, 0.001, 0.0],
                                                      [0.0, 0.00, 0.0, 0.05]]]
      params["ObserverModelParametric"] \
            ["EgoStateDeviationDist"]["Mean", "", [0.0, 0.0, 0.0, 0.0]]
      params["ObserverModelParametric"] \
            ["OtherStateDeviationDist"]["Covariance", "", [[0.05, 0.0, 0.0, 0.0],
                                                      [0.0, 0.01, 0.0, 0.0],
                                                      [0.0, 0.00, 0.001, 0.0],
                                                      [0.0, 0.00, 0.0, 0.05]]]
      params["ObserverModelParametric"] \
            ["OtherStateDeviationDist"]["Mean", "", [0.0, 0.0, 0.0, 0.0]]
      observer_model = ObserverModelParametric(params)
    scenario_generation = \
      ConfigWithEase(
        num_scenarios=num_scenarios,
        random_seed=random_seed,
        params=params,
        map_interface=map_interface,
        lane_corridor_configs=lane_configs,
        observer_model=observer_model)
    if viewer:
      viewer = MPViewer(params=params,
                        x_range=[-150, 150],
                        y_range=[-150, 150],
                        follow_agent_id=True)
      if params["Experiment"]["ExportVideos"]:
        viewer = VideoRenderer(renderer=viewer, world_step_time=dt)
    evaluator = GeneralEvaluator(params)
    observer = NearestAgentsObserver(params)

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
               dt=0.2,
               laneCorrConfigs={},
               csv_path=None,
               goalConfigs={}):
    ml_behavior = BehaviorContinuousML(params)
    SingleLaneBlueprint.__init__(self,
                              params=params,
                              num_scenarios=num_scenarios,
                              random_seed=random_seed,
                              ml_behavior=ml_behavior,
                              viewer=viewer,
                              dt=dt,
                              laneCorrConfigs=laneCorrConfigs,
                              csv_path=csv_path,
                              goalConfigs=goalConfigs)


class DiscreteSingleLaneBlueprint(SingleLaneBlueprint):
  def __init__(self,
               params=None,
               num_scenarios=25,
               random_seed=0,
               viewer=True,
               dt=0.2,
               laneCorrConfigs={},
               csv_path=None,
               goalConfigs={}):
    ml_behavior = BehaviorDiscreteMacroActionsML(params)
    SingleLaneBlueprint.__init__(self,
                              params=params,
                              num_scenarios=num_scenarios,
                              random_seed=random_seed,
                              ml_behavior=ml_behavior,
                              viewer=viewer,
                              dt=dt,
                              laneCorrConfigs=laneCorrConfigs,
                              csv_path=csv_path,
                              goalConfigs=goalConfigs)
    
class ContinuousNaiveSingleLaneBlueprint(NaiveSingleLaneBlueprint):
  def __init__(self,
               params=None,
               num_scenarios=25,
               random_seed=0,
               viewer=True,
               dt=0.2,
               xodr_path=None,
               goalConfigs={}):
    ml_behavior = BehaviorContinuousML(params)
    NaiveSingleLaneBlueprint.__init__(self,
                              params=params,
                              num_scenarios=num_scenarios,
                              random_seed=random_seed,
                              ml_behavior=ml_behavior,
                              viewer=viewer,
                              dt=dt,
                              xodr_path=xodr_path,
                              goalConfigs=goalConfigs)