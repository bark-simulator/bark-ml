from bark.core.geometry import *

def GetGoalLaneCorridorForGoal(observed_world):
  """Returns the lanecorridor the goal is in."""
  ego_agent = observed_world.ego_agent
  goal_def = ego_agent.goal_definition
  goal_shape = goal_def.goal_shape
  rc = observed_world.ego_agent.road_corridor
  lane_corr = None
  for lc in rc.lane_corridors:
    if Collide(lc.polygon, goal_shape):
      lane_corr = lc
  return lane_corr

def CalculateDistanceToGoal(observed_world):
  """Calculates the distance to the goal of the ego_agent."""
  goal_lane_corr = GetGoalLaneCorridorForGoal(observed_world)
  goal_center_line = goal_lane_corr.center_line
  ego_agent = observed_world.ego_agent
  ego_agent_state = ego_agent.state
  distance_to_goal = Distance(
    goal_center_line,
    Point2d(ego_agent_state[1], ego_agent_state[2]))
  return distance_to_goal