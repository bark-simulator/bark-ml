import numpy as np
from bark.world.evaluation import \
  EvaluatorGoalReached, EvaluatorCollisionEgoAgent, \
  EvaluatorStepCount, EvaluatorDrivableArea
from modules.runtime.commons.parameters import ParameterServer
from bark.geometry import *

from src.evaluators.goal_reached import GoalReached

class CustomEvaluator(GoalReached):
  """Shows the capability of custom elements inside
     a configuration.
  """
  def __init__(self,
               params=ParameterServer(),
               eval_agent=None):
    GoalReached.__init__(self,
                         params,
                         eval_agent)

  def _add_evaluators(self):
    self._evaluators["goal_reached"] = EvaluatorGoalReached(self._eval_agent)
    self._evaluators["ego_collision"] = \
      EvaluatorCollisionEgoAgent(self._eval_agent)
    self._evaluators["step_count"] = EvaluatorStepCount()
    self._evaluators["drivable_area"] = EvaluatorDrivableArea()

  def _distance_to_center_line(self, world):
    """calculates the distance of the agent
       to its centerline
    
    Arguments:
        world {bark.world} -- bark world
    
    Returns:
        float -- distance to centerline
    """
    agent = world.agents[self._eval_agent]
    agent_state = agent.state
    # centerline = agent.local_map.get_driving_corridor().center
    # TODO(@hart): HACK; to see whether a lane-change can be learned
    #agent_xy = Point2d(agent.state[1] + 4., agent.state[2])
    return 0 # distance(centerline, agent_xy)

  def _evaluate(self, world, eval_results, action):
    """Returns information about the current world state
    """
    # should read parameter that has been set in the observer
    # print(self._params["ML"]["Maneuver"]["lane_change"])
    agent_state = world.agents[self._eval_agent].state
    done = False
    success = eval_results["goal_reached"]
    distance = self._distance_to_center_line(world)
    collision = eval_results["ego_collision"]
    step_count = eval_results["step_count"]
    drivable_area = eval_results["drivable_area"]
    # determine whether the simulation should terminate
    if success or collision or step_count > self._max_steps or drivable_area:
      done = True
    # calculate reward
    reward = collision * self._collision_penalty + \
      success * self._goal_reward - 0.1*distance + \
      drivable_area * self._collision_penalty
    return reward, done, eval_results
    

