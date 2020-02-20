import numpy as np
from bark.world.evaluation import \
  EvaluatorGoalReached, EvaluatorCollisionAgents, \
  EvaluatorStepCount, EvaluatorDrivableArea, EvaluatorCollisionEgoAgent
from modules.runtime.commons.parameters import ParameterServer
from bark.geometry import *
from bark.models.dynamic import StateDefinition

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

  def _add_evaluators(self, agents_to_evaluate):
    self._evaluators["goal_reached"] = EvaluatorGoalReached(
      agents_to_evaluate[0])
    self._evaluators["drivable_area"] = EvaluatorDrivableArea(
      agents_to_evaluate[0])
    self._evaluators["collision"] = \
      EvaluatorCollisionEgoAgent(
        agents_to_evaluate[0])
    self._evaluators["step_count"] = EvaluatorStepCount()

  def distance_to_goal(self, world):
    d = 0.
    for idx in [self._eval_agent]:
      agent = world.agents[idx]
      state = agent.state
      # TODO(@hart): fix.. offset 0.75 so we drive to the middle of the polygon
      center_line = agent.road_corridor.lane_corridors[0].center_line
      d += Distance(center_line, Point2d(state[1], state[2]))
    return d

  def deviation_velocity(self, world):
    desired_v = 15.
    delta_v = 0.
    for idx in [self._eval_agent]:
      vel = world.agents[idx].state[int(StateDefinition.VEL_POSITION)]
      delta_v += (desired_v-vel)**2
    return delta_v
  
  def calculate_reward(self, observed_world, eval_results, action, observed_state):  # NOLINT
    success = eval_results["goal_reached"]
    collision = eval_results["collision"]
    drivable_area = eval_results["drivable_area"]

    # ego_agent = observed_world.agents[self._eval_agent]
    # goal_def = ego_agent.goal_definition
    # goal_center_line = goal_def.center_line
    # ego_agent_state = ego_agent.state
    # lateral_offset = Distance(goal_center_line,
    #                           Point2d(ego_agent_state[1], ego_agent_state[2]))

    distance_to_goals = self.distance_to_goal(observed_world)
    actions = np.reshape(action, (-1, 2))
    accs = actions[:, 0]
    delta = actions[:, 1]

    # TODO(@hart): use parameter server
    inpt_reward = np.sum((4/0.15*delta)**2 + (accs)**2)
    reward = collision * self._collision_penalty + \
      success * self._goal_reward + \
      drivable_area * self._collision_penalty - \
      0.001*distance_to_goals**2
    
    return reward

  def _evaluate(self, world, eval_results, action, observed_state):
    """Returns information about the current world state
    """
    done = False
    success = eval_results["goal_reached"]
    collision = eval_results["collision"]
    drivable_area = eval_results["drivable_area"]
    step_count = eval_results["step_count"]

    # if this is a FrenetCorr we will use this for the observer and evaluator
    # print(success, collision, drivable_area, step_count)

    reward = self.calculate_reward(world, eval_results, action, observed_state)    
    if success or collision or step_count > self._max_steps or drivable_area:
      done = True
    return reward, done, eval_results
    
