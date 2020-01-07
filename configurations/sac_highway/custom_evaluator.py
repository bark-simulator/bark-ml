import numpy as np
from bark.world.evaluation import \
  EvaluatorGoalReached, EvaluatorCollisionAgents, \
  EvaluatorCollisionDrivingCorridor, EvaluatorStepCount
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

  def _add_evaluators(self):
    self._evaluators["goal_reached"] = EvaluatorGoalReached()
    self._evaluators["collision"] = \
      EvaluatorCollisionAgents()
    self._evaluators["step_count"] = EvaluatorStepCount()

  def distance_to_goal(self, world):
    d = 0.
    for i, agent in world.agents.items():
      shape = agent.shape
      state = agent.state
      pose = np.zeros(3)
      pose[0] = state[int(StateDefinition.X_POSITION)]
      pose[1] = state[int(StateDefinition.Y_POSITION)]
      pose[2] = state[int(StateDefinition.THETA_POSITION)]
      transformed_polygon = shape.transform(pose)
      # TODO(@hart): scenario generation should support sequential goal
      # goal_poly = agent.goal_definition.GetCurrentGoal(agent).xy_limits
      goal_poly = agent.goal_definition.goal_shape
      d += distance(transformed_polygon, goal_poly)
    d /= i
    return d

  def calculate_reward(self, world, eval_results, action):
    success = eval_results["goal_reached"]
    collision = eval_results["collision"]
    distance_to_goals = self.distance_to_goal(world)
    actions = np.reshape(action, (-1, 2))
    accs = actions[:, 0]
    delta = actions[:, 1]
    inpt_reward = np.sum(10.*delta**2 + accs**2)
    reward = collision * self._collision_penalty + \
      success * self._goal_reward - 0.1*distance_to_goals - inpt_reward
    return reward

  def _evaluate(self, world, eval_results, action):
    """Returns information about the current world state
    """
    done = False
    success = eval_results["goal_reached"]
    collision = eval_results["collision"]
    step_count = eval_results["step_count"]

    reward = self.calculate_reward(world, eval_results, action)    
    if success or collision or step_count > self._max_steps:
      done = True
    return reward, done, eval_results
    
