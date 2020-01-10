import numpy as np
from bark.world.evaluation import \
  EvaluatorGoalReached, EvaluatorCollisionAgents, \
  EvaluatorCollisionDrivingCorridor, EvaluatorStepCount, EvaluatorDrivableArea
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
    self._evaluators["drivable_area"] = EvaluatorDrivableArea()
    self._evaluators["collision"] = \
      EvaluatorCollisionAgents()
    self._evaluators["step_count"] = EvaluatorStepCount()

  def distance_to_goal(self, world):
    d = 0.
    for i, agent in world.agents.items():
      state = agent.state
      goal_poly = agent.goal_definition.goal_shape
      d += distance(goal_poly, Point2d(state[1], state[2]))
    d /= i
    return d

  def calculate_reward(self, world, eval_results, action):
    success = eval_results["goal_reached"]
    collision = eval_results["collision"]

    distance_to_goals = self.distance_to_goal(world)
    actions = np.reshape(action, (-1, 2))
    accs = actions[:, 0]
    delta = actions[:, 1]

    # TODO(@hart): use parameter server
    inpt_reward = 4.*np.sum((1/0.15*delta)**2 + (accs)**2)
    reward = collision * self._collision_penalty + \
      success * self._goal_reward - inpt_reward + \
      0.1*distance_to_goals 
    return reward

  def _evaluate(self, world, eval_results, action):
    """Returns information about the current world state
    """
    done = False
    success = eval_results["goal_reached"]
    collision = eval_results["collision"]
    drivable_area = eval_results["drivable_area"]
    step_count = eval_results["step_count"]

    reward = self.calculate_reward(world, eval_results, action)    
    if success or collision or step_count > self._max_steps or drivable_area:
      done = True
    return reward, done, eval_results
    
