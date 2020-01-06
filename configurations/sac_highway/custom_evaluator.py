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
    for _, agent in world.agents.items():
      shape = agent.shape
      state = agent.state
      pose = np.zeros(3)
      pose[0] = state[int(StateDefinition.X_POSITION)]
      pose[1] = state[int(StateDefinition.Y_POSITION)]
      pose[2] = state[int(StateDefinition.THETA_POSITION)]
      transformed_polygon = shape.transform(pose)
      goal_poly = agent.goal_definition.GetCurrentGoal(agent).xy_limits
      d += distance(transformed_polygon, goal_poly)
    return d

  def _evaluate(self, world, eval_results, action):
    """Returns information about the current world state
    """
    done = False
    success = eval_results["goal_reached"]
    collision = eval_results["collision"]
    step_count = eval_results["step_count"]
    # TODO(@hart): distance to goal

    # calculate reward
    reward = collision * self._collision_penalty + \
      success * self._goal_reward
    
    # determine if terminal
    if success or collision or step_count > self._max_steps:
      done = True
    
    print("Distance to goal: ", self.distance_to_goal(world))
    return reward, done, eval_results
    
