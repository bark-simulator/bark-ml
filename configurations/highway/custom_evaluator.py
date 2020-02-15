import numpy as np
from bark.world.evaluation import \
  EvaluatorGoalReached, EvaluatorCollisionAgents, \
  EvaluatorStepCount, EvaluatorDrivableArea
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

  # def deviation_velocity(self, observed_world):
  #   desired_v = 10.
  #   delta_v = 0.
  #   for _, agent in observed_world.agents.items():
  #     vel = agent.state[int(StateDefinition.VEL_POSITION)]
  #     delta_v += (desired_v-vel)**2
  #   return delta_v
  
  def calculate_reward(self, observed_world, eval_results, action, observed_state):  # NOLINT
    success = eval_results["goal_reached"]
    collision = eval_results["collision"]
    drivable_area = eval_results["drivable_area"]

    # distance_to_goals = self.distance_to_goal(observed_world)
    lateral_offset = observed_state[0, 1]**2
    actions = np.reshape(action, (-1, 2))
    accs = actions[:, 0]
    delta = actions[:, 1]

    # TODO(@hart): use parameter server
    inpt_reward = np.sum((1/0.15*delta)**2 + (accs)**2)
    reward = collision * self._collision_penalty + \
      success * self._goal_reward + \
      drivable_area * self._collision_penalty - \
      0.01*lateral_offset - 0.01*inpt_reward
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
    
