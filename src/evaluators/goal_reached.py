from bark.world.evaluation import \
  EvaluatorGoalReached, EvaluatorCollisionAgents, \
  EvaluatorStepCount
from modules.runtime.commons.parameters import ParameterServer

from src.evaluators.evaluator import StateEvaluator

class GoalReached(StateEvaluator):
  def __init__(self,
               params=ParameterServer(),
               eval_agent=None):
    StateEvaluator.__init__(self, params)
    self._goal_reward = \
      self._params["ML"]["Evaluator"]["goal_reward",
        "Reward for reaching the goal.",
        100.]
    self._collision_penalty = \
      self._params["ML"]["Evaluator"]["collision_penalty",
        "Reward given for a collisions.",
        -100.]
    self._max_steps = \
      self._params["ML"]["Evaluator"]["max_steps",
        "Maximum steps per episode.",
        50]
    self._eval_agent = eval_agent

  def _add_evaluators(self):
    self._evaluators["goal_reached"] = EvaluatorGoalReached()
    self._evaluators["collision"] = EvaluatorCollisionAgents()
    self._evaluators["step_count"] = EvaluatorStepCount()

  def _evaluate(self, observed_world, eval_results, action, observed_state):
    """Returns information about the current world state
    """
    done = False
    success = eval_results["goal_reached"]
    collision = eval_results["collision"]
    step_count = eval_results["step_count"]
    # determine whether the simulation should terminate
    if success or collision or step_count > self._max_steps:
      done = True
    # calculate reward
    reward = collision * self._collision_penalty + \
      success * self._goal_reward
    return reward, done, eval_results
    
  def reset(self, world):
    return super(GoalReached, self).reset(world)