from bark.world.evaluation import \
  EvaluatorGoalReached, EvaluatorCollisionEgoAgent, \
  EvaluatorCollisionDrivingCorridor, EvaluatorStepCount
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
    self._evaluators["goal_reached"] = EvaluatorGoalReached(self._eval_agent)
    self._evaluators["ego_collision"] = \
      EvaluatorCollisionEgoAgent(self._eval_agent)
    self._evaluators["collision_driving_corridor"] = \
      EvaluatorCollisionDrivingCorridor()
    self._evaluators["step_count"] = EvaluatorStepCount()

  def _evaluate(self, world, eval_results):
    """Returns information about the current world state
    """
    done = False
    success = eval_results["goal_reached"]
    collision = eval_results["ego_collision"] or \
      eval_results["collision_driving_corridor"]
    step_count = eval_results["step_count"]
    # determine whether the simulation should terminate
    if success or collision or step_count > self._max_steps:
      done = True
    # calculate reward
    reward = collision * self._collision_penalty + \
      success * self._goal_reward
    return reward, done, eval_results
    
  def reset(self, world, agents_to_evaluate):
    return super(GoalReached, self).reset(world, agents_to_evaluate)