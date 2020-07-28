# Copyright (c) 2020 Patrick Hart, Julian Bernhard,
# Klemens Esterle, Tobias Kessler
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from bark.core.world.evaluation import \
  EvaluatorGoalReached, EvaluatorCollisionEgoAgent, \
  EvaluatorStepCount, EvaluatorDrivableArea
from bark.runtime.commons.parameters import ParameterServer

from bark_ml.evaluators.evaluator import StateEvaluator


class GoalReachedGuiding(StateEvaluator):
  def __init__(self,
               params=ParameterServer(),
               eval_agent=None):
    StateEvaluator.__init__(self, params)
    self._goal_reward = \
      self._params["ML"]["GoalReachedEvaluator"]["GoalReward",
        "Reward for reaching the goal.",
        1.]
    self._col_penalty = \
      self._params["ML"]["GoalReachedEvaluator"]["CollisionPenalty",
        "Reward given for a collisions.",
        -1.]
    self._max_steps = \
      self._params["ML"]["GoalReachedEvaluator"]["MaxSteps",
        "Maximum steps per episode.",
        50]
    self._eval_agent = eval_agent

  def _add_evaluators(self):
    self._evaluators["goal_reached"] = EvaluatorGoalReached()
    self._evaluators["collision"] = EvaluatorCollisionEgoAgent()
    self._evaluators["step_count"] = EvaluatorStepCount()
    self._evaluators["drivable_area"] = EvaluatorDrivableArea()

  def CalculateGuidingReward(self, observed_world, action):
    ego_agent = observed_world.ego_agent
    goal_def = ego_agent.goal_definition
    goal_center_line = goal_def.center_line
    ego_agent_state = ego_agent.state
    lateral_offset = Distance(goal_center_line,
                              Point2d(ego_agent_state[1], ego_agent_state[2]))

    actions = np.reshape(action, (-1, 2))
    accs = actions[:, 0]
    delta = actions[:, 1]
    return 0.001*lateral_offset**2 + 0.001*inpt_reward

  def _evaluate(self, observed_world, eval_results, action):
    """Returns information about the current world state
    """
    done = False
    success = eval_results["goal_reached"]
    collision = eval_results["collision"] or eval_results["drivable_area"]
    step_count = eval_results["step_count"]
    # determine whether the simulation should terminate
    if success or collision or step_count > self._max_steps:
      done = True
    guiding_reward = CalculateGuidingReward(observed_world, action)
    # calculate reward
    reward = collision * self._col_penalty + \
      success * self._goal_reward + guiding_reward
    return reward, done, eval_results
    
  def Reset(self, world):
    return super(GoalReached, self).Reset(world)