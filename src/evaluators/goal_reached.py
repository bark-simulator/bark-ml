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
      self._params["Runtime"]["RL"]["StateEvaluator"]["GoalReward",
        "Reward for reaching the goal.",
        0.01]
    self._collision_reward = \
      self._params["Runtime"]["RL"]["StateEvaluator"]["CollisionReward",
        "Reward given for a collisions.",
        -1.]
    self._max_steps = \
      self._params["Runtime"]["RL"]["StateEvaluator"]["MaxSteps",
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

  def evaluate(self, world):
    eval_results = None
    reward = 0.
    done = False
    if self._eval_agent in world.agents:
      eval_results = world.evaluate()
      success = eval_results["goal_reached"]
      collision = eval_results["ego_collision"] or \
        eval_results["collision_driving_corridor"]
      step_count = eval_results["step_count"]
      # determine whether the simulation should terminate
      if success or collision or step_count > self._max_steps:
        done = True

      # TODO(@hart): determine reward
      """
      collision = eval_results["collision_agents"] or \
        eval_results["collision_driving_corridor"]
      success = eval_results["success"]
      reward = collision * self._collision_reward + \
        success * self._goal_reward
      max_steps_reached = eval_results["step_count"] > self._max_steps
      done = success or collision or max_steps_reached
      info = {"success": success,
              "collision_agents": eval_results["collision_agents"], 
              "collision_driving_corridor": \
                eval_results["collision_driving_corridor"],
              "outside_map": False,
              "num_steps": eval_results["step_count"]}
      """
    return reward, done, eval_results
      
  def reset(self, world, agents_to_evaluate):
    if len(agents_to_evaluate) != 1:
      raise ValueError("Invalid number of agents provided for GoalReached \
                        evaluation, number= {}" \
                        .format(len(agents_to_evaluate)))
    self._eval_agent = agents_to_evaluate[0]
    world.clear_evaluators()
    self._add_evaluators()
    for key, evaluator in self._evaluators.items():
      world.add_evaluator(key, evaluator)
    return world
    

