import sys
import logging
import time
import tensorflow as tf
import numpy as np
tf.compat.v1.enable_v2_behavior()

# BARK imports
from bark.runtime.commons.parameters import ParameterServer

# tf agent imports
from tf_agents.drivers import dynamic_step_driver
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.metrics import tf_metrics
from tf_agents.eval import metric_utils
from tf_agents.utils import common
from tf_agents.trajectories import time_step as ts

# BARK-ML imports
from bark_ml.library_wrappers.lib_tf_agents.runners.tfa_runner import TFARunner


class SACRunner(TFARunner):
  def __init__(self,
               environment=None,
               agent=None,
               params=None):
    TFARunner.__init__(self,
                       environment=environment,
                       agent=agent,
                       params=params)

  def _train(self):
    iterator = iter(self._agent._dataset)
    for _ in range(0, self._params["ML"]["SACRunner"]["NumberOfCollections", "", 1000000]):
      self._agent._training = True
      global_iteration = self._agent._agent._train_step_counter.numpy()
      self._collection_driver.run()
      experience, _ = next(iterator)
      self._agent._agent.train(experience)
      if global_iteration % self._params["ML"]["SACRunner"]["EvaluateEveryNSteps", "", 100] == 0:
        self.Evaluate()
        self._agent.Save()
  
class SACRunnerGenerator(SACRunner):
  """SAC Runner extension to generate expert trajectories
  """
  def _GetBounds(self):
    return [
        self._environment._observer._world_x_range,
        self._environment._observer._world_y_range,
        self._environment._observer._ThetaRange,
        self._environment._observer._VelocityRange
        ]  
  
  def _DenormalizeState(self, state):
    bounds = self._GetBounds()
    return [state[i] * (bounds[i % 4][1] - bounds[i % 4][0]) + bounds[i % 4][0] for i in range(len(state))]
    
  def GenerateExpertTrajectories(self, num_trajectories: int = 1000, render: bool = False) -> dict:
    """Generates expert trajectories based on a tfa agent.

    Args:
        num_trajectories (int, optional): The minimal number of generated expert trajectories. Defaults to 1000. 
        render (bool, optional): Render the simulation during simulation. Defaults to False.

    Returns:
        dict: The expert trajectories.
    """
    self._agent._training = False
    per_scenario_expert_trajectories = {}

    while len(per_scenario_expert_trajectories) < num_trajectories:
      expert_trajectories = {'obs': [], 'act': []}

      state = self._environment.reset()
      expert_trajectories['obs'].append(state)
      is_terminal = False

      while not is_terminal:
        action_step = self._agent._eval_policy.action(ts.transition(state, reward=0.0, discount=1.0))

        state, reward, is_terminal, info = self._environment.step(action_step.action.numpy())
        expert_trajectories['obs'].append(state)
        expert_trajectories['act'].append(action_step.action.numpy())

        if render:
          self._environment.render()

      if info and info['goal_reached']:
        expert_trajectories['act'].append(expert_trajectories['act'][-1])
        assert len(expert_trajectories['obs']) == len(expert_trajectories['act'])

        per_scenario_expert_trajectories[len(per_scenario_expert_trajectories)] = expert_trajectories
        print(f'Generated {len(per_scenario_expert_trajectories)}/{num_trajectories} expert trajectories.')
      else:
        print('Expert trajectory invalid. Skipping.')

    return per_scenario_expert_trajectories