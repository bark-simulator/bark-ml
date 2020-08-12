import sys
import logging
import time
import tensorflow as tf
import numpy as np
tf.compat.v1.enable_v2_behavior()

# BARK imports
from bark.runtime.commons.parameters import ParameterServer
from bark_ml.core.observers import NearestObserver

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
    for _ in range(
            0, self._params["ML"]["SACRunner"]
            ["NumberOfCollections", "", 1000000]):
      self._agent._training = True
      global_iteration = self._agent._agent._train_step_counter.numpy()
      self._collection_driver.run()
      experience, _ = next(iterator)
      self._agent._agent.train(experience)
      if global_iteration % self._params["ML"]["SACRunner"][
              "EvaluateEveryNSteps", "", 100] == 0:
        self.Evaluate()
        self._agent.Save()


class SACRunnerGenerator(SACRunner):
  """SAC Runner extension to generate expert trajectories
  """

  def __init__(self,
               environment=None,
               agent=None,
               params=None):
    """ constructor """
    SACRunner.__init__(self,
                       environment=environment,
                       agent=agent,
                       params=params)

    local_params = self._params
    local_params["ML"]["NearestObserver"]["NormalizationEnabled"] = False
    self.observer_not_normalized = NearestObserver(local_params)

  def GetStateNotNormalized(self):
    """Gets the current state observation using a not normalizing observer
    """
    eval_id = self._environment._scenario._eval_agent_ids[0]
    observed_world = self._environment._world.Observe([eval_id])[0]
    return self.observer_not_normalized.Observe(observed_world)

  def GenerateExpertTrajectories(
          self, num_trajectories: int = 1000, render: bool = False) -> dict:
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
      expert_trajectories = {'obs_norm': [], 'obs': [], 'act': []}

      # Merging Blueprint
      # X: 881.707/1006.72
      # Y: 1001.59/1010.82
      # Theta: 0/6.28
      # Velo: 0/50
      state = self._environment.reset()
      state_not_norm = self.GetStateNotNormalized()

      expert_trajectories['obs_norm'].append(state)
      expert_trajectories['obs'].append(state_not_norm)
      is_terminal = False

      while not is_terminal:
        action_step = self._agent._eval_policy.action(
          ts.transition(state, reward=0.0, discount=1.0))

        state, reward, is_terminal, info = self._environment.step(
            action_step.action.numpy())
        if not info:
          break

        state_not_norm = self.GetStateNotNormalized()
        expert_trajectories['obs_norm'].append(state)
        expert_trajectories['obs'].append(state_not_norm)
        expert_trajectories['act'].append(action_step.action.numpy())

        if render:
          self._environment.render()

      if info and info['goal_reached']:
        expert_trajectories['act'].append(expert_trajectories['act'][-1])
        assert len(
            expert_trajectories['obs_norm']) == len(
            expert_trajectories['act'])
        assert len(
            expert_trajectories['obs']) == len(
            expert_trajectories['act'])

        per_scenario_expert_trajectories[len(
          per_scenario_expert_trajectories)] = expert_trajectories
        print(
          f'Generated {len(per_scenario_expert_trajectories)}/{num_trajectories} expert trajectories.')
      else:
        print('Expert trajectory invalid. Skipping.')

    return per_scenario_expert_trajectories
