import sys
import logging
import time
import tensorflow as tf
tf.compat.v1.enable_v2_behavior()
from IPython.display import clear_output
import matplotlib.pyplot as plt
from bark_ml.library_wrappers.lib_tf2rl.compare_trajectories import *

# tf2rl imports
import tf2rl
from pprint import pprint

class TF2RLRunner:
  """Base class for runners based on tf2rl library"""

  def __init__(self,
                environment=None,
                agent=None,
                params=None):
    """TF2RL base class initialization"""
      
    self._params = params
    self._agent = agent
    self._environment = environment

    self._trainer = self.GetTrainer()


  def GetTrainer(self):
    """Creates an trainer instance. Agent specific."""
    raise NotImplementedError

  def Train(self):
    """trains the agent."""
    self._train()


  def _train(self):
    """agent specific training method."""
    raise NotImplementedError

  def _PrintEvaluationResults(self, message: str, agent, expert = None):
    print('*' * 80)
    print(message)
    print('*' * 80)
    print('Agent:      \t', agent)
    if expert is not None:
      print('Expert:     \t', expert)

      abs_difference = np.abs(agent - expert)
      abs_difference = np.abs(agent - expert)
      print('Difference: \t', abs_difference)

      deviation_agent = np.abs(abs_difference / agent)
      deviation_expert = np.abs(abs_difference / expert)
      deviation = np.maximum(deviation_agent, deviation_expert) 
      print('Deviation:  \t', deviation)
    print('')

  def _EvaluateMeanActions(self, expert_trajectories: dict, avg_trajectory_length: float, num_trajectories: int):
    self._trainer._test_episodes = num_trajectories
    avg_test_return, agent_trajectories, avg_step_count = self._trainer.evaluate_policy(total_steps=0)

    self._PrintEvaluationResults('Average test return: ', avg_test_return)
    self._PrintEvaluationResults('Average step count: ', avg_step_count, avg_trajectory_length)

    agent_actions = {'acts': [],}
    for trajectory in agent_trajectories:
      agent_actions['acts'].extend(trajectory['act'])  

    expert_mean_action = calculate_mean_action(expert_trajectories)
    agent_mean_action = calculate_mean_action(agent_actions)
    self._PrintEvaluationResults('Mean action: ', agent_mean_action, expert_mean_action)


  def Evaluate(self, expert_trajectories: dict, avg_trajectory_length: float, num_trajectories: int):
    """Evaluates the agent."""
    self._EvaluateMeanActions(expert_trajectories, avg_trajectory_length, num_trajectories)

  def Visualize(self, num_episodes=1, renderer=""):
    """Visualizes the agent."""

    if renderer == "matplotlib_viewer":
      self._get_viewer(param_server=self._params, renderer=renderer)

    for _ in range(0, num_episodes):
      obs = self._environment.reset()
      is_terminal = False

      while not is_terminal:
        action = self._agent.Act(obs)
        obs, reward, is_terminal, _ = self._environment.step(action)
        world_state = self._environment._scenario.GetWorldState()

        if renderer == 'matplotlib_jupyter':
          self._get_viewer(param_server=self._params, renderer=renderer)
          clear_output(wait=True)
          self._viewer.clear()
          self._viewer.drawWorld(
            world_state,
            self._environment._scenario._eval_agent_ids,
            scenario_idx=self._environment._scenario_idx)
          self._viewer.clear()
        else:
          self._environment.render()


  def _get_viewer(self, param_server: None, renderer: str):
    """Getter for a viewer to display the simulation.

    Args:
        param_server (ParameterServer): The parameters that specify the scenario.
        renderer (str): The renderer type used. [pygame, matplotlib]

    Returns:
        bark.runtime.viewer.Viewer: A viewer depending on the renderer type
    """
    fig = plt.figure(figsize=[10, 10])

    if renderer == "pygame":
        from bark.runtime.viewer.pygame_viewer import PygameViewer
        self._viewer = PygameViewer(params=param_server,
                              use_world_bounds=True, axis=fig.gca())
    else:
        from bark.runtime.viewer.matplotlib_viewer import MPViewer
        self._viewer = MPViewer(params=param_server,
                          use_world_bounds=True, axis=fig.gca())

  

