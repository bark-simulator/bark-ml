import sys
import logging
import time
import tensorflow as tf
tf.compat.v1.enable_v2_behavior()
from IPython.display import clear_output
import matplotlib.pyplot as plt

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


  def Evaluate(self):
    """Evaluates the agent."""
    total_steps = 0   
    avg_test_return, trajectories, avg_step_count = self._trainer.evaluate_policy(total_steps=total_steps)
    print('Average test return: ', avg_test_return)
    print('Average step count: ', avg_step_count)
    pprint('Trajectories: ', trajectories)

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

  

