import sys
import logging
import time
import tensorflow as tf
tf.compat.v1.enable_v2_behavior()

# BARK imports
#from bark_project.modules.runtime.commons.parameters import ParameterServer

# tf2rl imports
import tf2rl

# BARK-ML imports


class TF2RLRunner:
  """Base class for runners based on tf2rl library"""

  def __init__(self,
                environment=None,
                agent=None,
                params=None):
      
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
    # does not realy matter, it will only be written in the summary writer.
    # it has got a meaning during training, where the same method is used for evaluation, and
    # there in the summary writer it really makes sense to keep track of the step number.
    total_steps = 0   
    self._trainer.evaluate_policy(total_steps=total_steps)

    

  def Visualize(self, num_episodes=1):
    """Visualizes the agent."""

    for _ in range(0, num_episodes):
      obs = self._environment.reset()
      is_terminal = False

      while not is_terminal:
        print(obs)
        action = self._agent.Act(obs)
        obs, reward, is_terminal, _ = self._environment.step(action)
        print(reward)
        self._environment.render()

  
