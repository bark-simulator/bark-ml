import tensorflow as tf

# tfa
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import normal_projection_network
from tf_agents.agents.ddpg import critic_network
from tf_agents.policies import greedy_policy

from tf_agents.agents.sac import sac_agent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils.common import Checkpointer

from src.agents.base_agent import BaseAgent


class TFAgent(BaseAgent):
  """TFAgent
     This class handles checkpoints and TF specific
     functionalities
  
  Arguments:
      BaseAgent {BaseAgent} -- Abstract base class
  
  """
  def __init__(self,
               environment=None,
               params=None):
    BaseAgent.__init__(self,
                       params=params)
    self._ckpt = tf.train.Checkpoint(step=tf.Variable(0, dtype=tf.int64))
    self._env = environment
    self._ckpt_manager  = self.get_checkpointer()

  def reset(self):
    pass

  def get_checkpointer(self, log_path="/"):
    """Checkpointer for handling saving and loading of agents
    
    Keyword Arguments:
        log_path {string} -- path to the checkpoints (default: {"/"})
    
    Returns:
        Checkpointer -- tf checkpoint handler
    """
    checkpointer = Checkpointer(log_path,
      global_step=self._ckpt.step,
      tf_agent=self._agent,
      max_to_keep=self._params["ML"]["Agent"]["max_ckpts_to_keep"])
    checkpointer.initialize_or_restore()
    return checkpointer

  def save(self):
    """Saves an agent
    """
    save_path = self._ckpt_manager.save()
    print("Saved checkpoint for step {}: {}".format(int(self._ckpt.step) + 1,
                                                    save_path))

  def load(self):
    """Loads an agent
    """
    try:
      self._ckpt.restore(self._ckpt_manager.latest_checkpoint)
    except:
      return RuntimeError("Could not load agent.")
    if self._ckpt_manager.latest_checkpoint:
      print("Restored agent from {}".format(
        self._ckpt_manager.latest_checkpoint))
    else:
      print("Initializing agent from scratch.")