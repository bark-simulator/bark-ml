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


class TFAAgent(BaseAgent):
  """This class serves as a base class for all
     tf-agents agents. 
  
  Arguments:
      BaseAgent {BaseAgent} -- Abstract base class
  """
  def __init__(self,
               environment=None,
               params=None):
    BaseAgent.__init__(self,
                       environment=environment,
                       params=params)
    self._ckpt = tf.train.Checkpoint(step=tf.Variable(0, dtype=tf.int64))
    self._agent = self.get_agent(environment, params)
    self._ckpt = tf.train.Checkpoint(step=tf.Variable(0, dtype=tf.int64),
                                     agent=self._agent)
    self._ckpt_manager = self.get_checkpointer()

  def reset(self):
    pass

  def act(self):
    pass
  
  def get_checkpointer(self):
    """Checkpointer handling the saving and loading of agents
    
    Keyword Arguments:
        log_path {string} -- path to the checkpoints (default: {"/"})
    
    Returns:
        Checkpointer -- tf-checkpoint handler
    """
    checkpointer = Checkpointer(
      self._params["BaseDir", "Base directory", "."] + "/" + self._params["ML"]["Agent"]["checkpoint_path", "", ""],
      global_step=self._ckpt.step,
      tf_agent=self._agent,
      max_to_keep=self._params["ML"]["Agent"]["max_ckpts_to_keep", "", 3])
    checkpointer.initialize_or_restore()
    return checkpointer

  def save(self):
    """Saves the agent
    """
    save_path = self._ckpt_manager.save(
      global_step=self._agent._train_step_counter)
    print("Saved checkpoint for step {}.".format(
      int(self._agent._train_step_counter.numpy())))

  def load(self):
    """Loads the agent
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