import tensorflow as tf
import logging

# BARK imports
from bark.core.models.behavior import BehaviorModel

# tfa
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import normal_projection_network
from tf_agents.agents.ddpg import critic_network
from tf_agents.policies import greedy_policy
from tf_agents.environments import tf_py_environment
from tf_agents.agents.sac import sac_agent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils.common import Checkpointer

# BARK-ML imports
from bark_ml.library_wrappers.lib_tf_agents.tfa_wrapper import TFAWrapper


# TODO(@hart): pass individual observer?
class BehaviorTFAAgent:
  def __init__(self,
               environment=None,
               params=None):
    self._params = params
    self._environment = environment
    self._wrapped_env = tf_py_environment.TFPyEnvironment(
      TFAWrapper(self._environment))
    self._ckpt = tf.train.Checkpoint(step=tf.Variable(0, dtype=tf.int64))
    self._agent = self.GetAgent(self._wrapped_env, params)
    self._ckpt = tf.train.Checkpoint(step=tf.Variable(0, dtype=tf.int64),
                                     agent=self._agent)
    self._ckpt_manager = self.GetCheckpointer()
    self._logger = logging.getLogger()
    self._training = False

  def Reset(self):
    pass

  def Act(self, state):
    pass

  def GetCheckpointer(self):
    checkpointer = Checkpointer(
        self._params["ML"]["BehaviorTFAAgents"]["CheckpointPath", "", ""],
      global_step=self._ckpt.step,
      tf_agent=self._agent,
      max_to_keep=self._params["ML"]["BehaviorTFAAgents"]["NumCheckpointsToKeep", "", 3])
    checkpointer.initialize_or_restore()
    return checkpointer

  def Save(self):
    save_path = self._ckpt_manager.save(
      global_step=self._agent._train_step_counter)
    self._logger.info("Saved checkpoint for step {}.".format(
      int(self._agent._train_step_counter.numpy())))

  def Load(self):
    try:
      self._ckpt.restore(self._ckpt_manager.latest_checkpoint)
    except:
      return RuntimeError("Could not load agent.")
    if self._ckpt_manager.latest_checkpoint:
      self._logger.info("Restored agent from {}".format(
        self._ckpt_manager.latest_checkpoint))
    else:
      self._logger.info("Initializing agent from scratch.")