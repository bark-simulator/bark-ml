import numpy as np
import tensorflow as tf
import logging

# BARK imports
from bark.core.models.behavior import BehaviorModel, BehaviorDynamicModel

# tfa
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import normal_projection_network
from tf_agents.agents.ddpg import critic_network
from tf_agents.policies import greedy_policy
from tf_agents.environments import tf_py_environment
from tf_agents.agents.sac import sac_agent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils.common import Checkpointer
from tf_agents.trajectories import time_step as ts

# BARK-ML imports
from bark_ml.library_wrappers.lib_tf_agents.tfa_wrapper import TFAWrapper
from bark_ml.commons.py_spaces import BoundedContinuous


class BehaviorTFAContAgent(BehaviorModel):
  def __init__(self,
               environment=None,
               params=None):
    BehaviorModel.__init__(self, params)
    self._params = params
    self._lower_bounds = params["ML"]["BehaviorContinuousML"][
      "ActionsLowerBound",
      "Lower-bound for actions.",
      [-5.0, -0.2]]
    self._upper_bounds = params["ML"]["BehaviorContinuousML"][
      "ActionsUpperBound",
      "Upper-bound for actions.",
      [4.0, 0.2]]
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
    self._bark_behavior_model = BehaviorDynamicModel(params)

  def Reset(self):
    pass

  def Act(self, state):
    pass
  
  def GetCheckpointer(self):
    checkpointer = Checkpointer(
        self._params["ML"]["BehaviorTFAContAgents"]["CheckpointPath", "", ""],
      global_step=self._ckpt.step,
      tf_agent=self._agent,
      max_to_keep=self._params["ML"]["BehaviorTFAContAgents"]["NumCheckpointsToKeep", "", 3])
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

  def Act(self, state):
    # NOTE: this is the greedy policy
    action_step = self.eval_policy.action(
      ts.transition(state, reward=0.0, discount=1.0))
    return action_step.action.numpy()

  def Plan(self, dt, observed_world):
    # NOTE: if training is enabled the action is set from the outside
    if not self._training:
      observed_state = self._environment._observer.Observe(
        observed_world)
      action = self.Act(observed_state)
      # NOTE: remove reshape
      self._bark_behavior_model.ActionToBehavior(np.reshape(action, [-1, 1]))
    trajectory = self._bark_behavior_model.Plan(dt, observed_world)
    BehaviorModel.SetLastTrajectory(self, trajectory)
    return trajectory

  @property
  def action_space(self):
    return BoundedContinuous(
      2, # acceleration and steering-rate
      low=np.array(self._lower_bounds, dtype=np.float32),
      high=np.array(self._upper_bounds, dtype=np.float32))

  def Clone(self):
    return self