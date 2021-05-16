# Copyright (c) 2020 fortiss GmbH
#
# Authors: Patrick Hart
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import numpy as np
import tensorflow as tf
import logging

# BARK imports
from bark.core.models.behavior import BehaviorModel

# tfa
from tf_agents.environments import tf_py_environment
from tf_agents.utils.common import Checkpointer
from tf_agents.trajectories import time_step as ts

# BARK-ML imports
from bark_ml.library_wrappers.lib_tf_agents.py_bark_environment import PyBARKEnvironment
from bark_ml.commons.py_spaces import BoundedContinuous
from bark_ml.behaviors.cont_behavior import BehaviorContinuousML


class BehaviorTFAAgent(BehaviorModel):
  """TF-Agents agent for BARK."""

  def __init__(self,
               environment=None,
               params=None,
               bark_behavior=None,
               observer=None):
    BehaviorModel.__init__(self, params)
    self._params = params
    self._observer = observer
    self._environment = environment
    self._wrapped_env = tf_py_environment.TFPyEnvironment(
      PyBARKEnvironment(self._environment))
    self._ckpt = tf.train.Checkpoint(step=tf.Variable(0, dtype=tf.int64))
    self._agent = self.GetAgent(self._wrapped_env, params)
    self._ckpt = tf.train.Checkpoint(step=tf.Variable(0, dtype=tf.int64),
                                     agent=self._agent)
    self._ckpt_manager = self.GetCheckpointer()
    self._logger = logging.getLogger()
    # NOTE: by default we do not want the action to be set externally
    #       as this enables the agents to be plug and played in BARK.
    self._set_action_externally = False
    self._bark_behavior_model = bark_behavior or BehaviorContinuousML(params)

  def Reset(self):
    pass

  @property
  def set_action_externally(self):
    return self._set_action_externally

  @set_action_externally.setter
  def set_action_externally(self, externally):
    # if externally:
    #   self._logger.info("Actions are now set externally.")
    self._set_action_externally = externally

  def GetCheckpointer(self):
    checkpointer = Checkpointer(
        self._params["ML"]["BehaviorTFAAgents"]["CheckpointPath", "", ""],
      global_step=self._ckpt.step,
      tf_agent=self._agent,
      max_to_keep=self._params["ML"]["BehaviorTFAAgents"][
        "NumCheckpointsToKeep", "", 3])
    checkpointer.initialize_or_restore()
    return checkpointer

  def Save(self):
    self._ckpt_manager.save(
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
    # NOTE: greedy action
    action_step = self.eval_policy.action(
      ts.transition(state, reward=0.0, discount=1.0))
    return action_step.action.numpy()

  def ActionToBehavior(self, action):
    # NOTE: will either be set externally or internally
    self._action = action

  def Plan(self, dt, observed_world):
    # NOTE: if training is enabled the action is set externally
    if not self._set_action_externally:
      # NOTE: we need to store the observer differently
      observed_state = self._environment._observer.Observe(
        observed_world)
      self._action = self.Act(observed_state)
    # NOTE: BARK expects (m, 1) actions
    action = self._action
    if isinstance(self.action_space, BoundedContinuous):
      action = np.reshape(self._action, (-1, 1))
    # set action to be executed
    self._bark_behavior_model.ActionToBehavior(action)
    trajectory = self._bark_behavior_model.Plan(dt, observed_world)
    # NOTE: BARK requires models to have trajectories of the past
    BehaviorModel.SetLastTrajectory(self, trajectory)
    return trajectory

  @property
  def action_space(self):
    return self._bark_behavior_model.action_space

  def Clone(self):
    return self
