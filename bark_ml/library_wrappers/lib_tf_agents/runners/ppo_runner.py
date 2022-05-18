# Copyright (c) 2020 fortiss GmbH
#
# Authors: Patrick Hart
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import tensorflow as tf
tf.compat.v1.enable_v2_behavior()

from bark_ml.library_wrappers.lib_tf_agents.runners.tfa_runner import TFARunner


class PPORunner(TFARunner):
  """
  Used to train, evaluate and visualize a proximal policy optimization (PPO)
  agent.
  """

  def __init__(self,
               environment=None,
               agent=None,
               params=None):
    TFARunner.__init__(self,
                       environment=environment,
                       agent=agent,
                       params=params)

  def _train(self):
    global_iteration = self._agent._agent._train_step_counter.numpy()
    for i in range(1, self._params["ML"]["PPORunner"]["NumberOfCollections", "", 10000]):
      tf.keras.backend.clear_session()
      print(f"Collection {i}")
      global_iteration = self._agent._agent._train_step_counter.numpy()
      tf.summary.experimental.set_step(global_iteration)
      self._collection_driver.run()
      trajectories = self._agent._replay_buffer.gather_all()
      self._agent._agent.train(experience=trajectories)
      self._agent._replay_buffer.clear()
      if i % self._params["ML"]["PPORunner"]["EvaluateEveryNSteps", "", 100] == 0:
        self._tracer.Reset()
        self.Run(
          num_episodes=self._params["ML"]["TFARunner"]["EvaluationSteps", "", 20],
          mode="training")
        self._agent.Save()