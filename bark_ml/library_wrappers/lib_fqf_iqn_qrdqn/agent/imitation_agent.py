# Copyright (c) 2020 fortiss GmbH
#
# Authors: Patrick Hart, Julian Bernhard, Klemens Esterle, and
# Tobias Kessler, Mansoor Nasir
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# The code is adapted from opensource implementation - https://github.com/ku2482/fqf-iqn-qrdqn.pytorch
# MIT License -Copyright (c) 2020 Toshiki Watanabe

from collections import deque
import numpy as np
import logging


import torch
from torch import nn
from torch.optim import Adam, RMSprop

from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.model import Imitation
from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.utils \
 import disable_gradients, update_params, \
 calculate_quantile_huber_loss, evaluate_quantile_at_action
from .base_agent import BaseAgent


class ImitationAgent(BaseAgent):
  def __init__(self, demonstrations_train, demonstrations_test, *args, **kwargs):
    super(ImitationAgent, self).__init__(*args, **kwargs)
    self.demonstrations_train = demonstrations_train
    self.demonstrations_test = demonstrations_test

  def reset_params(self, params):
    self.num_train_episodes = params["NumTrainEpisodes", "", 1000]
    self.batch_size = params["BatchSize", "", 32]
    self.eval_batch_size = params["EvalBatchSize", "", 1024]
    self.eval_every_updates = params["EvalEveryUpdates", "", 4000]
    self.max_episode_steps = params["NotUsed", "", 30]
    self.grad_cliping = params["GradCliping", "", 5.0]
    self.use_cuda = params["Cuda", "", False] 
    self.running_loss_length = params["RunningLossLength", "", 1000]
    self.learning_rate = params["LearningRate", "", 0.001]
    self.num_value_functions = params["NumValueFunctions", "", 3]
    self.num_eval_episodes = params["NotUsed", "", 3]

    self.summary_log_interval = params["SummaryLogInterval", "", 100]

  def reset_training_variables(self):
    # Replay memory which is memory-efficient to store stacked frames.
    self.update_steps = 0
    self.running_loss = deque(maxlen=self.running_loss_length)
    self.best_eval_results = None
  
  def init_always(self):
    super(ImitationAgent, self).init_always()
    
    # Target network.
    self.imitation_net = Imitation(num_channels=self.observer.observation_space.shape[0],
                          num_actions=self.num_actions,
                          num_value_functions=self.num_value_functions,
                          params=self._params).to(self.device)
    self.optim = RMSprop(
        self.imitation_net.parameters(),
        lr=self.learning_rate,
        alpha=0.95,
        eps=0.00001)

  def clean_pickables(self, pickables):
    del pickables["_env"]
    del pickables["_training_benchmark"]
    del pickables["device"]
    del pickables["writer"]
    del pickables["imitation_net"]
    del pickables["optim"]
    del pickables["demonstrations_train"]
    del pickables["demonstrations_test"]

  def sample_batch(self, demonstrations_list, batch_size):
    state_size = len(demonstrations_list[0][0])
    num_action_values = len(demonstrations_list[0][1])
    states = np.empty((batch_size, state_size), dtype=np.float)
    action_values = np.empty((batch_size, num_action_values), dtype=np.float)

    indices = np.random.randint(low=0, high=len(demonstrations_list), size=batch_size)
    for i, index in enumerate(indices):
      states[i, ...] = demonstrations_list[index][0]
      action_values[i, ...] = demonstrations_list[index][1]

    states = torch.FloatTensor(states).to(self.device)
    action_values = torch.FloatTensor(action_values).to(self.device)

    return states, action_values

  def calculate_loss(self, action_values_desired, action_values_current):
    # if we have missing actions during demo collections these become nan values
    # they should influence the loss and are thus set to current nn output
    nans_desired = torch.isnan(action_values_desired).nonzero()
    action_values_desired[nans_desired] = action_values_current[nans_desired]
    criterion = nn.MSELoss()
    loss = criterion(action_values_current, action_values_desired)

    return loss

  def save_models(self, checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)
    torch.save(self.imitation_net.state_dict(),
               os.path.join(checkpoint_dir, 'imitation_net.pth'))

  def load_models(self, checkpoint_dir):
    try: 
      self.imitation_net.load_state_dict(
        torch.load(os.path.join(checkpoint_dir, 'imitation_net.pth')))
    except RuntimeError:
      self.imitation_net.load_state_dict(
        torch.load(os.path.join(checkpoint_dir, 'imitation_net.pth'), map_location=torch.device('cpu')))

  def run(self, num_episodes=None):
    for epoch in range(num_episodes or self.num_train_episodes):
        states, action_values_desired = self.sample_batch(self.demonstrations_train, self.batch_size)

        self.optim.zero_grad()
        action_values_current = self.imitation_net(states)
        loss = self.calculate_loss(action_values_desired, action_values_current)
        loss.backward()
        self.optim.step()

        self.running_loss.append(loss.item())
        # We log evaluation results along with training frames = 4 * steps.
        if self.update_steps % self.summary_log_interval == 0:
            self.writer.add_scalar('mse_loss/train', sum(self.running_loss)/len(self.running_loss),
                                    self.update_steps)
            logging.info(f"Training: Loss(i={self.update_steps}={sum(self.running_loss)/len(self.running_loss)}")
        if self.update_steps % self.eval_every_updates == 0:
            self.eval()
        self.update_steps += 1

  def eval(self, eval_batch_size=None):
    with torch.no_grad():
          states, action_values_desired = self.sample_batch(self.demonstrations_test, self.eval_batch_size)
          action_values_current = self.imitation_net(states)
          loss = self.calculate_loss(action_values_desired, action_values_current)

          self.writer.add_scalar('mse_loss/test', loss.item(),
                                  self.update_steps)
          logging.info(f"Test: Loss(i={self.update_steps}={loss.item()}")
    

    

