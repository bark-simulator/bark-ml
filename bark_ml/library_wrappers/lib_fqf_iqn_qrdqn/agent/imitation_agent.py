# Copyright (c) 2020 fortiss GmbH
#
# Authors: Patrick Hart, Julian Bernhard, Klemens Esterle, and
# Tobias Kessler, Mansoor Nasir
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# The code is adapted from opensource implementation - https://github.com/ku2482/fqf-iqn-qrdqn.pytorch
# MIT License -Copyright (c) 2020 Toshiki Watanabe

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
    self.num_epochs = params["NumEpochs", "", 5]
    self.batch_size = params["BatchSize", "", 32]
    self.grad_cliping = params["GradCliping", "", 5.0]
    self.use_cuda = params["Cuda", "", False] 

    self.summary_log_interval = params["SummaryLogInterval", "", 100]

  
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

  def eval(self):
    pass

  def sample_batch(self, demonstrations_list, batch_size):
    state_size = len(demonstrations_list[0][0])
    num_action_values = len(demonstrations_list[0][1])
    states = np.empty((batch_size, state_size), dtype=np.float)
    action_values = np.empty((batch_size, num_action_values), dtype=np.float)

    indices = np.random.randint(low=0, high=len(self), size=batch_size)
    for i, index in enumerate(indices):
      states[i, ...] = demonstrations_list[index][0]
      action_values[i, ...] = demonstrations_list[index][1]

    states = torch.FloatTensor(states).to(self.device)
    action_values = torch.FloatTensor(self['action'][indices]).to(self.device)

    return states, action_values

  def run(self):
    update_steps = 0
    running_loss = 0.0
    for epoch in range(self.num_epochs):
        states, action_values_desired = self.sample_batch(self.demonstrations_train, self.batch_size)

        self.optim.zero_grad()

        action_values_current = self.imitation_net(states)

        criterion = nn.MSELoss()
        loss = criterion(action_values_current, action_values_desired)

        self.optim.step()
        running_loss += self.optim.item()
        # We log evaluation results along with training frames = 4 * steps.
        if i % self.summary_log_interval == 0:
            self.writer.add_scalar('mse_loss', 
                                    4 * self.steps)
    

    

