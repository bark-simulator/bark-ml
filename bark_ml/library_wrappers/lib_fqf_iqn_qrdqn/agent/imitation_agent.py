# Copyright (c) 2020 fortiss GmbH
#
# Authors: Patrick Hart, Julian Bernhard, Klemens Esterle, and
# Tobias Kessler, Mansoor Nasir
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from collections import deque
import numpy as np
import logging
import os

import torch
from torch import nn
from torch.optim import Adam, RMSprop

from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.model import Imitation
from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.utils \
 import disable_gradients, update_params, \
 calculate_quantile_huber_loss, evaluate_quantile_at_action
from .base_agent import BaseAgent, TrainingBenchmark

class BenchmarkSupervisedLoss(TrainingBenchmark):
    def __init__(self, demonstrations_test):
      self.demonstrations_test = demonstrations_test

    def run(self):
      with torch.no_grad():
          states, action_values_desired = self.agent.sample_batch( \
                  self.demonstrations_test, self.agent.num_eval_episodes)
          action_values_current = self.agent.online_net(states)
          loss = self.agent.calculate_loss(action_values_desired, action_values_current)
          pair_wise_diff = action_values_current - action_values_desired
          mean_value_diffs = torch.mean(pair_wise_diff**2, dim = 0).tolist()

          return {"mse_loss/test" : loss.item()},\
               f"Loss = {loss.item()}\n Mean Diff: {mean_value_diffs}"

    def is_better(self, eval_result1, than_eval_result2):
        return eval_result1["mse_loss/test"] < than_eval_result2["mse_loss/test"]

class ImitationAgent(BaseAgent):
  def __init__(self, demonstrations_train, demonstrations_test, *args, **kwargs):
    super(ImitationAgent, self).__init__(*args, **kwargs)
    self._training_benchmark = BenchmarkSupervisedLoss(demonstrations_test)
    self._training_benchmark.reset(None, self.num_eval_episodes, None, self)
    self.demonstrations_train = demonstrations_train
    
  def reset_params(self, params):
    super(ImitationAgent, self).reset_params(params)
    self.running_loss_length = params["RunningLossLength", "", 1000]
    self.num_value_functions = params["NumValueFunctions", "", 3]
    self.learning_rate = params["LearningRate", "", 0.001]

  def reset_training_variables(self):
    # Replay memory which is memory-efficient to store stacked frames.
    self.running_loss = deque(maxlen=self.running_loss_length)
    self.steps = 0
    self.best_eval_results = None
  
  def init_always(self):
    super(ImitationAgent, self).init_always()
    
    # Target network.
    self.online_net = Imitation(num_channels=self.observer.observation_space.shape[0],
                          num_actions=self.num_actions,
                          num_value_functions=self.num_value_functions,
                          params=self._params).to(self.device)
    self.optim = RMSprop(
        self.online_net.parameters(),
        lr=self.learning_rate,
        alpha=0.95,
        eps=0.00001)

  def clean_pickables(self, pickables):
    del pickables["_env"]
    del pickables["_training_benchmark"]
    del pickables["device"]
    del pickables["writer"]
    del pickables["online_net"]
    del pickables["optim"]
    del pickables["demonstrations_train"]

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

  def calculate_actions(self, state):
    with torch.no_grad():
        state_torch = torch.FloatTensor(state).to(self.device)
        action_values = self.online_net(state_torch)
    return action_values.tolist()

  def save_models(self, checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)
    torch.save(self.online_net.state_dict(),
               os.path.join(checkpoint_dir, 'online_net.pth'))
    online_net_script = torch.jit.script(self.online_net)
    online_net_script.save(os.path.join(checkpoint_dir, 'online_net_script.pt'))

  @property
  def script_model_file_name(self):
    return os.path.join(checkpoint_dir, 'online_net_script.pt')

  def load_models(self, checkpoint_dir):
    try: 
      self.online_net.load_state_dict(
        torch.load(os.path.join(checkpoint_dir, 'online_net.pth')))
    except RuntimeError:
      self.online_net.load_state_dict(
        torch.load(os.path.join(checkpoint_dir, 'online_net.pth'), map_location=torch.device('cpu')))

  def train_episode(self):
    states, action_values_desired = self.sample_batch(self.demonstrations_train, self.batch_size)

    self.optim.zero_grad()
    action_values_current = self.online_net(states)
    loss = self.calculate_loss(action_values_desired, action_values_current)
    loss.backward()
    self.optim.step()

    self.running_loss.append(loss.item())
    # We log evaluation results along with training frames = 4 * steps.
    if self.steps % self.summary_log_interval == 0:
        self.writer.add_scalar('mse_loss/train', sum(self.running_loss)/len(self.running_loss),
                                self.steps)
        logging.info(f"Training: Loss(i={self.steps}={sum(self.running_loss)/len(self.running_loss)})")
    if self.steps % self.eval_interval == 0:
      self.evaluate()
      self.save("final")
      self.online_net.train() 
    self.steps += 1

    

