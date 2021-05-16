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
from torch.optim import Adam

from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.model import QRDQN
from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.utils \
 import disable_gradients, update_params, \
 calculate_quantile_huber_loss, evaluate_quantile_at_action
from .base_agent import BaseAgent


class QRDQNAgent(BaseAgent):
  """QRDQNAgent that can be used in BARK and BARK-ML."""

  def __init__(self, env, test_env, params):
    super(QRDQNAgent, self).__init__(env, test_env, params)

    self.N = self._params["ML"]["QRDQNAgent"]["N", "", 200]
    self.kappa = self._params["ML"]["QRDQNAgent"]["Kappa", "", 1.0]

    # Online network.
    self.online_net = QRDQN(num_channels=env.observation_space.shape[0],
                            num_actions=self.num_actions,
                            N=self.N,
                            dueling_net=self.dueling_net,
                            noisy_net=self.noisy_net,
                            params=self._params).to(self.device)
    # Target network.
    self.target_net = QRDQN(num_channels=env.observation_space.shape[0],
                            num_actions=self.num_actions,
                            N=self.N,
                            dueling_net=self.dueling_net,
                            noisy_net=self.noisy_net,
                            params=self._params).to(self.device).to(
                                self.device)

    # Copy parameters of the learning network to the target network.
    self.update_target()
    # Disable calculations of gradients of the target network.
    disable_gradients(self.target_net)

    self.optim = Adam(self.online_net.parameters(),
                      lr=self._params["ML"]["QRDQNAgent"]["LearningRate", "",
                                                          5e-5],
                      eps=1e-2 / self.batch_size)

    # Fixed fractions.
    taus = torch.arange(0, self.N + 1, device=self.device,
                        dtype=torch.float32) / self.N
    self.tau_hats = ((taus[1:] + taus[:-1]) / 2.0).view(1, self.N)

  def learn(self):
    self.learning_steps += 1
    self.online_net.sample_noise()
    self.target_net.sample_noise()

    if self.use_per:
      (states, actions, rewards, next_states, dones), weights = \
       self.memory.sample(self.batch_size)
    else:
      states, actions, rewards, next_states, dones = \
       self.memory.sample(self.batch_size)
      weights = None

    quantile_loss, mean_q, errors = self.calculate_loss(
        states, actions, rewards, next_states, dones, weights)

    update_params(self.optim,
                  quantile_loss,
                  networks=[self.online_net],
                  retain_graph=False,
                  grad_cliping=self.grad_cliping)

    if self.use_per:
      self.memory.update_priority(errors)

    if 4 * self.steps % self.summary_log_interval == 0:
      self.writer.add_scalar('loss/quantile_loss',
                             quantile_loss.detach().item(), 4 * self.steps)
      self.writer.add_scalar('stats/mean_Q', mean_q, 4 * self.steps)

  def calculate_loss(self, states, actions, rewards, next_states, dones,
                     weights):

    # Calculate quantile values of current states and actions at taus.
    current_sa_quantiles = evaluate_quantile_at_action(
        self.online_net(states=states), actions)
    assert current_sa_quantiles.shape == (self.batch_size, self.N, 1)

    with torch.no_grad():
      # Calculate Q values of next states.
      if self.double_q_learning:
        # Sample the noise of online network to decorrelate between
        # the action selection and the quantile calculation.
        self.online_net.sample_noise()
        next_q = self.online_net.calculate_q(states=next_states)
      else:
        next_q = self.target_net.calculate_q(states=next_states)

      # Calculate greedy actions.
      next_actions = torch.argmax(next_q, dim=1, keepdim=True)
      assert next_actions.shape == (self.batch_size, 1)

      # Calculate quantile values of next states and actions at tau_hats.
      next_sa_quantiles = evaluate_quantile_at_action(
          self.target_net(states=next_states), next_actions).transpose(1, 2)
      assert next_sa_quantiles.shape == (self.batch_size, 1, self.N)

      # Calculate target quantile values.
      target_sa_quantiles = rewards[..., None] + (
          1.0 - dones[..., None]) * self.gamma_n * next_sa_quantiles
      assert target_sa_quantiles.shape == (self.batch_size, 1, self.N)

    td_errors = target_sa_quantiles - current_sa_quantiles
    assert td_errors.shape == (self.batch_size, self.N, self.N)

    quantile_huber_loss = calculate_quantile_huber_loss(
        td_errors, self.tau_hats, weights, self.kappa)

    return quantile_huber_loss, next_q.detach().mean().item(), \
        td_errors.detach().abs()
