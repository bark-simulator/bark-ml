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

from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.model import IQN
from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.utils \
 import disable_gradients, update_params, \
 calculate_quantile_huber_loss, evaluate_quantile_at_action
from .base_agent import BaseAgent


class IQNAgent(BaseAgent):
  """IQNAgent that can be used in BARK and BARK-ML."""

  def __init__(self, env, test_env, params):
    super(IQNAgent, self).__init__(env, test_env, params)

    self.N = self._params["ML"]["IQNAgent"]["N", "", 64]
    self.N_dash = self._params["ML"]["IQNAgent"]["N_dash", "", 64]
    self.K = self._params["ML"]["IQNAgent"]["N_dash", "", 32]
    self.num_cosines = self._params["ML"]["IQNAgent"]["NumCosines", "", 64]
    self.kappa = self._params["ML"]["IQNAgent"]["Kappa", "", 1.0]

    # Online network.
    self.online_net = IQN(num_channels=env.observation_space.shape[0],
                          num_actions=self.num_actions,
                          num_cosines=self.num_cosines,
                          dueling_net=self.dueling_net,
                          noisy_net=self.noisy_net,
                          params=self._params).to(self.device)
    # Target network.
    self.target_net = IQN(num_channels=env.observation_space.shape[0],
                          num_actions=self.num_actions,
                          num_cosines=self.num_cosines,
                          dueling_net=self.dueling_net,
                          noisy_net=self.noisy_net,
                          params=self._params).to(self.device)

    # Copy parameters of the learning network to the target network.
    self.update_target()
    # Disable calculations of gradients of the target network.
    disable_gradients(self.target_net)

    self.optim = Adam(self.online_net.parameters(),
                      lr=self._params["ML"]["IQNAgent"]["LearningRate", "",
                                                        5e-5],
                      eps=1e-2 / self.batch_size)

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

    # Calculate features of states.
    state_embeddings = self.online_net.calculate_state_embeddings(states)

    quantile_loss, mean_q, errors = self.calculate_loss(
        state_embeddings, actions, rewards, next_states, dones, weights)

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

  def calculate_loss(self, state_embeddings, actions, rewards, next_states,
                     dones, weights):
    # Sample fractions.
    taus = torch.rand(self.batch_size,
                      self.N,
                      dtype=state_embeddings.dtype,
                      device=state_embeddings.device)

    # Calculate quantile values of current states and actions at tau_hats.
    current_sa_quantiles = evaluate_quantile_at_action(
        self.online_net.calculate_quantiles(taus,
                                            state_embeddings=state_embeddings),
        actions)
    assert current_sa_quantiles.shape == (self.batch_size, self.N, 1)

    with torch.no_grad():
      # Calculate Q values of next states.
      if self.double_q_learning:  # note: double q learning set to always false.
        # Sample the noise of online network to decorrelate between
        # the action selection and the quantile calculation.
        self.online_net.sample_noise()
        next_q = self.online_net.calculate_q(states=next_states)
      else:
        next_state_embeddings = \
         self.target_net.calculate_state_embeddings(next_states)
        next_q = self.target_net.calculate_q(
            state_embeddings=next_state_embeddings)

      # Calculate greedy actions.
      next_actions = torch.argmax(next_q, dim=1, keepdim=True)
      assert next_actions.shape == (self.batch_size, 1)

      # Calculate features of next states.
      if self.double_q_learning:
        next_state_embeddings = \
         self.target_net.calculate_state_embeddings(next_states)

      # Sample next fractions.
      tau_dashes = torch.rand(self.batch_size,
                              self.N_dash,
                              dtype=state_embeddings.dtype,
                              device=state_embeddings.device)

      # Calculate quantile values of next states and next actions.
      next_sa_quantiles = evaluate_quantile_at_action(
          self.target_net.calculate_quantiles(
              tau_dashes, state_embeddings=next_state_embeddings),
          next_actions).transpose(1, 2)
      assert next_sa_quantiles.shape == (self.batch_size, 1, self.N_dash)

      # Calculate target quantile values.
      target_sa_quantiles = rewards[..., None] + (
          1.0 - dones[..., None]) * self.gamma_n * next_sa_quantiles
      assert target_sa_quantiles.shape == (self.batch_size, 1, self.N_dash)

    td_errors = target_sa_quantiles - current_sa_quantiles
    assert td_errors.shape == (self.batch_size, self.N, self.N_dash)

    quantile_huber_loss = calculate_quantile_huber_loss(
        td_errors, taus, weights, self.kappa)

    return quantile_huber_loss, next_q.detach().mean().item(), \
        td_errors.detach().abs()
