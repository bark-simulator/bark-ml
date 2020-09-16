# Copyright (c) 2020 fortiss GmbH
#
# Authors: Patrick Hart, Julian Bernhard, Klemens Esterle,
# Tobias Kessler and Mansoor Nasir
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# The code is adapted from opensource
# implementation - https://github.com/ku2482/fqf-iqn-qrdqn.pytorch
# MIT License -Copyright (c) 2020 Toshiki Watanabe

import torch
from torch.optim import Adam, RMSprop

from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.model import FQF
from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.utils \
 import disable_gradients, update_params, \
 calculate_quantile_huber_loss, evaluate_quantile_at_action
from .base_agent import BaseAgent


class FQFAgent(BaseAgent):

  def __init__(self, env, test_env, params):
    super(FQFAgent, self).__init__(env, test_env, params)

    # NOTE: The author said the training of Fraction Proposal Net is
    # unstable and value distribution degenerates into a deterministic
    # one rarely (e.g. 1 out of 20 seeds). So you can use entropy of value
    # distribution as a regularizer to stabilize (but possibly slow down)
    # training.
    self._ent_coef = self._params["ML"]["FQFAgent"]["Ent_coefs", "", 0]
    self._N = self._params["ML"]["FQFAgent"]["N", "", 32]
    self._num_cosines = self._params["ML"]["FQFAgent"]["NumCosines", "", 64]
    self._kappa = self._params["ML"]["FQFAgent"]["Kappa", "", 1.0]

    # Online network.
    self._online_net = FQF(num_channels=env.observation_space.shape[0],
                           num_actions=self._num_actions,
                           N=self._N,
                           num_cosines=self._num_cosines,
                           noisy_net=self._noisy_net,
                           params=self._params).to(self._device)
    # Target network.
    self._target_net = FQF(num_channels=env.observation_space.shape[0],
                           num_actions=self._num_actions,
                           N=self._N,
                           num_cosines=self._num_cosines,
                           noisy_net=self._noisy_net,
                           target=True,
                           params=self._params).to(self._device)

    # Copy parameters of the learning network to the target network.
    self.update_target()
    # Disable calculations of gradients of the target network.
    disable_gradients(self._target_net)

    self._fraction_optim = RMSprop(
        self._online_net.fraction_net.parameters(),
        lr=self._params["ML"]["FQFAgent"]["FractionalLearningRate", "", 2.5e-9],
        alpha=0.95,
        eps=0.00001)

    self._quantile_optim = Adam(
        list(self._online_net.dqn_net.parameters()) +
        list(self._online_net.cosine_net.parameters()) +
        list(self._online_net.quantile_net.parameters()),
        lr=self._params["ML"]["FQFAgent"]["QuantileLearningRate", "", 5e-5],
        eps=1e-2 / self._batch_size)

  def update_target(self):
    self._target_net.dqn_net.load_state_dict(
        self._online_net.dqn_net.state_dict())
    self._target_net.quantile_net.load_state_dict(
        self._online_net.quantile_net.state_dict())
    self._target_net.cosine_net.load_state_dict(
        self._online_net.cosine_net.state_dict())

  def learn(self):
    self._learning_steps += 1
    self._online_net.sample_noise()
    self._target_net.sample_noise()

    if self._use_per:
      (states, actions, rewards, next_states, dones), weights = \
       self._memory.sample(self._batch_size)
    else:
      states, actions, rewards, next_states, dones = \
       self._memory.sample(self._batch_size)
      weights = None

    # Calculate embeddings of current states.
    state_embeddings = self._online_net.calculate_state_embeddings(states)

    # Calculate fractions of current states and entropies.
    taus, tau_hats, entropies = \
     self._online_net.calculate_fractions(
      state_embeddings=state_embeddings.detach())

    # Calculate quantile values of current states and actions at tau_hats.
    current_sa_quantile_hats = evaluate_quantile_at_action(
        self._online_net.calculate_quantiles(tau_hats,
                                             state_embeddings=state_embeddings),
        actions)
    assert current_sa_quantile_hats.shape == (self._batch_size, self._N, 1)

    # NOTE: Detach state_embeddings not to update convolution layers. Also,
    # detach current_sa_quantile_hats because I calculate gradients of taus
    # explicitly, not by backpropagation.
    fraction_loss = self._calculate_fraction_loss(
        state_embeddings.detach(), current_sa_quantile_hats.detach(), taus,
        actions, weights)

    quantile_loss, mean_q, errors = self._calculate_quantile_loss(
        tau_hats, current_sa_quantile_hats, rewards, next_states, dones,
        weights)

    entropy_loss = -self._ent_coef * entropies.mean()

    update_params(self._fraction_optim,
                  fraction_loss + entropy_loss,
                  networks=[self._online_net.fraction_net],
                  retain_graph=True,
                  grad_cliping=self._grad_cliping)
    update_params(self._quantile_optim,
                  quantile_loss,
                  networks=[
                      self._online_net.dqn_net, self._online_net.cosine_net,
                      self._online_net.quantile_net
                  ],
                  retain_graph=False,
                  grad_cliping=self._grad_cliping)

    if self._use_per:
      self._memory.update_priority(errors)

    if self._learning_steps % self._summary_log_interval == 0:
      self._writer.add_scalar('loss/fraction_loss',
                              fraction_loss.detach().item(), 4 * self._steps)
      self._writer.add_scalar('loss/quantile_loss',
                              quantile_loss.detach().item(), 4 * self._steps)
      if self._ent_coef > 0.0:
        self._writer.add_scalar('loss/entropy_loss',
                                entropy_loss.detach().item(), 4 * self._steps)

      self._writer.add_scalar('stats/mean_Q', mean_q, 4 * self._steps)
      self._writer.add_scalar('stats/mean_entropy_of_value_distribution',
                              entropies.mean().detach().item(), 4 * self._steps)

  def calculate_fraction_loss(self, state_embeddings, sa_quantile_hats, taus,
                              actions, weights):
    assert not state_embeddings.requires_grad
    assert not sa_quantile_hats.requires_grad

    batch_size = state_embeddings.shape[0]

    with torch.no_grad():
      sa_quantiles = evaluate_quantile_at_action(
          self._online_net.calculate_quantiles(
              taus=taus[:, 1:-1], state_embeddings=state_embeddings), actions)
      assert sa_quantiles.shape == (batch_size, self._N - 1, 1)

    # NOTE: Proposition 1 in the paper requires F^{-1} is non-decreasing.
    # I relax this requirements and calculate gradients of taus even when
    # F^{-1} is not non-decreasing.

    values_1 = sa_quantiles - sa_quantile_hats[:, :-1]
    signs_1 = sa_quantiles > torch.cat(
        [sa_quantile_hats[:, :1], sa_quantiles[:, :-1]], dim=1)
    assert values_1.shape == signs_1.shape

    values_2 = sa_quantiles - sa_quantile_hats[:, 1:]
    signs_2 = sa_quantiles < torch.cat(
        [sa_quantiles[:, 1:], sa_quantile_hats[:, -1:]], dim=1)
    assert values_2.shape == signs_2.shape

    gradient_of_taus = (torch.where(signs_1, values_1, -values_1) +
                        torch.where(signs_2, values_2, -values_2)).view(
                            batch_size, self._N - 1)
    assert not gradient_of_taus.requires_grad
    assert gradient_of_taus.shape == taus[:, 1:-1].shape

    # Gradients of the network parameters and corresponding loss
    # are calculated using chain rule.
    if weights is not None:
      fraction_loss = ((
          (gradient_of_taus * taus[:, 1:-1]).sum(dim=1, keepdim=True)) *
                       weights).mean()
    else:
      fraction_loss = \
       (gradient_of_taus * taus[:, 1:-1]).sum(dim=1).mean()

    return fraction_loss

  def calculate_quantile_loss(self, tau_hats, current_sa_quantile_hats, rewards,
                              next_states, dones, weights):
    assert not tau_hats.requires_grad

    with torch.no_grad():
      # NOTE: Current and target quantiles share the same proposed
      # fractions to reduce computations. (i.e. next_tau_hats = tau_hats)

      # Calculate Q values of next states.
      if self._double_q_learning:
        # Sample the noise of online network to decorrelate between
        # the action selection and the quantile calculation.
        self._online_net.sample_noise()
        next_q = self._online_net.calculate_q(states=next_states)
      else:
        next_state_embeddings = \
         self._target_net.calculate_state_embeddings(next_states)
        next_q = \
         self._target_net.calculate_q(
          state_embeddings=next_state_embeddings)

      # Calculate greedy actions.
      next_actions = torch.argmax(next_q, dim=1, keepdim=True)
      assert next_actions.shape == (self._batch_size, 1)

      # Calculate features of next states.
      if self._double_q_learning:
        next_state_embeddings = \
         self._target_net.calculate_state_embeddings(next_states)

      # Calculate quantile values of next states and actions at tau_hats.
      next_sa_quantile_hats = evaluate_quantile_at_action(
          self._target_net.calculate_quantiles(
              taus=tau_hats, state_embeddings=next_state_embeddings),
          next_actions).transpose(1, 2)
      assert next_sa_quantile_hats.shape == (self._batch_size, 1, self._N)

      # Calculate target quantile values.
      target_sa_quantile_hats = rewards[..., None] + (
          1.0 - dones[..., None]) * self._gamma_n * next_sa_quantile_hats
      assert target_sa_quantile_hats.shape == (self._batch_size, 1, self._N)

    td_errors = target_sa_quantile_hats - current_sa_quantile_hats
    assert td_errors.shape == (self._batch_size, self._N, self._N)

    quantile_huber_loss = calculate_quantile_huber_loss(td_errors, tau_hats,
                                                        weights, self._kappa)

    return quantile_huber_loss, next_q.detach().mean().item(), \
        td_errors.detach().abs()
