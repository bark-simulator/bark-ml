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
from torch.optim import Adam, RMSprop

from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.model import FQF
from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.utils \
 import disable_gradients, update_params, \
 calculate_quantile_huber_loss, evaluate_quantile_at_action
from .base_agent import BaseAgent


class FQFAgent(BaseAgent):
  """FQFAgent that can be used in BARK and BARK-ML."""

  def __init__(self, env, test_env, params):
    super(FQFAgent, self).__init__(env, test_env, params)

    # NOTE: The author said the training of Fraction Proposal Net is
    # unstable and value distribution degenerates into a deterministic
    # one rarely (e.g. 1 out of 20 seeds). So you can use entropy of value
    # distribution as a regularizer to stabilize (but possibly slow down)
    # training.
    self.ent_coef = self._params["ML"]["FQFAgent"]["Ent_coefs", "", 0]
    self.N = self._params["ML"]["FQFAgent"]["N", "", 32]
    self.num_cosines = self._params["ML"]["FQFAgent"]["NumCosines", "", 64]
    self.kappa = self._params["ML"]["FQFAgent"]["Kappa", "", 1.0]

    # Online network.
    self.online_net = FQF(num_channels=env.observation_space.shape[0],
                          num_actions=self.num_actions,
                          N=self.N,
                          num_cosines=self.num_cosines,
                          dueling_net=self.dueling_net,
                          noisy_net=self.noisy_net,
                          params=self._params).to(self.device)
    # Target network.
    self.target_net = FQF(num_channels=env.observation_space.shape[0],
                          num_actions=self.num_actions,
                          N=self.N,
                          num_cosines=self.num_cosines,
                          dueling_net=self.dueling_net,
                          noisy_net=self.noisy_net,
                          target=True,
                          params=self._params).to(self.device)

    # Copy parameters of the learning network to the target network.
    self.update_target()
    # Disable calculations of gradients of the target network.
    disable_gradients(self.target_net)

    self.fraction_optim = RMSprop(
        self.online_net.fraction_net.parameters(),
        lr=self._params["ML"]["FQFAgent"]["FractionalLearningRate", "",
                                          2.5e-9],
        alpha=0.95,
        eps=0.00001)

    self.quantile_optim = Adam(
        list(self.online_net.dqn_net.parameters()) +
        list(self.online_net.cosine_net.parameters()) +
        list(self.online_net.quantile_net.parameters()),
        lr=self._params["ML"]["FQFAgent"]["QuantileLearningRate", "", 5e-5],
        eps=1e-2 / self.batch_size)

  def update_target(self):
    self.target_net.dqn_net.load_state_dict(
        self.online_net.dqn_net.state_dict())
    self.target_net.quantile_net.load_state_dict(
        self.online_net.quantile_net.state_dict())
    self.target_net.cosine_net.load_state_dict(
        self.online_net.cosine_net.state_dict())

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

    # Calculate embeddings of current states.
    state_embeddings = self.online_net.calculate_state_embeddings(states)

    # Calculate fractions of current states and entropies.
    taus, tau_hats, entropies = \
     self.online_net.calculate_fractions(
      state_embeddings=state_embeddings.detach())

    # Calculate quantile values of current states and actions at tau_hats.
    current_sa_quantile_hats = evaluate_quantile_at_action(
        self.online_net.calculate_quantiles(tau_hats,
                                            state_embeddings=state_embeddings),
        actions)
    assert current_sa_quantile_hats.shape == (self.batch_size, self.N, 1)

    # NOTE: Detach state_embeddings not to update convolution layers. Also,
    # detach current_sa_quantile_hats because I calculate gradients of taus
    # explicitly, not by backpropagation.
    fraction_loss = self.calculate_fraction_loss(
        state_embeddings.detach(), current_sa_quantile_hats.detach(), taus,
        actions, weights)

    quantile_loss, mean_q, errors = self.calculate_quantile_loss(
        state_embeddings, tau_hats, current_sa_quantile_hats, actions, rewards,
        next_states, dones, weights)

    entropy_loss = -self.ent_coef * entropies.mean()

    update_params(self.fraction_optim,
                  fraction_loss + entropy_loss,
                  networks=[self.online_net.fraction_net],
                  retain_graph=True,
                  grad_cliping=self.grad_cliping)
    update_params(self.quantile_optim,
                  quantile_loss,
                  networks=[
                      self.online_net.dqn_net, self.online_net.cosine_net,
                      self.online_net.quantile_net
                  ],
                  retain_graph=False,
                  grad_cliping=self.grad_cliping)

    if self.use_per:
      self.memory.update_priority(errors)

    if self.learning_steps % self.summary_log_interval == 0:
      self.writer.add_scalar('loss/fraction_loss',
                             fraction_loss.detach().item(), 4 * self.steps)
      self.writer.add_scalar('loss/quantile_loss',
                             quantile_loss.detach().item(), 4 * self.steps)
      if self.ent_coef > 0.0:
        self.writer.add_scalar('loss/entropy_loss',
                               entropy_loss.detach().item(), 4 * self.steps)

      self.writer.add_scalar('stats/mean_Q', mean_q, 4 * self.steps)
      self.writer.add_scalar('stats/mean_entropy_of_value_distribution',
                             entropies.mean().detach().item(), 4 * self.steps)

  def calculate_fraction_loss(self, state_embeddings, sa_quantile_hats, taus,
                              actions, weights):
    assert not state_embeddings.requires_grad
    assert not sa_quantile_hats.requires_grad

    batch_size = state_embeddings.shape[0]

    with torch.no_grad():
      sa_quantiles = evaluate_quantile_at_action(
          self.online_net.calculate_quantiles(
              taus=taus[:, 1:-1], state_embeddings=state_embeddings), actions)
      assert sa_quantiles.shape == (batch_size, self.N - 1, 1)

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
                            batch_size, self.N - 1)
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

  def calculate_quantile_loss(self, state_embeddings, tau_hats,
                              current_sa_quantile_hats, actions, rewards,
                              next_states, dones, weights):
    assert not tau_hats.requires_grad

    with torch.no_grad():
      # NOTE: Current and target quantiles share the same proposed
      # fractions to reduce computations. (i.e. next_tau_hats = tau_hats)

      # Calculate Q values of next states.
      if self.double_q_learning:
        # Sample the noise of online network to decorrelate between
        # the action selection and the quantile calculation.
        self.online_net.sample_noise()
        next_q = self.online_net.calculate_q(states=next_states)
      else:
        next_state_embeddings = \
         self.target_net.calculate_state_embeddings(next_states)
        next_q = \
         self.target_net.calculate_q(
          state_embeddings=next_state_embeddings)

      # Calculate greedy actions.
      next_actions = torch.argmax(next_q, dim=1, keepdim=True)
      assert next_actions.shape == (self.batch_size, 1)

      # Calculate features of next states.
      if self.double_q_learning:
        next_state_embeddings = \
         self.target_net.calculate_state_embeddings(next_states)

      # Calculate quantile values of next states and actions at tau_hats.
      next_sa_quantile_hats = evaluate_quantile_at_action(
          self.target_net.calculate_quantiles(
              taus=tau_hats, state_embeddings=next_state_embeddings),
          next_actions).transpose(1, 2)
      assert next_sa_quantile_hats.shape == (self.batch_size, 1, self.N)

      # Calculate target quantile values.
      target_sa_quantile_hats = rewards[..., None] + (
          1.0 - dones[..., None]) * self.gamma_n * next_sa_quantile_hats
      assert target_sa_quantile_hats.shape == (self.batch_size, 1, self.N)

    td_errors = target_sa_quantile_hats - current_sa_quantile_hats
    assert td_errors.shape == (self.batch_size, self.N, self.N)

    quantile_huber_loss = calculate_quantile_huber_loss(
        td_errors, tau_hats, weights, self.kappa)

    return quantile_huber_loss, next_q.detach().mean().item(), \
        td_errors.detach().abs()
