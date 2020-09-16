# Copyright (c) 2020 fortiss GmbH
#
# Authors: Patrick Hart, Julian Bernhard, Klemens Esterle, and
# Tobias Kessler, Mansoor Nasir
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# The code is adapted from opensource
# implementation - https://github.com/ku2482/fqf-iqn-qrdqn.pytorch
# MIT License -Copyright (c) 2020 Toshiki Watanabe

from copy import copy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def initialize_weights_xavier(m, gain=1.0):
  if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
    torch.nn.init.xavier_uniform_(m.weight, gain=gain)
    if m.bias is not None:
      torch.nn.init.constant_(m.bias, 0)


def initialize_weights_he(m):
  if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
    torch.nn.init.kaiming_uniform_(m.weight)
    if m.bias is not None:
      torch.nn.init.constant_(m.bias, 0)


class Flatten(nn.Module):

  def forward(self, x):
    return x.view(x.size(0), -1)


class DQNBase(nn.Module):

  def __init__(self, num_channels, hidden=512, embedding_dim=512):
    super(DQNBase, self).__init__()

    self._net = nn.Sequential(
        torch.nn.Linear(num_channels, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, embedding_dim),
    ).apply(initialize_weights_he)

    self._embedding_dim = embedding_dim

  def forward(self, states):
    batch_size = states.shape[0]

    # Calculate embeddings of states.
    state_embedding = self._net(states)
    assert state_embedding.shape == (batch_size, self._embedding_dim)

    return state_embedding


class FractionProposalNetwork(nn.Module):

  def __init__(self, N=32, embedding_dim=7 * 7 * 64):
    super(FractionProposalNetwork, self).__init__()

    self._net = nn.Sequential(
        nn.Linear(embedding_dim,
                  N)).apply(lambda x: initialize_weights_xavier(x, gain=0.01))

    self._N = N
    self._embedding_dim = embedding_dim

  def forward(self, state_embeddings):
    batch_size = state_embeddings.shape[0]

    # Calculate (log of) probabilities q_i in the paper.
    log_probs = F.log_softmax(self._net(state_embeddings), dim=1)
    probs = log_probs.exp()
    assert probs.shape == (batch_size, self._N)

    tau_0 = torch.zeros((batch_size, 1),
                        dtype=state_embeddings.dtype,
                        device=state_embeddings.device)
    taus_1_N = torch.cumsum(probs, dim=1)

    # Calculate \tau_i (i=0,...,N).
    taus = torch.cat((tau_0, taus_1_N), dim=1)
    assert taus.shape == (batch_size, self._N + 1)

    # Calculate \hat \tau_i (i=0,...,N-1).
    tau_hats = (taus[:, :-1] + taus[:, 1:]).detach() / 2.
    assert tau_hats.shape == (batch_size, self._N)

    # Calculate entropies of value distributions.
    entropies = -(log_probs * probs).sum(dim=-1, keepdim=True)
    assert entropies.shape == (batch_size, 1)

    return taus, tau_hats, entropies


class CosineEmbeddingNetwork(nn.Module):

  def __init__(self, num_cosines=64, embedding_dim=7 * 7 * 64, noisy_net=False):
    super(CosineEmbeddingNetwork, self).__init__()
    linear = NoisyLinear if noisy_net else nn.Linear

    self._net = nn.Sequential(linear(num_cosines, embedding_dim), nn.ReLU())
    self._num_cosines = num_cosines
    self._embedding_dim = embedding_dim

  def forward(self, taus):
    batch_size = taus.shape[0]
    N = taus.shape[1]

    # Calculate i * \pi (i=1,...,N).
    i_pi = np.pi * torch.arange(start=1,
                                end=self._num_cosines + 1,
                                dtype=taus.dtype,
                                device=taus.device).view(
                                    1, 1, self._num_cosines)

    # Calculate cos(i * \pi * \tau).
    cosines = torch.cos(taus.view(batch_size, N, 1) * i_pi).view(
        batch_size * N, self._num_cosines)

    # Calculate embeddings of taus.
    tau_embeddings = self._net(cosines).view(batch_size, N, self._embedding_dim)

    return tau_embeddings


class QuantileNetwork(nn.Module):

  def __init__(self, num_actions, embedding_dim=7 * 7 * 64, noisy_net=False):
    super(QuantileNetwork, self).__init__()
    linear = NoisyLinear if noisy_net else nn.Linear

    self._net = nn.Sequential(
        linear(embedding_dim, 512),
        nn.ReLU(),
        linear(512, num_actions),
    )
    self._num_actions = num_actions
    self._embedding_dim = embedding_dim
    self._noisy_net = noisy_net

  def forward(self, state_embeddings, tau_embeddings):
    assert state_embeddings.shape[0] == tau_embeddings.shape[0]
    assert state_embeddings.shape[1] == tau_embeddings.shape[2]

    # NOTE: Because variable taus correspond to either \tau or \hat \tau
    # in the paper, N isn't neccesarily the same as fqf.N.
    batch_size = state_embeddings.shape[0]
    N = tau_embeddings.shape[1]

    # Reshape into (batch_size, 1, embedding_dim).
    state_embeddings = state_embeddings.view(batch_size, 1, self._embedding_dim)

    # Calculate embeddings of states and taus.
    embeddings = (state_embeddings * tau_embeddings).view(
        batch_size * N, self._embedding_dim)

    # Calculate quantile values.
    quantiles = self._net(embeddings)

    return quantiles.view(batch_size, N, self._num_actions)


class NoisyLinear(nn.Module):

  def __init__(self, in_features, out_features, sigma=0.5):
    super(NoisyLinear, self).__init__()

    # Learnable parameters.
    self._mu_W = nn.Parameter(torch.FloatTensor(out_features, in_features))
    self._sigma_W = nn.Parameter(torch.FloatTensor(out_features, in_features))
    self._mu_bias = nn.Parameter(torch.FloatTensor(out_features))
    self._sigma_bias = nn.Parameter(torch.FloatTensor(out_features))

    # Factorized noise parameters.
    self.register_buffer('eps_p', torch.FloatTensor(in_features))
    self.register_buffer('eps_q', torch.FloatTensor(out_features))

    self._in_features = in_features
    self._out_features = out_features
    self._sigma = sigma

    self.reset()
    self.sample()

  def reset(self):
    bound = 1 / np.sqrt(self._in_features)
    self._mu_W.data.uniform_(-bound, bound)
    self._mu_bias.data.uniform_(-bound, bound)
    self._sigma_W.data.fill_(self._sigma / np.sqrt(self._in_features))
    self._sigma_bias.data.fill_(self._sigma / np.sqrt(self._out_features))

  def f(self, x):
    return x.normal_().sign().mul(x.abs().sqrt())

  def sample(self):
    self._eps_p.copy_(self.f(self._eps_p))
    self._eps_q.copy_(self.f(self._eps_q))

  def forward(self, x):
    if self._training:
      weight = self._mu_W + self._sigma_W * self._eps_q.ger(self._eps_p)
      bias = self._mu_bias + self._sigma_bias * self._eps_q.clone()
    else:
      weight = self._mu_W
      bias = self._mu_bias

    return F.linear(x, weight, bias)
