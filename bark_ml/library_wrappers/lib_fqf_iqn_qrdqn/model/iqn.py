# Copyright (c) 2020 fortiss GmbH
#
# Authors: Patrick Hart, Julian Bernhard, Klemens Esterle,
# Tobias Kessler and Mansoor Nasir
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# The code is adapted from opensource implementation
# - https://github.com/ku2482/fqf-iqn-qrdqn.pytorch
# MIT License -Copyright (c) 2020 Toshiki Watanabe

import torch

from .base_model import BaseModel
from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.network import DQNBase, CosineEmbeddingNetwork, \
 QuantileNetwork


class IQN(BaseModel):

  def __init__(self, num_channels, num_actions, params, num_cosines, noisy_net):
    super(IQN, self).__init__()

    self._num_channels = num_channels
    self._num_actions = num_actions
    self._num_cosines = num_cosines
    self._noisy_net = noisy_net
    self._K = params["ML"]["IQNModel"]["K", "", 32]
    self._embedding_dim = params["ML"]["IQNModel"]["EmbeddingDims", "", 512]

    # Feature extractor of DQN.
    self._dqn_net = DQNBase(num_channels=num_channels,
                            embedding_dim=self._embedding_dim,
                            hidden=params["ML"]["IQNModel"]["HiddenDims", "",
                                                            512])
    # Cosine embedding network.
    self._cosine_net = CosineEmbeddingNetwork(num_cosines=num_cosines,
                                              embedding_dim=self._embedding_dim,
                                              noisy_net=noisy_net)
    # Quantile network.
    self._quantile_net = QuantileNetwork(num_actions=num_actions,
                                         embedding_dim=self._embedding_dim,
                                         noisy_net=noisy_net)

  @property
  def dqn_net(self):
    return self._dqn_net

  @property
  def cosine_net(self):
    return self._cosine_net

  @property
  def quantile_net(self):
    return self._quantile_net

  def calculate_state_embeddings(self, states):
    return self._dqn_net(states)

  def calculate_quantiles(self, taus, states=None, state_embeddings=None):
    assert states is not None or state_embeddings is not None

    if state_embeddings is None:
      state_embeddings = self._dqn_net(states)

    tau_embeddings = self._cosine_net(taus)
    return self._quantile_net(state_embeddings, tau_embeddings)

  def calculate_q(self, states=None, state_embeddings=None):
    assert states is not None or state_embeddings is not None
    batch_size = states.shape[0] if states is not None \
     else state_embeddings.shape[0]

    if state_embeddings is None:
      state_embeddings = self._dqn_net(states)

    # Sample fractions.
    taus = torch.rand(batch_size,
                      self._K,
                      dtype=state_embeddings.dtype,
                      device=state_embeddings.device)

    # Calculate quantiles.
    quantiles = self._calculate_quantiles(taus,
                                          state_embeddings=state_embeddings)
    assert quantiles.shape == (batch_size, self._K, self._num_actions)

    # Calculate expectations of value distributions.
    q = quantiles.mean(dim=1)
    assert q.shape == (batch_size, self._num_actions)

    return q

  def forward(self, states):
    assert states is not None
    batch_size = states.shape[0]

    state_embeddings = self._dqn_net(states)

    # Sample fractions. # note. causing bug in torch script.
    taus = torch.ones(batch_size,
                      self._K,
                      dtype=state_embeddings.dtype,
                      device=state_embeddings.device)

    # Calculate quantiles.
    tau_embeddings = self._cosine_net(taus)
    quantiles = self._quantile_net(state_embeddings, tau_embeddings)
    assert quantiles.shape == (batch_size, self._K, self._num_actions)

    # Calculate expectations of value distributions.
    q = quantiles.mean(dim=1)
    assert q.shape == (batch_size, self._num_actions)

    return q
