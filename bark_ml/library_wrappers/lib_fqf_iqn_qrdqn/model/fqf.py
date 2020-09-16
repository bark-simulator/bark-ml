# Copyright (c) 2020 fortiss GmbH
#
# Authors: Patrick Hart, Julian Bernhard, Klemens Esterle,
# Tobias Kessler and Mansoor Nasir
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# The code is adapted from opensource implementation -
# https://github.com/ku2482/fqf-iqn-qrdqn.pytorch
# MIT License -Copyright (c) 2020 Toshiki Watanabe

from .base_model import BaseModel
from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.network import \
  DQNBase, CosineEmbeddingNetwork, FractionProposalNetwork, QuantileNetwork


class FQF(BaseModel):

  def __init__(self,
               num_channels,
               num_actions,
               params,
               N=64,
               num_cosines=32,
               noisy_net=False,
               target=False):
    super(FQF, self).__init__()

    self._N = N
    self._num_channels = num_channels
    self._num_actions = num_actions
    self._num_cosines = num_cosines
    self._noisy_net = noisy_net
    self._target = params["ML"]["FQFModel"]["Target", "", False]
    self._embedding_dim = params["ML"]["FQFModel"]["EmbeddingDims", "", 512]

    # Feature extractor of DQN.
    self._dqn_net = DQNBase(num_channels=num_channels,
                            embedding_dim=self._embedding_dim,
                            hidden=params["ML"]["FQFModel"]["HiddenDims", "",
                                                            512])
    # Cosine embedding network.
    self._cosine_net = CosineEmbeddingNetwork(num_cosines=num_cosines,
                                              embedding_dim=self._embedding_dim,
                                              noisy_net=noisy_net)
    # Quantile network.
    self._quantile_net = QuantileNetwork(num_actions=num_actions,
                                         noisy_net=noisy_net,
                                         embedding_dim=self._embedding_dim)

    # Fraction proposal network.
    if not target:
      self._fraction_net = FractionProposalNetwork(
          N=N, embedding_dim=self._embedding_dim)

  @property
  def dqn_net(self):
    return self._dqn_net

  @property
  def cosine_net(self):
    return self._cosine_net

  @property
  def quantile_net(self):
    return self._quantile_net

  @property
  def fraction_net(self):
    return self._fraction_net

  def calculate_state_embeddings(self, states):
    return self._dqn_net(states)

  def calculate_fractions(self, state_embeddings):
    assert state_embeddings is not None
    assert not self._target or self._fraction_net is not None

    fraction_net = self._fraction_net  # fraction_net if self._target else self._fraction_net
    taus, tau_hats, entropies = fraction_net(state_embeddings)

    return taus, tau_hats, entropies

  def calculate_quantiles(self, taus, states=None, state_embeddings=None):
    assert states is not None or state_embeddings is not None

    if state_embeddings is None:
      state_embeddings = self._dqn_net(states)

    tau_embeddings = self._cosine_net(taus)
    return self._quantile_net(state_embeddings, tau_embeddings)

  def calculate_q(self,
                  taus=None,
                  tau_hats=None,
                  states=None,
                  state_embeddings=None):
    assert states is not None or state_embeddings is not None
    assert not self._target or self.fraction_net is not None

    if state_embeddings is None:
      state_embeddings = self._dqn_net(states)

    batch_size = state_embeddings.shape[0]

    # Calculate fractions.
    if taus is None or tau_hats is None:
      taus, tau_hats, _ = self.calculate_fractions(
          state_embeddings=state_embeddings)

    # Calculate quantiles.
    quantile_hats = self._calculate_quantiles(tau_hats,
                                              state_embeddings=state_embeddings)
    assert quantile_hats.shape == (batch_size, self._N, self._num_actions)

    # Calculate expectations of value distribution.
    q = ((taus[:, 1:, None] - taus[:, :-1, None]) * quantile_hats) \
     .sum(dim=1)
    assert q.shape == (batch_size, self._num_actions)

    return q

  # Only used by torch script from c++
  # similar to above method calculate_q
  # but for inference only
  def forward(self, states):
    state_embeddings = self._dqn_net(states)
    batch_size = state_embeddings.shape[0]

    # Calculate fractions.
    taus, tau_hats, _ = self.calculate_fractions(
        state_embeddings=state_embeddings)

    # Calculate quantiles.
    tau_embeddings = self._cosine_net(tau_hats)
    quantile_hats = self._quantile_net(state_embeddings, tau_embeddings)
    assert quantile_hats.shape == (batch_size, self._N, self._num_actions)

    # Calculate expectations of value distribution.
    q = ((taus[:, 1:, None] - taus[:, :-1, None]) * quantile_hats) \
     .sum(dim=1)
    assert q.shape == (batch_size, self._num_actions)

    return q
