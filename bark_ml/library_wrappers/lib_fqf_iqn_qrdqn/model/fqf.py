# Copyright (c) 2020 fortiss GmbH
#
# Authors: Patrick Hart, Julian Bernhard, Klemens Esterle, and
# Tobias Kessler, Mansoor Nasir
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# The code is adapted from opensource implementation - https://github.com/ku2482/fqf-iqn-qrdqn.pytorch
# MIT License -Copyright (c) 2020 Toshiki Watanabe

from .base_model import BaseModel
from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.network import DQNBase, CosineEmbeddingNetwork, \
 FractionProposalNetwork, QuantileNetwork


class FQF(BaseModel):
  """FQF Model."""

  def __init__(self,
               num_channels,
               num_actions,
               params,
               N=64,
               num_cosines=32,
               dueling_net=False,
               noisy_net=False,
               target=False):
    super(FQF, self).__init__()

    self.N = N
    self.num_channels = num_channels
    self.num_actions = num_actions
    self.num_cosines = num_cosines
    self.dueling_net = dueling_net
    self.noisy_net = noisy_net
    self.target = params["ML"]["FQFModel"]["Target", "", False]
    self.embedding_dim = params["ML"]["FQFModel"]["EmbeddingDims", "", 512]

    # Feature extractor of DQN.
    self.dqn_net = DQNBase(num_channels=num_channels,
                           embedding_dim=self.embedding_dim,
                           hidden=params["ML"]["FQFModel"]["HiddenDims", "",
                                                           512])
    # Cosine embedding network.
    self.cosine_net = CosineEmbeddingNetwork(num_cosines=num_cosines,
                                             embedding_dim=self.embedding_dim,
                                             noisy_net=noisy_net)
    # Quantile network.
    self.quantile_net = QuantileNetwork(num_actions=num_actions,
                                        dueling_net=dueling_net,
                                        noisy_net=noisy_net,
                                        embedding_dim=self.embedding_dim)

    # Fraction proposal network.
    if not target:
      self.fraction_net = FractionProposalNetwork(
          N=N, embedding_dim=self.embedding_dim)

  def calculate_state_embeddings(self, states):
    return self.dqn_net(states)

  def calculate_fractions(self, state_embeddings):
    assert state_embeddings is not None
    assert not self.target or self.fraction_net is not None

    fraction_net = self.fraction_net  # fraction_net if self.target else self.fraction_net
    taus, tau_hats, entropies = fraction_net(state_embeddings)

    return taus, tau_hats, entropies

  def calculate_quantiles(self, taus, states=None, state_embeddings=None):
    assert states is not None or state_embeddings is not None

    if state_embeddings is None:
      state_embeddings = self.dqn_net(states)

    tau_embeddings = self.cosine_net(taus)
    return self.quantile_net(state_embeddings, tau_embeddings)

  def calculate_q(self,
                  taus=None,
                  tau_hats=None,
                  states=None,
                  state_embeddings=None):
    assert states is not None or state_embeddings is not None
    assert not self.target or fraction_net is not None

    if state_embeddings is None:
      state_embeddings = self.dqn_net(states)

    batch_size = state_embeddings.shape[0]

    # Calculate fractions.
    if taus is None or tau_hats is None:
      taus, tau_hats, _ = self.calculate_fractions(
          state_embeddings=state_embeddings)

    # Calculate quantiles.
    quantile_hats = self.calculate_quantiles(tau_hats,
                                             state_embeddings=state_embeddings)
    assert quantile_hats.shape == (batch_size, self.N, self.num_actions)

    # Calculate expectations of value distribution.
    q = ((taus[:, 1:, None] - taus[:, :-1, None]) * quantile_hats) \
     .sum(dim=1)
    assert q.shape == (batch_size, self.num_actions)

    return q

  # Only used by torch script from c++
  # similar to above method calculate_q
  # but for inference only
  def forward(self, states):
    state_embeddings = self.dqn_net(states)
    batch_size = state_embeddings.shape[0]

    # Calculate fractions.
    taus, tau_hats, _ = self.calculate_fractions(
        state_embeddings=state_embeddings)

    # Calculate quantiles.
    tau_embeddings = self.cosine_net(tau_hats)
    quantile_hats = self.quantile_net(state_embeddings, tau_embeddings)
    assert quantile_hats.shape == (batch_size, self.N, self.num_actions)

    # Calculate expectations of value distribution.
    q = ((taus[:, 1:, None] - taus[:, :-1, None]) * quantile_hats) \
     .sum(dim=1)
    assert q.shape == (batch_size, self.num_actions)

    return q
