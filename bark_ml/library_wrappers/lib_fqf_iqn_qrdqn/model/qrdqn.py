# Copyright (c) 2020 fortiss GmbH
#
# Authors: Patrick Hart, Julian Bernhard, Klemens Esterle,
# Tobias Kessler and Mansoor Nasir
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# The code is adapted from opensource implementation
#  - https://github.com/ku2482/fqf-iqn-qrdqn.pytorch
# MIT License -Copyright (c) 2020 Toshiki Watanabe

from torch import nn

from .base_model import BaseModel
from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.network import DQNBase, NoisyLinear


class QRDQN(BaseModel):

  def __init__(self, num_channels, num_actions, N, params, noisy_net=False):
    super(QRDQN, self).__init__()

    self._N = N
    self._num_channels = num_channels
    self._num_actions = num_actions
    self._noisy_net = noisy_net
    self._embedding_dim = params["ML"]["QRDQN"]["EmbeddingDims", "", 512]

    linear = NoisyLinear if noisy_net else nn.Linear

    # Feature extractor of DQN.
    self._dqn_net = DQNBase(num_channels=num_channels,
                            embedding_dim=self._embedding_dim,
                            hidden=params["ML"]["QRDQN"]["HiddenDims", "", 512])
    # Quantile network.
    self._q_net = nn.Sequential(
        linear(self._embedding_dim, 512),
        nn.ReLU(),
        linear(512, num_actions * N),
    )

  @property
  def dqn_net(self):
    return self._dqn_net

  @property
  def q_net(self):
    return self._q_net

  def calculate_quantiles(self, states=None, state_embeddings=None):
    assert states is not None or state_embeddings is not None
    batch_size = states.shape[0] if states is not None \
     else state_embeddings.shape[0]

    if state_embeddings is None:
      state_embeddings = self._dqn_net(states)

    quantiles = self._q_net(state_embeddings).view(batch_size, self._N,
                                                   self._num_actions)
    assert quantiles.shape == (batch_size, self._N, self._num_actions)

    return quantiles

  def calculate_q(self, states=None, state_embeddings=None):
    assert states is not None or state_embeddings is not None
    batch_size = states.shape[0] if states is not None \
     else state_embeddings.shape[0]

    # Calculate quantiles.
    quantiles = self._calculate_quantiles(states=states,
                                          state_embeddings=state_embeddings)

    # Calculate expectations of value distributions.
    q = quantiles.mean(dim=1)
    assert q.shape == (batch_size, self._num_actions)

    return q

  def forward(self, states):
    assert states is not None
    batch_size = states.shape[0]

    state_embeddings = self._dqn_net(states)

    # Calculate quantiles.
    quantiles = self._q_net(state_embeddings).view(batch_size, self._N,
                                                   self._num_actions)

    # Calculate expectations of value distributions.
    q = quantiles.mean(dim=1)
    assert q.shape == (batch_size, self._num_actions)

    return q
