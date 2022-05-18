# The code is adapted from opensource implementation - https://github.com/ku2482/fqf-iqn-qrdqn.pytorch
# MIT License -Copyright (c) 2020 Toshiki Watanabe
from torch import nn

from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.network import NoisyLinear


class BaseModel(nn.Module):
  """Base model."""

  def __init__(self):
    super().__init__()

  def sample_noise(self):
    if self.noisy_net:
      for m in self.modules():
        if isinstance(m, NoisyLinear):
          m.sample()
