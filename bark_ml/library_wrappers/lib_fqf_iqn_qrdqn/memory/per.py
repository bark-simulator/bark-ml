# The code is adapted from opensource implementation - https://github.com/ku2482/fqf-iqn-qrdqn.pytorch
# MIT License -Copyright (c) 2020 Toshiki Watanabe

import numpy as np
import torch

from .base import LazyMultiStepMemory
from .segment_tree import SumTree, MinTree


class LazyPrioritizedMultiStepMemory(LazyMultiStepMemory):

  def __init__(self,
               capacity,
               state_shape,
               device,
               gamma=0.99,
               multi_step=3,
               alpha=0.5,
               beta=0.4,
               beta_steps=2e5,
               min_pa=0.0,
               max_pa=1.0,
               eps=0.01):
    super().__init__(capacity, state_shape, device, gamma, multi_step)

    self._alpha = alpha
    self._beta = beta
    self._beta_diff = (1.0 - beta) / beta_steps
    self._min_pa = min_pa
    self._max_pa = max_pa
    self._eps = eps
    self._cached = None

    it_capacity = 1
    while it_capacity < capacity:
      it_capacity *= 2
    self._it_sum = SumTree(it_capacity)
    self._it_min = MinTree(it_capacity)

  def _pa(self, p):
    return np.clip((p + self._eps)**self._alpha, self._min_pa, self._max_pa)

  def append(self, state, action, reward, next_state, done, p=None):
    # Calculate priority.
    if p is None:
      pa = self._max_pa
    else:
      pa = self._pa(p)

    if self._multi_step != 1:
      self._buff.append(state, action, reward)

      if self._buff.is_full():
        state, action, reward = self._buff.get(self._gamma)
        self._append(state, action, reward, next_state, done, pa)

      if done:
        while not self._buff.is_empty():
          state, action, reward = self._buff.get(self._gamma)
          self._append(state, action, reward, next_state, done, pa)
    else:
      self._append(state, action, reward, next_state, done, pa)

  def _append(self, state, action, reward, next_state, done, pa):
    # Store priority, which is done efficiently by SegmentTree.
    self._it_min[self._p] = pa
    self._it_sum[self._p] = pa
    super()._append(state, action, reward, next_state, done)

  def _sample_idxes(self, batch_size):
    total_pa = self._it_sum.sum(0, self._n)
    rands = np.random.rand(batch_size) * total_pa
    indices = [self._it_sum.find_prefixsum_idx(r) for r in rands]
    self._beta = min(1., self._beta + self._beta_diff)
    return indices

  def sample(self, batch_size):
    assert self._cached is None, 'Update priorities before sampling.'

    self._cached = self._sample_idxes(batch_size)
    batch = self._sample(self._cached, batch_size)
    weights = self._calc_weights(self._cached)
    return batch, weights

  def _calc_weights(self, indices):
    min_pa = self._it_min.min()
    weights = [(self._it_sum[i] / min_pa)**-self._beta for i in indices]
    return torch.FloatTensor(weights).to(self._device).view(-1, 1)

  def update_priority(self, errors):
    assert self._cached is not None

    ps = errors.detach().cpu().abs().numpy().flatten()
    pas = self._pa(ps)

    for index, pa in zip(self._cached, pas):
      assert 0 <= index < self._n
      assert 0 < pa
      self._it_sum[index] = pa
      self._it_min[index] = pa

    self._cached = None
