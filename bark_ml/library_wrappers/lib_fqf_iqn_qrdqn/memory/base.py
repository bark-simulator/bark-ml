# The code is adapted from opensource implementation - https://github.com/ku2482/fqf-iqn-qrdqn.pytorch
# MIT License -Copyright (c) 2020 Toshiki Watanabe

from collections import deque
import numpy as np
import torch


class MultiStepBuff:

  def __init__(self, maxlen=3):
    super(MultiStepBuff, self).__init__()
    self._maxlen = int(maxlen)
    self.reset()

  def append(self, state, action, reward):
    self._states.append(state)
    self._actions.append(action)
    self._rewards.append(reward)

  def get(self, gamma=0.99):
    assert len(self._rewards) > 0
    state = self._states.popleft()
    action = self._actions.popleft()
    reward = self._nstep_return(gamma)
    return state, action, reward

  def _nstep_return(self, gamma):
    r = np.sum([r * (gamma**i) for i, r in enumerate(self._rewards)])
    self._rewards.popleft()
    return r

  def reset(self):
    # Buffer to store n-step transitions.
    self._states = deque(maxlen=self._maxlen)
    self._actions = deque(maxlen=self._maxlen)
    self._rewards = deque(maxlen=self._maxlen)

  def is_empty(self):
    return len(self._rewards) == 0

  def is_full(self):
    return len(self._rewards) == self._maxlen

  def __len__(self):
    return len(self._rewards)


class LazyMemory(dict):
  _state_keys = ['state', 'next_state']
  _np_keys = ['action', 'reward', 'done']
  _keys = _state_keys + _np_keys

  def __init__(self, capacity, state_shape, device):
    super(LazyMemory, self).__init__()
    self._capacity = int(capacity)
    self._state_shape = state_shape
    self._device = device
    self.reset()

  def reset(self):
    self['state'] = []
    self['next_state'] = []

    self['action'] = np.empty((self._capacity, 1), dtype=np.int64)
    self['reward'] = np.empty((self._capacity, 1), dtype=np.float32)
    self['done'] = np.empty((self._capacity, 1), dtype=np.float32)

    self._n = 0
    self._p = 0

  def append(self, state, action, reward, next_state, done):
    self._append(state, action, reward, next_state, done)

  def _append(self, state, action, reward, next_state, done):
    self['state'].append(state)
    self['next_state'].append(next_state)
    self['action'][self._p] = action
    self['reward'][self._p] = reward
    self['done'][self._p] = done

    self._n = min(self._n + 1, self._capacity)
    self._p = (self._p + 1) % self._capacity

    self.truncate()

  def truncate(self):
    while len(self) > self._capacity:
      del self['state'][0]
      del self['next_state'][0]

  def sample(self, batch_size):
    indices = np.random.randint(low=0, high=len(self), size=batch_size)
    return self._sample(indices, batch_size)

  def _sample(self, indices, batch_size):
    bias = -self._p if self._n == self._capacity else 0

    states = np.empty((batch_size, *self._state_shape), dtype=np.uint8)
    next_states = np.empty((batch_size, *self._state_shape), dtype=np.uint8)

    for i, index in enumerate(indices):
      _index = np.mod(index + bias, self._capacity)
      states[i, ...] = self['state'][_index]
      next_states[i, ...] = self['next_state'][_index]

    states = torch.ByteTensor(states).to(self._device).float()
    next_states = torch.ByteTensor(next_states).to(self._device).float()
    actions = torch.LongTensor(self['action'][indices]).to(self._device)
    rewards = torch.FloatTensor(self['reward'][indices]).to(self._device)
    dones = torch.FloatTensor(self['done'][indices]).to(self._device)

    return states, actions, rewards, next_states, dones

  def __len__(self):
    return len(self['state'])

  def get(self):
    return dict(self)

  def load(self, memory):
    for key in self._state_keys:
      self[key].extend(memory[key])

    num_data = len(memory['state'])
    if self._p + num_data <= self._capacity:
      for key in self._np_keys:
        self[key][self._p:self._p + num_data] = memory[key]
    else:
      mid_index = self._capacity - self._p
      end_index = num_data - mid_index
      for key in self._np_keys:
        self[key][self._p:] = memory[key][:mid_index]
        self[key][:end_index] = memory[key][mid_index:]

    self._n = min(self._n + num_data, self._capacity)
    self._p = (self._p + num_data) % self._capacity
    self.truncate()
    assert self._n == len(self)


class LazyMultiStepMemory(LazyMemory):

  def __init__(self, capacity, state_shape, device, gamma=0.99, multi_step=3):
    super(LazyMultiStepMemory, self).__init__(capacity, state_shape, device)

    self._gamma = gamma
    self._multi_step = int(multi_step)
    if self._multi_step != 1:
      self._buff = MultiStepBuff(maxlen=self._multi_step)

  def append(self, state, action, reward, next_state, done):
    if self._multi_step != 1:
      self._buff.append(state, action, reward)

      if self._buff.is_full():
        state, action, reward = self._buff.get(self._gamma)
        self._append(state, action, reward, next_state, done)

      if done:
        while not self._buff.is_empty():
          state, action, reward = self._buff.get(self._gamma)
          self._append(state, action, reward, next_state, done)
    else:
      self._append(state, action, reward, next_state, done)
