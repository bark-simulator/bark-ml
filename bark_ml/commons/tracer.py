# Copyright (c) 2020 fortiss GmbH
#
# Authors: Patrick Hart
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import pickle

class Tracer:
  """The tracer can be used to log certain values during episodes."""

  def __init__(self, states=None, trace_history=True):
    self._trace_history = trace_history
    self._states = []

  def Trace(self, eval_dict):
    """Traces and stores a state."""
    if self._trace_history:
      self._states.append(eval_dict)

  def Reset(self):
    self._trace_history = []