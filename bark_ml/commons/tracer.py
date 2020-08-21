# Copyright (c) 2020 fortiss GmbH
#
# Authors: Patrick Hart
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import pandas as pd
from enum import Enum

class Tracer:
  class QueryTypes(str, Enum):
    """
    Trace: returns the trace
    Mean: calculates the mean of the trace
    """
    TRACE = "TRACE"
    MEAN  = "MEAN"
    SUM  = "SUM"
    LAST_VALUE  = "LAST_VALUE"
    ANY_TRUE  = "ANY_TRUE"
  
  def __init__(self):
    self._states = []
    self._df = None

  def EvalStateToDict(self, eval_state):
    """Converts an eval_state to a flat eval_dict"""
    eval_dict = {}
    if type(eval_state) is tuple:
      (state, reward, is_terminal, info) = eval_state
      eval_dict["state"] = state
      eval_dict["reward"] = reward
      eval_dict["is_terminal"] = is_terminal
      for info_key, info_value in info.items():
        eval_dict[info_key] = info_value
    if type(eval_state) is dict:
      for info_key, info_value in eval_state.items():
        eval_dict[info_key] = info_value
    return eval_dict

  def Trace(self, eval_state, **kwargs):
    """Traces and stores a state"""
    eval_dict = self.EvalStateToDict(eval_state)
    # additional things to log
    for key, value in kwargs.items():
      eval_dict[key] = value
    self._states.append(eval_dict)
  
  def Query(
    self,
    key="collision",
    group_by="num_episode",
    agg_type="TRACE"):
    if self.df is None:
      self.ConvertToDf()
    df = self.df.groupby(group_by)[key]
    # NOTE: different aggregation types
    if agg_type == Tracer.QueryTypes.MEAN:
      return df.mean()
    if agg_type == Tracer.QueryTypes.SUM:
      return df.sum()
    if agg_type == Tracer.QueryTypes.LAST_VALUE:
      return df.tail(1)
    elif agg_type == Tracer.QueryTypes.ANY_TRUE:
      return df.any().mean()
    return df

  @property
  def df(self):
    return self._df
  
  def ConvertToDf(self):
    """Conversts states to pandas dataframe"""
    self._df = pd.DataFrame(self._states)
  
  def Save(self, filepath="./"):
    """Saves trace as pandas dataframe"""
    if self._df is None:
      self.ConvertoToDf()
    self._df.to_pickle(filepath)

  def Reset(self):
    self._states = []
    self._df = None