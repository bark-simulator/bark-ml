import pandas as pd
from enum import Enum

class Tracer:
  class QueryTypes(str, Enum):
    """
    Trace: returns the trace
    Mean: calculates the mean of the trace
    """
    trace = "Trace"
    mean  = "Mean"
  
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
    type="Mean"):
    if self._df == None:
      self.ConvertToDf()
    # NOTE: insert pandas logic here
    df = self.df.groupby([group_by])

  @property
  def df(self):
    return self._df
  
  def ConvertToDf(self):
    """Conversts states to pandas dataframe"""
    self._df = pd.DataFrame(self._states)
  
  def Save(self, filepath="./"):
    """Saves trace as pandas dataframe"""
    if self._df == None:
      self.ConvertoToDf()
    self._df.to_pickle(filepath)

  def Reset(self):
    self._states = []
    self._df = None