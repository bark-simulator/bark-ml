

import numpy as np
from abc import ABC, abstractmethod

from src.commons.spaces import Discrete, BoundedContinuous
from bark.models.behavior import BehaviorMotionPrimitives, \
  DynamicBehaviorModel
from bark.models.dynamic import SingleTrackModel
from modules.runtime.commons.parameters import ParameterServer


class ActionWrapper(ABC):
  def __init__(self, params):
    self._params = params

  @abstractmethod
  def action_to_behavior(self, world, actions):
    pass

  @abstractmethod
  def reset(self, world, agents_to_act):
    pass
  
  @property
  def action_space(self):
    pass


