import numpy as np
import math
import operator

from bark.models.dynamic import StateDefinition
from modules.runtime.commons.parameters import ParameterServer
from src.observers.simple_observer import SimpleObserver

class CustomObserver(SimpleObserver):
  def __init__(self, params=ParameterServer()):
    SimpleObserver.__init__(self,
                            normalize_observations=True,
                            params=params)

  def observe(self, world):
    # TODO(@hart): e.g. include distance to goal
    return super(CustomObserver, self).observe(world)

  def reset(self, world, agents_to_observe):
    world = super(CustomObserver, self).reset(world, agents_to_observe)
    return world