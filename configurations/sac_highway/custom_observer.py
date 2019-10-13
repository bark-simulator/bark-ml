import numpy as np
from bark.models.dynamic import StateDefinition
from modules.runtime.commons.parameters import ParameterServer
import math
import operator
from src.commons.spaces import BoundedContinuous, Discrete

from src.observers.simple_observer import SimpleObserver

class CustomObserver(SimpleObserver):
  def __init__(self, params=ParameterServer()):
    SimpleObserver.__init__(self,
                            params)
    self._perform_lane_change = False

  def observe(self, world, agents_to_observe):
    # TODO(@hart): modify for conditional input
    state = super(CustomObserver, self).observe(world, agents_to_observe)
    
    return state

  def reset(self, world, agents_to_observe):
    super(CustomObserver, self).reset(world, agents_to_observe)
    extended_state = super(CustomObserver, self).observe(world, agents_to_observe)
    rn = np.random.randint(0, 2)
    # extend the state space with lane change parameter rn
    extended_state = np.append(extended_state,rn)
    self._params["ML"]["Maneuver"]["lane_change",
      "Whether a lane change should be performed or not.",
       rn]
    #print(rn, self._params["ML"]["Maneuver"]["lane_change"])
    return world