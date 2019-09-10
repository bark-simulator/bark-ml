
import numpy as np

from src.commons.spaces import Discrete, BoundedContinuous
from bark.models.behavior import DynamicBehaviorModel
from bark.models.dynamic import SingleTrackModel
from modules.runtime.commons.parameters import ParameterServer
from src.wrappers.action_wrapper import ActionWrapper

class DynamicModel(ActionWrapper):
  def __init__(self,
               params=ParameterServer(),
               dynamic_model=SingleTrackModel()):
    ActionWrapper.__init__(self, params)
    self._control_inputs = \
      self._params["Runtime"]["RL"]["ActionWrapper"]["ActionDimension",
      "Dimension of action",
      2] # (acceleration, steering angle)
    self._dynamic_model = dynamic_model
    self._behavior_model = DynamicBehaviorModel(dynamic_model,
                                                self._params)

  def reset(self, world, agents_to_act):
    super(DynamicModel, self).reset(world=world,
                                    agents_to_act=agents_to_act)
    self._behavior_model = DynamicBehaviorModel(self._dynamic_model,
                                                self._params)
    ego_agent_id = agents_to_act[0]
    if ego_agent_id in world.agents:
      world.agents[ego_agent_id].behavior_model = self._behavior_model
    else:
      raise ValueError("Id of controlled agent not in world agent map.")
    return world

  def action_to_behavior(self, world, action):
    if self._behavior_model:
      self._behavior_model.set_action(action)
    return world

  @property
  def action_space(self):
    # TODO(@hart): get these parameters
    return BoundedContinuous(self._control_inputs,
                             low=[-1.0, -0.1],
                             high=[1.0, 0.1])