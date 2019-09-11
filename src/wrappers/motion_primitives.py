
import numpy as np

from src.commons.spaces import Discrete, BoundedContinuous
from bark.models.behavior import BehaviorMotionPrimitives
from bark.models.dynamic import SingleTrackModel
from modules.runtime.commons.parameters import ParameterServer
from src.wrappers.action_wrapper import ActionWrapper


class MotionPrimitives(ActionWrapper):
  def __init__(self, params=ParameterServer()):
    ActionWrapper.__init__(self, params)
    self.control_inputs = \
      self._params["Runtime"]["RL"]["ActionWrapper"]["MotionPrimitives",
      "Motion primitives available as discrete actions", \
      [[4.,0.], [2.,0.],[-0.5,0.],[-1.,0.]]] # (acceleration, steering angle)
    self.behavior_model = None

  def reset(self, world, agents_to_act):
    super(MotionPrimitives, self).reset(world=world,
                                        agents_to_act=agents_to_act)
    self.behavior_model = BehaviorMotionPrimitives(SingleTrackModel(),
                                                   self._params)
    for control_input in self.control_inputs:
        self.behavior_model.add_motion_primitive(np.array(control_input))
    ego_agent_id = agents_to_act[0]
    if ego_agent_id in world.agents:
        world.agents[ego_agent_id].behavior_model = self.behavior_model
    else:
        raise ValueError("Id of controlled agent not in world agent map.")
    return world

  def action_to_behavior(self, world, action):
    if self.behavior_model:
      self.behavior_model.action_to_behavior(action)
    return world

  @property
  def action_space(self):
      return Discrete(len(self.control_inputs))