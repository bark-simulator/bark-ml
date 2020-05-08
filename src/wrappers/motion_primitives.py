
import numpy as np

from src.commons.py_spaces import Discrete, BoundedContinuous
from bark.models.behavior import BehaviorMotionPrimitives, BehaviorMPMacroActions, \
                              BehaviorMacroActionsFromParamServer
from bark.models.dynamic import SingleTrackModel
from modules.runtime.commons.parameters import ParameterServer
from src.wrappers.action_wrapper import ActionWrapper


class MotionPrimitives(ActionWrapper):
  """ActionWrapper that uses MotionPrimitives
  """
  def __init__(self, params=ParameterServer()):
    ActionWrapper.__init__(self, params)
    model_type = params["ML"]["MotionPrimitives"]["ModelType", "Type of behavior model \
                used for all vehicles", "BehaviorMPMacroActions"]
    behavior_model_params = \
      self._params.AddChild("ML").AddChild("MotionPrimitives").AddChild("ModelParams")
    self._behavior_model = \
       self.get_motion_primitive_model(behavior_model_params, model_type)

  def get_motion_primitive_model(self, params, model_type):
    if model_type == "BehaviorMPMacroActions":
      model = BehaviorMacroActionsFromParamServer(params)
    elif model_type == "BehaviorMotionPrimtives":
      control_inputs = params["DynamicInputs",
        "Motion primitives available as discrete actions", \
        [[4.,0.], [2.,0.],[-0.5,0.],[-1.,0.]]]
      model, _ = self.model_from_model_type(model_type, params)
      for control_input in control_inputs:
        model.AddMotionPrimitive(np.array(control_input))
    return model

  def model_from_model_type(self, model_type, params):
    bark_model = eval("{}(params)".format(model_type))    
    return bark_model, params

  def reset(self, world, agents_to_act):
    """see base class
    """
    super(MotionPrimitives, self).reset(world=world,
                                        agents_to_act=agents_to_act)
    ego_agent_id = agents_to_act[0]
    if ego_agent_id in world.agents:
        world.agents[ego_agent_id].behavior_model = self._behavior_model
    else:
        raise ValueError("AgentID does not exist in world.")
    return world

  def action_to_behavior(self, world, action):
    """see base class
    """
    if self._behavior_model:
      self._behavior_model.ActionToBehavior(action)
    return world

  @property
  def action_space(self):
    """see base class
    """
    return Discrete(self._behavior_model.GetNumMotionPrimitives(None))