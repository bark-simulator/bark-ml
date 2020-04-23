
import numpy as np
import itertools
from src.commons.py_spaces import Discrete, BoundedContinuous
from bark.models.behavior import DynamicBehaviorModel
from bark.models.dynamic import SingleTrackModel, TripleIntegratorModel
from modules.runtime.commons.parameters import ParameterServer
from src.wrappers.action_wrapper import ActionWrapper

class DynamicModel(ActionWrapper):
  """This module wraps the SingleTrack model
     and requires the steering angle and acceleration
     as system inputs. 
  """
  def __init__(self,
               model_name="SingleTrackModel",
               params=ParameterServer()):
    ActionWrapper.__init__(self, params)
    self._control_inputs = \
      self._params["ML"]["DynamicModel"]["action_dimension",
      "Dimension of action",
      2]
    self._dynamic_model = eval("{}(self._params)".format(model_name))
    self._behavior_models = []
    self._controlled_agents = []

  def reset(self, world, agents_to_act):
    """see base class
    """
    super(DynamicModel, self).reset(world=world,
                                    agents_to_act=agents_to_act)
    self._behavior_models = []
    self._controlled_agents = agents_to_act
    for agent_id in agents_to_act:
      self._behavior_models.append(DynamicBehaviorModel(self._dynamic_model,
                                                        self._params))
      if agent_id in world.agents:
        actions = np.zeros(shape=(self._control_inputs), dtype=np.float32)
        self._behavior_models[-1].SetLastAction(actions)
        world.agents[agent_id].behavior_model = self._behavior_models[-1]
      else:
        raise ValueError("AgentID does not exist in world.")
    return world

  def action_to_behavior(self, world, action):
    """see base class
    """
    actions = np.reshape(action, (-1, self._control_inputs))
    for i, a in enumerate(actions):
      self._behavior_models[i].SetLastAction(a)
    return world

  @property
  def action_space(self):
    """see base class
    """
    action_num = self._params["ML"]["DynamicModel"]["action_num",
        "Lower-bound for actions.",
        1]
    lower_bounds = [self._params["ML"]["DynamicModel"]["actions_lower_bound",
        "Lower-bound for actions.",
        [-0.5, -0.01]] for _ in range(action_num)]
    upper_bounds = [self._params["ML"]["DynamicModel"]["actions_upper_bound",
        "Upper-bound for actions.",
        [0.5, 0.01]] for _ in range(action_num)]
    return BoundedContinuous(
      self._control_inputs*action_num,
      low=list(itertools.chain(*lower_bounds)),
      high=list(itertools.chain(*upper_bounds)))