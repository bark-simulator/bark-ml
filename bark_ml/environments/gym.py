# Copyright (c) 2019 Patrick Hart, Julian Bernhard,
# Klemens Esterle, Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import gym
from gym.envs.registration import register

from bark_project.modules.runtime.commons.parameters import ParameterServer

from bark_ml.environments.blueprints.highway.highway import ContinuousHighwayBlueprint
from bark_ml.environments.single_agent_runtime import SingleAgentRuntime


class ContinuousHighwayGym(SingleAgentRuntime, gym.Env):
  def __init__(self):
    params = ParameterServer()
    cont_highway_bp = ContinuousHighwayBlueprint(params)
    SingleAgentRuntime.__init__(self,
      blueprint=cont_highway_bp)

class DiscreteHighwayGym(SingleAgentRuntime, gym.Env):
  def __init__(self):
    params = ParameterServer()
    discrete_highway_bp = DiscreteHighwayGym(params)
    SingleAgentRuntime.__init__(self,
      blueprint=discrete_highway_bp)

# register gym envs
register(
  id='highway-v0',
  entry_point='bark_ml.environments.gym:ContinuousHighwayGym'
)
register(
  id='highway-v1',
  entry_point='bark_ml.environments.gym:DiscreteHighwayGym'
)