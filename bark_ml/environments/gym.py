# Copyright (c) 2019 Patrick Hart, Julian Bernhard,
# Klemens Esterle, Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
from gym.envs.registration import register

from modules.runtime.commons.parameters import ParameterServer

from bark_ml.environments.blueprints.highway.highway import ContinuousHighwayBlueprint
from bark_ml.environments.single_agent_runtime import SingleAgentRuntime


register(
  id='highway-v0',
  entry_point='xxx'
)