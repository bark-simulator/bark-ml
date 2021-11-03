# Copyright (c) 2020 fortiss GmbH
#
# Authors: Patrick Hart
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import gym
import os
from gym.envs.registration import register

from bark.runtime.commons.parameters import ParameterServer

from bark_ml.environments.blueprints.highway.highway import \
  ContinuousHighwayBlueprint, DiscreteHighwayBlueprint
from bark_ml.environments.blueprints.merging.merging import \
  ContinuousMergingBlueprint, DiscreteMergingBlueprint
from bark_ml.environments.blueprints.single_lane.single_lane import \
  ContinuousSingleLaneBlueprint
from bark_ml.environments.blueprints.intersection.intersection import \
  ContinuousIntersectionBlueprint, DiscreteIntersectionBlueprint
from bark_ml.environments.single_agent_runtime import SingleAgentRuntime

# highway
class ContinuousHighwayGym(SingleAgentRuntime, gym.Env):
  """Highway scenario with continuous behavior model.

  Behavior model takes the steering-rate and acceleration.
  """
  def __init__(self):
    params = ParameterServer(filename=
      os.path.join(os.path.dirname(__file__),
      "../environments/blueprints/visualization_params.json"))
    cont_highway_bp = ContinuousHighwayBlueprint(params)
    SingleAgentRuntime.__init__(self,
      blueprint=cont_highway_bp, render=True)

class DiscreteHighwayGym(SingleAgentRuntime, gym.Env):
  """Highway scenario with discrete behavior model.

  Behavior model takes integers [0, 1, 2] as specified in the
  discrete behavior model.
  """

  def __init__(self,
    params = ParameterServer(filename= \
      os.path.join(os.path.dirname(__file__),
      "../environments/blueprints/visualization_params.json")),
    render=False):
    discrete_highway_bp = DiscreteHighwayBlueprint(params)
    SingleAgentRuntime.__init__(self,
      blueprint=discrete_highway_bp, render=render)

# merging
class ContinuousMergingGym(SingleAgentRuntime, gym.Env):
  """Merging scenario with continuous behavior model.

  Behavior model takes the steering-rate and acceleration.
  """

  def __init__(self):
    params = ParameterServer(filename= \
      os.path.join(os.path.dirname(__file__),
      "../environments/blueprints/visualization_params.json"))
    cont_merging_bp = ContinuousMergingBlueprint(params)
    SingleAgentRuntime.__init__(self,
      blueprint=cont_merging_bp, render=True)

class DiscreteMergingGym(SingleAgentRuntime, gym.Env):
  """
  Merging scenario with discrete behavior model.
  Behavior model takes integers [0, 1, 2] as specified in the
  discrete behavior model.
  """

  def __init__(self,
    params = ParameterServer(filename= \
      os.path.join(os.path.dirname(__file__),
      "../environments/blueprints/visualization_params.json")),
    render=False):
    discrete_merging_bp = DiscreteMergingBlueprint(params)
    SingleAgentRuntime.__init__(self,
      blueprint=discrete_merging_bp, render=render)

# merging
class MediumContinuousMergingGym(SingleAgentRuntime, gym.Env):
  """Merging scenario (medium dense traffic) with continuous behavior model.

  Behavior model takes the steering-rate and acceleration.
  """

  def __init__(self):
    params = ParameterServer(filename= \
      os.path.join(os.path.dirname(__file__),
      "../environments/blueprints/visualization_params.json"))
    cont_merging_bp = ContinuousMergingBlueprint(params, mode="medium")
    SingleAgentRuntime.__init__(self,
      blueprint=cont_merging_bp,
      render=True)

# intersection
class ContinuousIntersectionGym(SingleAgentRuntime, gym.Env):
  """Intersection scenario with continuous behavior model.

  Behavior model takes the steering-rate and acceleration.
  """

  def __init__(self):
    params = ParameterServer(filename= \
      os.path.join(os.path.dirname(__file__),
      "../environments/blueprints/visualization_params.json"))
    cont_merging_bp = ContinuousIntersectionBlueprint(params)
    SingleAgentRuntime.__init__(self,
      blueprint=cont_merging_bp, render=True)

class DiscreteIntersectionGym(SingleAgentRuntime, gym.Env):
  """Intersection scenario with discrete behavior model.

  Behavior model takes integers [0, 1, 2] as specified in the
  discrete behavior model.
  """

  def __init__(self,
    params = ParameterServer(filename= \
      os.path.join(os.path.dirname(__file__),
      "../environments/blueprints/visualization_params.json")),
    render=False):
    discrete_merging_bp = DiscreteIntersectionBlueprint(params)
    SingleAgentRuntime.__init__(self,
      blueprint=discrete_merging_bp, render=render)

class GymSingleAgentRuntime(SingleAgentRuntime, gym.Wrapper):
  """Wraps the BARK environment for OpenAI-Gym."""

  def __init__(self, *args, **kwargs):
    SingleAgentRuntime.__init__(self, *args, **kwargs)


# highway
class ContinuousSingleLaneGym(SingleAgentRuntime, gym.Env):
  """Highway scenario with continuous behavior model.

  Behavior model takes the steering-rate and acceleration.
  """
  def __init__(self):
    params = ParameterServer(filename=
      os.path.join(os.path.dirname(__file__),
      "../environments/blueprints/visualization_params.json"))
    cont_highway_bp = ContinuousSingleLaneBlueprint(params)
    SingleAgentRuntime.__init__(self,
      blueprint=cont_highway_bp, render=True)


# register gym envs
register(
  id='highway-v0',
  entry_point='bark_ml.environments.gym:ContinuousHighwayGym'
)
register(
  id='highway-v1',
  entry_point='bark_ml.environments.gym:DiscreteHighwayGym'
)
register(
  id='merging-v0',
  entry_point='bark_ml.environments.gym:ContinuousMergingGym'
)
register(
  id='merging-medium-v0',
  entry_point='bark_ml.environments.gym:MediumContinuousMergingGym'
)
register(
  id='merging-v1',
  entry_point='bark_ml.environments.gym:DiscreteMergingGym'
)
register(
  id='intersection-v0',
  entry_point='bark_ml.environments.gym:ContinuousIntersectionGym'
)
register(
  id='intersection-v1',
  entry_point='bark_ml.environments.gym:DiscreteIntersectionGym'
)
register(
  id='singlelane-v0',
  entry_point='bark_ml.environments.gym:ContinuousSingleLaneGym'
)