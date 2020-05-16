# Copyright (c) 2020 Patrick Hart, Julian Bernhard,
# Klemens Esterle, Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


import unittest
import numpy as np
import os
import matplotlib
import time

from bark_ml.behaviors.cont_behavior import ContinuousMLBehavior
from bark_ml.behaviors.discrete_behavior import DiscreteMLBehavior
from bark_project.modules.runtime.commons.parameters import ParameterServer
from bark_ml.environments.blueprints import ContinuousHighwayBlueprint, DiscreteHighwayBlueprint
from bark_ml.environments.single_agent_runtime import SingleAgentRuntime
from bark_ml.observers.nearest_state_observer import NearestAgentsObserver
from bark_ml_library.observers import NearestObserver


class PyObserverTests(unittest.TestCase):
  def test_nearest_observer(self):
    params = ParameterServer()
    bp = ContinuousHighwayBlueprint(params)
    env = SingleAgentRuntime(blueprint=bp, render=True)
    env.reset()
    world = env._world

    # under test
    observer = NearestAgentsObserver(params)

    eval_id = env._scenario._eval_agent_ids[0]
    observed_world = world.Observe([eval_id])[0]
    observed_state = observer.Observe(observed_world)
    print(observed_state)
    
  def test_nearest_observer_cpp(self):
    params = ParameterServer()
    bp = ContinuousHighwayBlueprint(params)
    env = SingleAgentRuntime(blueprint=bp, render=True)
    env.reset()
    world = env._world

    # under test
    observer = NearestObserver(params)

    eval_id = env._scenario._eval_agent_ids[0]
    observed_world = world.Observe([eval_id])[0]
    observed_state = observer.Observe(observed_world)
    print(observed_state)


if __name__ == '__main__':
  unittest.main()