# Copyright (c) 2020 fortiss GmbH
#
# Authors: Patrick Hart, Julian Bernhard, Klemens Esterle, and
# Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


import unittest
import time

# Bark imports
from bark.runtime.commons.parameters import ParameterServer

# Bark-ml imports
from bark_ml.environments.blueprints import ContinuousHighwayBlueprint
from bark_ml.environments.single_agent_runtime import SingleAgentRuntime
from bark_ml.observers.nearest_state_observer import NearestAgentsObserver
from bark_ml.core.observers import NearestObserver


class PyObserverTests(unittest.TestCase):
  """Observer tests."""

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
    start_time = time.time()
    observed_state = observer.Observe(observed_world)
    end_time = time.time()
    print(f"It took {end_time-start_time} seconds.")
    print(observed_state, observer.observation_space.shape)

  def test_nearest_observer_cpp(self):
    params = ParameterServer()
    bp = ContinuousHighwayBlueprint(params)
    env = SingleAgentRuntime(blueprint=bp, render=True)
    env.reset()
    world = env._world

    # under test
    observer = NearestObserver(params)
    observer.Reset(world)

    eval_id = env._scenario._eval_agent_ids[0]
    observed_world = world.Observe([eval_id])[0]
    start_time = time.time()
    observed_state = observer.Observe(observed_world)
    end_time = time.time()
    print(f"It took {end_time-start_time} seconds.")
    print(observed_state, observer.observation_space.shape)



if __name__ == '__main__':
  unittest.main()