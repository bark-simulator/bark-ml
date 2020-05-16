# Copyright (c) 2019 Patrick Hart, Julian Bernhard,
# Klemens Esterle, Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import numpy as np

# BARK imports
from bark_project.modules.runtime.commons.parameters import ParameterServer

# BARK-ML imports
from bark_ml.environments.single_agent_runtime import SingleAgentRuntime
from bark_ml_library.observers import NearestObserver
from bark_ml.environments.blueprints import ContinuousHighwayBlueprint

# create scenario
params = ParameterServer()
bp = ContinuousHighwayBlueprint(params,
                                number_of_senarios=10,
                                random_seed=0)

# arguments that are additionally set in the runtime
# overwrite the ones of the blueprint
# e.g. we can change observer to the cpp observer
observer = NearestObserver(params)
env = SingleAgentRuntime(blueprint=bp,
                         observer=observer,
                         render=True)

# gym interface
env.reset()

done = False
while done is False:
  action = np.random.uniform(low=-0.1, high=0.1, size=(2, ))
  observed_next_state, reward, done, info = env.step(action)
  print(f"Observed state: {observed_next_state}, Reward: {reward}, Done: {done}")