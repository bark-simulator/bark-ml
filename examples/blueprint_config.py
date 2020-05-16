# Copyright (c) 2019 Patrick Hart, Julian Bernhard,
# Klemens Esterle, Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import numpy as np

# BARK imports
from bark_project.modules.runtime.commons.parameters import ParameterServer
from bark_project.modules.runtime.viewer.matplotlib_viewer import MPViewer
from bark_project.modules.runtime.scenario.scenario_generation.config_with_ease import \
  LaneCorridorConfig, ConfigWithEase

# BARK-ML imports
from bark_ml.evaluators.goal_reached import GoalReached
from bark_ml.observers.nearest_state_observer import NearestAgentsObserver
from bark_ml.behaviors.cont_behavior import ContinuousMLBehavior
from bark_ml.environments.single_agent_runtime import SingleAgentRuntime


# create scenario
params = ParameterServer()
bp = ContinuousHighwayBlueprint(params,
                                number_of_senarios=10,
                                random_seed=0)
env = SingleAgentRuntime(blueprint=bp, render=False)

# gym interface
env.reset()

done = False
while done is False:
  action = np.random.uniform(low=-0.1, high=0.1, size=(2, ))
  observed_next_state, reward, done, info = env.step(action)
  print(f"Observed state: {observed_next_state}, Reward: {reward}, Done: {done}")