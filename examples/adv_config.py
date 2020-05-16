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
left_lane = LaneCorridorConfig(params=params,
                                road_ids=[16],
                                lane_corridor_id=0,
                                controlled_ids=None)
right_lane = LaneCorridorConfig(params=params,
                                road_ids=[16],
                                lane_corridor_id=1,
                                controlled_ids=True)
scenario_generation = \
  ConfigWithEase(
    num_scenarios=25,
    map_file_name="bark_ml/environments/blueprints/highway/city_highway_straight.xodr",  # NOLINT
    random_seed=0,
    params=params,
    lane_corridor_configs=[left_lane, right_lane])

# viewer
viewer = MPViewer(params=params,
                  x_range=[-35, 35],
                  y_range=[-35, 35],
                  follow_agent_id=True)
# define step_time, evaluator, observer and the behavior (cont./discrete)
dt = 0.2
evaluator = GoalReached(params)
observer = NearestAgentsObserver(params)
ml_behavior = ContinuousMLBehavior(params)

# create environment; has gym interface
env = SingleAgentRuntime(
  ml_behavior=ml_behavior,
  observer=observer,
  evaluator=evaluator,
  step_time=dt,
  viewer=viewer,
  scenario_generator=scenario_generation,
  render=True)

# now we have the same gym interface available
env.reset()

done = False
while done is False:
  action = np.random.uniform(low=-0.1, high=0.1, size=(2, ))
  observed_next_state, reward, done, info = env.step(action)
  print(f"Observed state: {observed_next_state}, Reward: {reward}, Done: {done}")