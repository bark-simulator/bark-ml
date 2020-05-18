# Copyright (c) 2020 Patrick Hart, Julian Bernhard,
# Klemens Esterle, Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import numpy as np

# BARK imports
from bark_project.modules.runtime.commons.parameters import ParameterServer
from modules.runtime.viewer.matplotlib_viewer import MPViewer
from modules.runtime.viewer.video_renderer import VideoRenderer

# BARK-ML imports
from bark_ml.environments.single_agent_runtime import SingleAgentRuntime
from bark_ml_library.observers import NearestObserver
from bark_ml.environments.blueprints import ContinuousHighwayBlueprint, \
  DiscreteHighwayBlueprint

# create scenario
params = ParameterServer()
bp = ContinuousHighwayBlueprint(params,
                                number_of_senarios=10,
                                random_seed=0)
# bp = DiscreteHighwayBlueprint(params,
#                               number_of_senarios=10,
#                               random_seed=0)


# arguments that are additionally set in the runtime
# overwrite the ones of the blueprint
# e.g. we can change observer to the cpp observer
observer = NearestObserver(params)
# viewer = MPViewer(params=params,
#                   x_range=[-35, 35],
#                   y_range=[-35, 35],
#                   follow_agent_id=True)
# viewer = VideoRenderer(renderer=viewer,
#                        world_step_time=0.2,
#                        fig_path="/Users/hart/2020/bark-ml/video/")
env = SingleAgentRuntime(blueprint=bp,
                         observer=observer,
                         render=True)

# gym interface
env.reset()
done = False
while done is False:
  action = np.random.uniform(
    low=np.array([-0.5, -0.02]), high=np.array([0.5, 0.02]), size=(2, ))
  observed_next_state, reward, done, info = env.step(action)
  print(f"Observed state: {observed_next_state}, Action: {action}, Reward: {reward}, Done: {done}")

# viewer.export_video(
#   filename="/Users/hart/2020/bark-ml/video/video", remove_image_dir=False)