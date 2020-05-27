# Copyright (c) 2020 The diadem authors 
# 
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# ========================================================
#

import os
from diadem.environments import Environment

class DiademBarkEnvironment(Environment):
    def __init__(self, runtime,  params=None):
        super().__init__()
        self.runtime = runtime

    def _initialize_params(self, params=None):
        super()._initialize_params(params=params)

    def step(self, actions):
        next_observation, reward, done, info = self.runtime.step(actions)
        return next_observation, reward, done, info

    def reset(self, state=None):
        current_observation = self.runtime.reset()
        return current_observation, False

    def create_init_states(self, size=None, idx=None):
        pass

    def render(self, filename=None, show=True):
        self.runtime.render()

    @property
    def actionspace(self):
        return self.runtime.action_space

    @property
    def observationspace(self):
        return self.runtime.observation_space

    def contains_training_data(self):
        return True

    def state_as_string(self):
        return "BarkEnvironment" #todo: maybe print scenario idx

    @property
    def action_timedelta(self):
        # only for visualization purposes actual step time unknown
        return self.runtime.step_time
