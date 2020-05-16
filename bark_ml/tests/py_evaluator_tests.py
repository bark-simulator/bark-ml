# Copyright (c) 2019 Patrick Hart, Julian Bernhard, Klemens Esterle, Tobias Kessler
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
from bark_ml.evaluators.goal_reached import GoalReached


class PyEvaluatorTests(unittest.TestCase):
  def test_goal_reached_observer(self):
    params = ParameterServer()
    bp = ContinuousHighwayBlueprint(params)
    env = SingleAgentRuntime(blueprint=bp, render=True)
    env.reset()
    world = env._world

    eval_id = env._scenario._eval_agent_ids[0]
    observed_world = world.Observe([eval_id])[0]
    evaluator = GoalReached(params)
    action = np.array([0., 0.], dtype=np.float32)
    print(evaluator.Evaluate(observed_world, action))
    

if __name__ == '__main__':
  unittest.main()