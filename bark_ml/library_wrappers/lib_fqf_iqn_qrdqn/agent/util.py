# Copyright (c) 2020 Julian Bernhard,
# Klemens Esterle, Patrick Hart, Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


def default_training_evaluators():
  default_config = {"success" : "EvaluatorGoalReached", "collision_other" : "EvaluatorCollisionEgoAgent",
       "out_of_drivable" : "EvaluatorDrivableArea", "max_steps": "EvaluatorStepCount"}
  return default_config

def default_terminal_criteria(max_episode_steps):
  terminal_when = {"collision_other" : lambda x: x, "out_of_drivable" : lambda x: x, \
        "max_steps": lambda x : x>max_episode_steps, "success" : lambda x: x}
  return terminal_when