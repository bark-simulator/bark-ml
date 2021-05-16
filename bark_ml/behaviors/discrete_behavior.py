# Copyright (c) 2020 fortiss GmbH
#
# Authors: Patrick Hart, Julian Bernhard, Klemens Esterle, and
# Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import numpy as np

from bark.core.models.behavior import BehaviorMPContinuousActions, \
  BehaviorMPMacroActions, BehaviorMacroActionsFromParamServer

from bark_ml.commons.py_spaces import Discrete


class BehaviorDiscreteMotionPrimitivesML(BehaviorMPContinuousActions):
  def __init__(self,
               params=None):
    BehaviorMPContinuousActions.__init__(
      self,
      params)
    self._min_max_acc = params["ML"]["BehaviorDiscretePrimitivesML"][
      "MinMaxAcc", "", [-3., 3.]]
    self._acc_d_steps = params["ML"]["BehaviorDiscretePrimitivesML"][
      "AccDiscretizationSteps", "", 10]
    self._min_max_steer = params["ML"]["BehaviorDiscretePrimitivesML"][
      "MinMaxSteeringRate", "", [-.2, .2]]
    self._steer_d_steps = params["ML"]["BehaviorDiscretePrimitivesML"][
      "SteeringRateDiscretizationSteps", "", 5]

    # add motion primitives
    for acc in np.linspace(
      self._min_max_acc[0], self._min_max_acc[1], self._acc_d_steps):
      for steering_rate in np.linspace(
        self._min_max_steer[0], self._min_max_steer[1], self._steer_d_steps):
        super().AddMotionPrimitive(
          np.array([acc, steering_rate], dtype=np.float32))

  @property
  def action_space(self):
    return Discrete(self.GetNumMotionPrimitives(None))


class BehaviorDiscreteMacroActionsML(BehaviorMPMacroActions):
  def __init__(self,
               params=None):
    self._macro_action_params = params.AddChild("ML").AddChild("BehaviorMPMacroActions")
    self._macro_action_params["BehaviorMPMacroActions"]["CheckValidityInPlan", "", False]
    default_model = BehaviorMacroActionsFromParamServer(self._macro_action_params)
    super().__init__(self._macro_action_params, default_model.GetMotionPrimitives())

  @property
  def action_space(self):
    return Discrete(len(self.GetMotionPrimitives()))