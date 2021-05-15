# Copyright (c) 2020 fortiss GmbH
#
# Authors: Patrick Hart, Julian Bernhard, Klemens Esterle, and
# Tobias Kessler, Mansoor Nasir
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from collections import deque
import numpy as np
import logging
import os
import math
import random

import torch
from torch import nn
from torch.optim import Adam, RMSprop

from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.model import Imitation
from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.agent.demonstrations import ActionValuesCollector
from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.utils \
 import disable_gradients, update_params, \
 calculate_quantile_huber_loss, evaluate_quantile_at_action
from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.agent.loss.loss_function \
  import apply_sigmoid_to_dict, LossMSE, LossBCE
from .base_agent import BaseAgent, TrainingBenchmark

class BenchmarkSupervisedLoss(TrainingBenchmark):
    def __init__(self, demonstrations_test):
      self.demonstrations_test = demonstrations_test

    def run(self):
      with torch.no_grad():
          states, action_values_desired = self.agent.sample_batch( \
                  self.demonstrations_test, self.agent.num_eval_episodes)
          action_values_current = self.agent.online_net(states)

          converted_desired_values = self.agent.convert_values(action_values_desired)
          converted_current_values = self.agent.convert_values(action_values_current)

          loss = self.agent.calculate_loss(converted_current_values,
                                           converted_desired_values)

          return self.evaluate_loss(loss.item(), converted_current_values, converted_desired_values)

    def evaluate_loss(self, scalar_loss, converted_current_values, converted_desired_values,
                      phase="test", logits=False):
      mean_values_formatted = ""

      if logits:
        converted_current_values = apply_sigmoid_to_dict(converted_current_values)

      for key in converted_desired_values.keys():
        pair_wise_diff = converted_current_values[key] - converted_desired_values[key]
        mean_value_diffs = torch.mean(pair_wise_diff**2, dim = 0).tolist()
        mean_values_formatted += f"-> {key:10} {mean_value_diffs}\n"

      return {f"mse_loss/{phase}" : scalar_loss},\
             f"Loss = {scalar_loss}\n Mean Diff: \n{mean_values_formatted}"

    def is_better(self, eval_result1, than_eval_result2):
        return eval_result1["mse_loss/test"] < than_eval_result2["mse_loss/test"]

class BenchmarkSplitSupervisedLoss(TrainingBenchmark):
  def __init__(self, demonstrations_test, gradients_and_weights_dir=None):
    self.demonstrations_test = demonstrations_test
    self.gradients_and_weights_dir = gradients_and_weights_dir

  def run(self):
    with torch.no_grad():
      states, action_values_desired = self.agent.sample_batch( \
              self.demonstrations_test, self.agent.num_eval_episodes)
      action_values_current = self.agent.online_net(states)

      converted_desired_values = self.agent.convert_values(action_values_desired)
      converted_current_values = self.agent.convert_values(action_values_current)

      loss = self.agent.calculate_loss(converted_current_values,
                                       converted_desired_values)

      return self.evaluate_loss(loss.item(), converted_current_values, converted_desired_values)

  def evaluate_loss(self, scalar_loss, converted_current_values, converted_desired_values,
                    phase="test", logits=False):
    result = {f"loss/{phase}": scalar_loss}
    formatted_result = f"Loss = {scalar_loss}\n          | Mean SE | Var SE\n"

    if logits:
      converted_current_values = apply_sigmoid_to_dict(converted_current_values)

    for key in converted_desired_values.keys():
      pair_wise_diff = converted_desired_values[key] - converted_current_values[key]
      pair_wise_diff_squared = pair_wise_diff**2
      error_var, error_mean = torch.var_mean(pair_wise_diff_squared)
      error_var = error_var.item()
      error_mean = error_mean.item()

      result[f"squared_error_mean/{key}/{phase}"] = error_mean
      result[f"squared_error_var/{key}/{phase}"] = error_var

      formatted_result += f"{key:10}| {error_mean:7.3f} | {error_var:7.3f}\n"

    if self.gradients_and_weights_dir:
      self.save_gradients_and_weights_to_file(
          self.agent.online_net.named_parameters(), self.agent.steps, phase)

    return result, formatted_result

  def is_better(self, eval_result1, than_eval_result2):
    return eval_result1["loss/test"] < than_eval_result2["loss/test"]

  def save_gradients_and_weights_to_file(self, named_parameters, steps, phase):
    weights = dict()
    gradients = dict()

    for name, param in named_parameters:
      if "bias" in name:
        continue
      weights[name] = param.cpu().detach().numpy()
      gradients[name] = param.grad.cpu().detach().numpy()

    with open(f"{self.gradients_and_weights_dir}/weights_{phase}_{steps}.npz",
              "wb") as outfile:
      np.savez(outfile, **weights)
    with open(
        f"{self.gradients_and_weights_dir}/gradients_{phase}_{steps}.npz",
        "wb") as outfile:
      np.savez(outfile, **gradients)


class ImitationAgent(BaseAgent):
  def __init__(self, demonstration_collector = None, *args, **kwargs):
    self.demonstration_collector = demonstration_collector
    super(ImitationAgent, self).__init__(*args, **kwargs)
    self.define_training_test_data()

    if self._params["ML"]["ImitationModel"]["SaveGradientsWeights", "", False]:
      gradients_and_weights_dir = f"{self.agent_save_dir}/gradients_weights"
      if not os.path.exists(gradients_and_weights_dir):
        os.mkdir(gradients_and_weights_dir)
      logging.info("Saving gradients and weights to {}".format(
          os.path.abspath(gradients_and_weights_dir)))
    else:
      gradients_and_weights_dir = None

    self._training_benchmark = BenchmarkSplitSupervisedLoss(
      self.demonstrations_test, gradients_and_weights_dir=gradients_and_weights_dir)
    self._training_benchmark.reset(None, self.num_eval_episodes, None, self)

    self.select_loss_function(self._params)

  def define_training_test_data(self):
    demonstrations = self.demonstration_collector.GetDemonstrationExperiences()
    random.shuffle(demonstrations)
    self.demonstrations_train = demonstrations[0:math.floor(len(demonstrations)*self.train_test_ratio)]
    self.demonstrations_test = demonstrations[math.ceil(len(demonstrations)*self.train_test_ratio):]

  def reset_params(self, params):
    super(ImitationAgent, self).reset_params(params)
    self.running_loss_length = params["RunningLossLength", "", 1000]
    self.num_value_functions = params["NumValueFunctions", "", 3]
    self.learning_rate = params["LearningRate", "", 0.001]
    self.train_test_ratio = params["TrainTestRatio", "", 0.2]

  def reset_training_variables(self):
    # Replay memory which is memory-efficient to store stacked frames.
    self.running_loss = deque(maxlen=self.running_loss_length)
    self.steps = 0
    self.best_eval_results = None

  def reset_action_observer(self, env):
    pass

  @property
  def observer(self):
    return self.demonstration_collector.observer

  @property
  def motion_primitive_behavior(self):
    return self.demonstration_collector.motion_primitive_behavior

  @property
  def num_actions(self):
    return len(self.motion_primitive_behavior.GetMotionPrimitives())

  def init_always(self):
    super(ImitationAgent, self).init_always()
    self.init_network()

  def init_network(self):
    # Target network.
    self.online_net = Imitation(num_channels=self.observer.observation_space.shape[0],
                          num_actions=self.num_actions,
                          num_value_functions=self.num_value_functions,
                          params=self._params).to(self.device)
    self.optim = RMSprop(
        self.online_net.parameters(),
        lr=self.learning_rate,
        alpha=0.95,
        eps=0.00001)

  def clean_pickables(self, pickables):
    del pickables["_env"]
    del pickables["_training_benchmark"]
    del pickables["device"]
    del pickables["writer"]
    del pickables["_checkpoint_load"]
    del pickables["online_net"]
    del pickables["optim"]
    del pickables["demonstrations_train"]
    del pickables["demonstration_collector"]

  def sample_batch(self, demonstrations_list, batch_size):
    state_size = len(demonstrations_list[0][0])
    num_action_values = len(demonstrations_list[0][1])
    states = np.empty((batch_size, state_size), dtype=np.float)
    action_values = np.empty((batch_size, num_action_values), dtype=np.float)

    indices = np.random.randint(low=0, high=len(demonstrations_list), size=batch_size)
    for i, index in enumerate(indices):
      states[i, ...] = demonstrations_list[index][0]
      action_values[i, ...] = demonstrations_list[index][1]

    states = torch.FloatTensor(states).to(self.device)
    action_values = torch.FloatTensor(action_values).to(self.device)

    return states, action_values

  def calculate_loss(self, action_values_current, action_values_desired, logits=False):
    """
    If logits=True, the loss must use the sigmoid function before comparing
    current values to the desired values.
    Some loss functions (e.g., BCE) need logits for a better numerical
    stability while training.
    """
    # if we have missing actions during demo collections these become nan values
    # they should influence the loss and are thus set to current nn output
    for key in action_values_desired.keys():
      nans_desired = torch.isnan(action_values_desired[key]).nonzero()
      action_values_desired[key][nans_desired] = action_values_current[key][nans_desired].detach().clone()

    loss = self.selected_loss(action_values_current, action_values_desired, logits)
    return loss

  def calculate_actions(self, state):
    with torch.no_grad():
        state_torch = torch.FloatTensor(state).to(self.device)
        action_values = self.online_net(state_torch)
    return action_values.tolist()

  def save_models(self, checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)
    torch.save(self.online_net.state_dict(),
               os.path.join(checkpoint_dir, 'online_net.pth'))
    online_net_script = torch.jit.script(self.online_net)
    online_net_script.save(os.path.join(checkpoint_dir, 'online_net_script.pt'))

  def save(self, checkpoint_type="last"):
    self.demonstration_collector_dir = self.demonstration_collector.GetDirectory()
    super(ImitationAgent, self).save(checkpoint_type)

  def load_other(self):
    self.demonstration_collector = ActionValuesCollector.load(self.demonstration_collector_dir)

  @property
  def nn_to_value_converter(self):
    return self.online_net.nn_to_value_converter

  def load_models(self, checkpoint_dir):
    try:
      self.online_net.load_state_dict(
        torch.load(os.path.join(checkpoint_dir, 'online_net.pth')))
    except RuntimeError:
      self.online_net.load_state_dict(
        torch.load(os.path.join(checkpoint_dir, 'online_net.pth'), map_location=torch.device('cpu')))

  def train_episode(self):
    states, action_values_desired = self.sample_batch(self.demonstrations_train, self.batch_size)

    self.optim.zero_grad()
    action_values_current = self.online_net(states)

    converted_desired_values = self.convert_values(action_values_desired)
    converted_current_values = self.convert_values(action_values_current)

    loss = self.calculate_loss(converted_current_values, converted_desired_values, logits=True)
    loss.backward()
    self.optim.step()

    self.training_log(loss, converted_current_values, converted_desired_values)

  def training_log(self, loss, current_values, desired_values):
    self.running_loss.append(loss.item())
    # We log evaluation results along with training frames = 4 * steps.
    if self.steps % self.summary_log_interval == 0:
        self.online_net.eval()
        running_loss_avg = sum(self.running_loss)/len(self.running_loss)

        logging.info(f"Training: Loss(i={self.steps}={running_loss_avg})")
        eval_results, _ = self._training_benchmark.evaluate_loss(
          running_loss_avg, current_values, desired_values, phase="train", logits=True)

        for eval_result_name, eval_result in eval_results.items():
          self.writer.add_scalar(eval_result_name, eval_result, self.steps)

        self.online_net.train()
    if self.steps % self.eval_interval == 0:
      self.evaluate()
      self.save("final")
      self.online_net.train()
    self.steps += 1

  def select_loss_function(self, params):
    loss_params = params["ML"]["ImitationModel"]["Loss"]
    if loss_params["BinaryCrossEntropyLoss"]:
      if loss_params["BinaryCrossEntropyLoss"]["Weights"]:
        weights = loss_params["BinaryCrossEntropyLoss"]["Weights"]
      else:
        weights = None
      self.selected_loss = LossBCE(weights)

    elif loss_params["MeanSquaredErrorLoss"]:
      if loss_params["MeanSquaredErrorLoss"]["Weights"]:
        weights = loss_params["MeanSquaredErrorLoss"]["Weights"]
      else:
        weights = None
      self.selected_loss = LossMSE(weights)

    else:
      logging.warning("Loss not specified. Using MSE.")
      self.selected_loss = LossMSE()

  def convert_values(self, raw_values):
    """
    Convert a tensor of size (batch size, 3 * number of actions) to a
    dictionary with an entry for each of the 3 value.
    """
    num_actions = raw_values.shape[1] // 3
    return {
        "Return": raw_values[:, 0:num_actions],
        "Envelope": raw_values[:, num_actions:2*num_actions],
        "Collision": raw_values[:, 2*num_actions:]
    }
