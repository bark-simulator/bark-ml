# Copyright (c) 2020 fortiss GmbH
#
# Authors: Patrick Hart, Julian Bernhard, Klemens Esterle,
# Tobias Kessler and Mansoor Nasir
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# The code is adapted from opensource implementation - https://github.com/ku2482/fqf-iqn-qrdqn.pytorch
# MIT License - Copyright (c) 2020 Toshiki Watanabe

import os
import logging
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np

# BARK-ML imports
from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.utils \
  import RunningMeanStats, LinearAnneaer
from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.memory \
  import LazyMultiStepMemory, LazyPrioritizedMultiStepMemory
from bark_ml.behaviors.discrete_behavior import BehaviorDiscreteMacroActionsML

# BARK imports
from bark.core.models.behavior import BehaviorModel


class BaseAgent(BehaviorModel):

  def __init__(self, env, test_env, params, bark_behavior=None):
    BehaviorModel.__init__(self, params)
    self._params = params
    self._env = env
    self._test_env = test_env
    self._bark_behavior_model = bark_behavior or BehaviorDiscreteMacroActionsML(
        params)

    self._device = torch.device("cuda" if self._params["ML"]["BaseAgent"][
        "Cuda", "", True] and torch.cuda.is_available() else "cpu")

    self._online_net = None
    self._target_net = None

    self._steps = 0
    self._learning_steps = 0
    self._episodes = 0
    self._best_eval_score = -np.inf
    self._num_actions = self._env.action_space.n
    self._num_steps = self._params["ML"]["BaseAgent"]["NumSteps", "", 5000000]
    self._batch_size = self._params["ML"]["BaseAgent"]["BatchSize", "", 32]

    self._double_q_learning = self._params["ML"]["BaseAgent"][
        "Double_q_learning", "", False]
    self._noisy_net = self._params["ML"]["BaseAgent"]["NoisyNet", "", False]
    self._use_per = self._params["ML"]["BaseAgent"]["Use_per", "", False]

    self._reward_log_interval = self._params["ML"]["BaseAgent"][
        "RewardLogInterval", "", 5]
    self._summary_log_interval = self._params["ML"]["BaseAgent"][
        "SummaryLogInterval", "", 100]
    self._eval_interval = self._params["ML"]["BaseAgent"]["EvalInterval", "",
                                                          25000]
    self._num_eval_steps = self._params["ML"]["BaseAgent"]["NumEvalSteps", "",
                                                           12500]
    self._gamma_n = \
     self._params["ML"]["BaseAgent"]["Gamma", "", 0.99] ** \
     self._params["ML"]["BaseAgent"]["Multi_step", "", 1]

    self._start_steps = self._params["ML"]["BaseAgent"]["StartSteps", "", 5000]
    self._epsilon_train = LinearAnneaer(
        1.0, self._params["ML"]["BaseAgent"]["EpsilonTrain", "", 0.01],
        self._params["ML"]["BaseAgent"]["EpsilonDecaySteps", "", 25000])
    self._epsilon_eval = self._params["ML"]["BaseAgent"]["EpsilonEval", "",
                                                         0.001]
    self._update_interval = \
     self._params["ML"]["BaseAgent"]["Update_interval", "", 4]
    self._target_update_interval = self._params["ML"]["BaseAgent"][
        "TargetUpdateInterval", "", 5000]
    self._max_episode_steps = \
     self._params["ML"]["BaseAgent"]["MaxEpisodeSteps",  "", 10000]
    self._grad_cliping = self._params["ML"]["BaseAgent"]["GradCliping", "", 5.0]

    self._summary_dir = \
     self._params["ML"]["BaseAgent"]["SummaryPath", "", ""]
    self._model_dir = \
     self._params["ML"]["BaseAgent"]["CheckpointPath", "", ""]

    if not os.path.exists(self._model_dir) and self._model_dir:
      os.makedirs(self._model_dir)
    if not os.path.exists(self._summary_dir) and self._summary_dir:
      os.makedirs(self._summary_dir)

    self._writer = SummaryWriter(log_dir=self._summary_dir)
    self._train_return = RunningMeanStats(self._summary_log_interval)

    # NOTE: by default we do not want the action to be set externally
    #       as this enables the agents to be plug and played in BARK.
    self._set_action_externally = False

    # Replay memory which is memory-efficient to store stacked frames.
    if self._use_per:
      beta_steps = (self._num_steps - self._start_steps) / \
             self._update_interval
      self._memory = LazyPrioritizedMultiStepMemory(
          self._params["ML"]["BaseAgent"]["MemorySize", "", 10**6],
          self._env.observation_space.shape,
          self._device,
          self._params["ML"]["BaseAgent"]["Gamma", "", 0.99],
          self._params["ML"]["BaseAgent"]["Multi_step", "", 1],
          beta_steps=beta_steps)
    else:
      self._memory = LazyMultiStepMemory(
          self._params["ML"]["BaseAgent"]["MemorySize", "", 10**6],
          self._env.observation_space.shape, self._device,
          self._params["ML"]["BaseAgent"]["Gamma", "", 0.99],
          self._params["ML"]["BaseAgent"]["Multi_step", "", 1])

  def train(self):
    while True:
      self.train_episode()
      if self._steps > self._num_steps:
        break
    self._set_action_externally = True

  def is_update(self):
    return self._steps % self._update_interval == 0 \
        and self._steps >= self._start_steps

  def is_random(self, eval=False):
    # Use e-greedy for evaluation.
    if self._steps < self._start_steps:
      return True
    if eval:
      return np.random.rand() < self._epsilon_eval
    if self._noisy_net:
      return False
    return np.random.rand() < self._epsilon_train.get()

  def update_target(self):
    self._target_net.load_state_dict(self._online_net.state_dict())

  def explore(self):
    # Act with randomness.
    action = self._env.action_space.sample()
    return action

  @property
  def set_action_externally(self):
    return self._set_action_externally

  @set_action_externally.setter
  def set_action_externally(self, externally):
    self._set_action_externally = externally

  def ActionToBehavior(self, action):
    # NOTE: will either be set externally or internally
    self._action = action

  def Act(self, state):
    # Act without randomness.
    # state = torch.Tensor(state).unsqueeze(0).to(self._device).float()
    actions = self.calculate_actions(state).argmax().item()
    return actions

  def calculate_actions(self, state):
    # Act without randomness.
    state = torch.Tensor(state).unsqueeze(0).to(self._device).float()
    with torch.no_grad():
      actions = self._online_net(states=state)  # pylint: disable=not-callable
    return actions

  def Plan(self, dt, observed_world):
    # NOTE: if training is enabled the action is set externally
    if not self._set_action_externally:
      observed_state = self._env._observer.Observe(observed_world)
      action = self.Act(observed_state)
      self._action = action

    action = self._action
    # set action to be executed
    self._bark_behavior_model.ActionToBehavior(action)
    trajectory = self._bark_behavior_model.Plan(dt, observed_world)
    # NOTE: BARK requires models to have trajectories of the past
    BehaviorModel.SetLastTrajectory(self, trajectory)
    return trajectory

  def learn(self):
    pass

  def Clone(self):
    return self

  @property
  def action_space(self):
    return self._bark_behavior_model.action_space

  def save_models(self, save_dir):
    if not os.path.exists(save_dir):
      os.makedirs(save_dir)
    torch.save(self._online_net.state_dict(),
               os.path.join(save_dir, 'online_net.pth'))
    torch.save(self._target_net.state_dict(),
               os.path.join(save_dir, 'target_net.pth'))
    online_net_script = torch.jit.script(self._online_net)
    online_net_script.save(os.path.join(save_dir, 'online_net_script.pt'))

  def load_models(self, save_dir):
    self._online_net.load_state_dict(
        torch.load(os.path.join(save_dir, 'online_net.pth')))
    self._target_net.load_state_dict(
        torch.load(os.path.join(save_dir, 'target_net.pth')))

  def visualize(self, num_episodes=5):
    for _ in range(0, num_episodes):
      state = self._env.reset()
      done = False
      while (not done):
        action = self.Act(state)
        next_state, reward, done, _ = self._env.step(action)
        self._env.render()
        state = next_state

  def train_episode(self):
    self._online_net.train()
    self._target_net.train()

    self._episodes += 1
    episode_return = 0.
    episode_steps = 0

    done = False
    state = self._env.reset()

    while (not done) and episode_steps <= self._max_episode_steps:
      # NOTE: Noises can be sampled only after self._learn(). However, I
      # sample noises before every action, which seems to lead better
      # performances.
      self._online_net.sample_noise()

      if self.is_random(eval=False):
        action = self.explore()
      else:
        action = self.Act(state)

      next_state, reward, done, _ = self._env.step(action)
      if self._episodes % self._reward_log_interval == 0:
        # self._env.render()
        logging.info(f"Reward: {reward:<4}")

      # To calculate efficiently, I just set priority=max_priority here.
      self._memory.append(state, action, reward, next_state, done)

      self._steps += 1
      episode_steps += 1
      episode_return += reward
      state = next_state

      self.train_step_interval()

    # We log running mean of stats.
    self._train_return.append(episode_return)

    # We log evaluation results along with training frames = 4 * steps.
    if self._episodes % self._summary_log_interval == 0:
      self._writer.add_scalar('return/train', self._train_return.get(),
                              4 * self._steps)

    logging.info(f'Episode: {self._episodes:<4}  '
                 f'episode steps: {episode_steps:<4}  '
                 f'return: {episode_return:<5.1f}')

  def train_step_interval(self):
    self._epsilon_train.step()

    if self._steps % self._target_update_interval == 0:
      self.update_target()

    if self.is_update():
      self.learn()

    if self._steps % self._eval_interval == 0:
      self.evaluate()
      self.save_models(os.path.join(self._model_dir, 'final'))
      self._online_net.train()

  def evaluate(self):
    self._online_net.eval()
    num_episodes = 0
    num_steps = 0
    total_return = 0.0

    while True:
      state = self._test_env.reset()
      episode_steps = 0
      episode_return = 0.0
      done = False
      while (not done) and episode_steps <= self._max_episode_steps:
        if self.is_random(eval=True):
          action = self.explore()
        else:
          action = self.Act(state)

        next_state, reward, done, _ = self._test_env.step(action)
        num_steps += 1
        episode_steps += 1
        episode_return += reward
        state = next_state

      num_episodes += 1
      total_return += episode_return

      if num_steps > self._num_eval_steps:
        break

    mean_return = total_return / num_episodes

    if mean_return > self._best_eval_score:
      self._best_eval_score = mean_return
      self.save_models(os.path.join(self._model_dir, 'best'))

    # We log evaluation results along with training frames = 4 * steps.
    self._writer.add_scalar('return/test', mean_return, 4 * self._steps)
    logging.info('-' * 60)
    logging.info(f'Num steps: {self._steps:<5}  '
                 f'return: {mean_return:<5.1f}')
    logging.info('-' * 60)

  def __del__(self):
    self._writer.close()
