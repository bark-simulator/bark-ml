from abc import ABC, abstractmethod
import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.memory import LazyMultiStepMemory, \
    LazyPrioritizedMultiStepMemory
from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.utils import RunningMeanStats, LinearAnneaer


class BaseAgent(ABC):

    def __init__(self, env, test_env, params):
        self._params = params
        self.env = env
        self.test_env = test_env

        torch.manual_seed(self._params["ML"]["BaseAgent"]["Seed", "", 0])
        np.random.seed(self._params["ML"]["BaseAgent"]["Seed", "", 0])
        self.env.seed(self._params["ML"]["BaseAgent"]["Seed", "", 0])
        self.test_env.seed(2**31-1-self._params["ML"]["BaseAgent"]["Seed", "", 0])
        # torch.backends.cudnn.deterministic = True  # It harms a performance.
        # torch.backends.cudnn.benchmark = False  # It harms a performance.

        self.device = torch.device(
            "cuda" if self._params["ML"]["BaseAgent"]["Cuda", "", True] and torch.cuda.is_available() else "cpu")

        self.online_net = None
        self.target_net = None

        self.steps = 0
        self.learning_steps = 0
        self.episodes = 0
        self.best_eval_score = -np.inf
        self.num_actions = self.env.action_space.n
        self.num_steps = self._params["ML"]["BaseAgent"]["NumSteps", "", 5000000]
        self.batch_size = self._params["ML"]["BaseAgent"]["BatchSize", "", 32]

        self.double_q_learning = self._params["ML"]["BaseAgent"]["Double_q_learning", "", False]
        self.dueling_net = self._params["ML"]["BaseAgent"]["DuelingNet", "", False]
        self.noisy_net = self._params["ML"]["BaseAgent"]["NoisyNet", "", False]
        self.use_per = self._params["ML"]["BaseAgent"]["Use_per", "", False]

        self.log_interval = self._params["ML"]["BaseAgent"]["LogInterval", "", 100]
        self.eval_interval = self._params["ML"]["BaseAgent"]["EvalInterval", "", 25000]
        self.num_eval_steps = self._params["ML"]["BaseAgent"]["NumEvalSteps", "", 12500]
        self.gamma_n = self._params["ML"]["BaseAgent"]["Gamma", "",
                                                       0.99] ** self._params["ML"]["BaseAgent"]["Multi_step", "", 1]
        self.start_steps = self._params["ML"]["BaseAgent"]["StartSteps", "", 5000]
        self.epsilon_train = LinearAnneaer(
            1.0, self._params["ML"]["BaseAgent"]["EpsilonTrain", "", 0.01], self._params["ML"]["BaseAgent"]["EpsilonDecaySteps", "", 25000])
        self.epsilon_eval = self._params["ML"]["BaseAgent"]["EpsilonEval", "", 0.001]
        self.update_interval = self._params["ML"]["BaseAgent"]["Update_interval", "", 4]
        self.target_update_interval = self._params["ML"]["BaseAgent"]["TargetUpdateInterval", "", 5000]
        self.max_episode_steps = self._params["ML"]["BaseAgent"]["MaxEpisodeSteps", "", 10000]
        self.grad_cliping = self._params["ML"]["BaseAgent"]["GradCliping", "", 5.0]

        self.summary_dir = self._params["ML"]["BaseAgent"]["SummaryPath", "", ""]
        self.model_dir = self._params["ML"]["BaseAgent"]["CheckpointPath", "", ""]
        
        if not os.path.exists(self.model_dir) and self.model_dir:
            os.makedirs(self.model_dir)
        if not os.path.exists(self.summary_dir) and self.summary_dir:
            os.makedirs(self.summary_dir)

        self.writer = SummaryWriter(log_dir=self.summary_dir)
        self.train_return = RunningMeanStats(self.log_interval)

        # Replay memory which is memory-efficient to store stacked frames.
        if self.use_per:
            beta_steps = (self.num_steps - self.start_steps) / \
                self.update_interval
            self.memory = LazyPrioritizedMultiStepMemory(
                self._params["ML"]["BaseAgent"]["MemorySize",
                                                "", 10**6], self.env.observation_space.shape,
                self.device, self._params["ML"]["BaseAgent"]["Gamma", "", 0.99], self._params["ML"]["BaseAgent"]["Multi_step", "", 1], beta_steps=beta_steps)
        else:
            self.memory = LazyMultiStepMemory(
                self._params["ML"]["BaseAgent"]["MemorySize",
                                                "", 10**6], self.env.observation_space.shape,
                self.device, self._params["ML"]["BaseAgent"]["Gamma", "", 0.99], self._params["ML"]["BaseAgent"]["Multi_step", "", 1])

    def run(self):
        while True:
            self.train_episode()
            if self.steps > self.num_steps:
                break

    def is_update(self):
        return self.steps % self.update_interval == 0\
            and self.steps >= self.start_steps

    def is_random(self, eval=False):
        # Use e-greedy for evaluation.
        if self.steps < self.start_steps:
            return True
        if eval:
            return np.random.rand() < self.epsilon_eval
        if self.noisy_net:
            return False
        return np.random.rand() < self.epsilon_train.get()

    def update_target(self):
        self.target_net.load_state_dict(
            self.online_net.state_dict())

    def explore(self):
        # Act with randomness.
        action = self.env.action_space.sample()
        return action

    def exploit(self, state):
        # Act without randomness.
        #state = torch.Tensor(state).unsqueeze(0).to(self.device).float() 
        actions = self.calculate_actions(state).argmax().item()
        return actions

    def calculate_actions(self, state):
        # Act without randomness.
        state = torch.Tensor(state).unsqueeze(0).to(self.device).float() 
        with torch.no_grad():
            actions = self.online_net(states=state)
        return actions

    @abstractmethod
    def learn(self):
        pass

    def save_models(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(
            self.online_net.state_dict(),
            os.path.join(save_dir, 'online_net.pth'))
        torch.save(
            self.target_net.state_dict(),
            os.path.join(save_dir, 'target_net.pth'))
        online_net_script = torch.jit.script(self.online_net)
        online_net_script.save(os.path.join(save_dir, 'online_net_script.pt'))

    def load_models(self, save_dir):
        self.online_net.load_state_dict(torch.load(
            os.path.join(save_dir, 'online_net.pth')))
        self.target_net.load_state_dict(torch.load(
            os.path.join(save_dir, 'target_net.pth')))

    def visualize(self, num_episodes=5):
        for _ in range(0, num_episodes):
            state = self.env.reset()
            done = False
            while (not done):
                action = self.exploit(state)
                next_state, reward, done, _ = self.env.step(action)
                self.env.render()
                state = next_state

    def train_episode(self):
        self.online_net.train()
        self.target_net.train()

        self.episodes += 1
        episode_return = 0.
        episode_steps = 0

        done = False
        state = self.env.reset()

        while (not done) and episode_steps <= self.max_episode_steps:
            # NOTE: Noises can be sampled only after self.learn(). However, I
            # sample noises before every action, which seems to lead better
            # performances.
            self.online_net.sample_noise()

            if self.is_random(eval=False):
                action = self.explore()
            else:
                action = self.exploit(state)

            next_state, reward, done, _ = self.env.step(action)
            if self.episodes % 10000 == 0:
                #self.env.render()
                print("Reward:", reward)

            # To calculate efficiently, I just set priority=max_priority here.
            self.memory.append(state, action, reward, next_state, done)

            self.steps += 1
            episode_steps += 1
            episode_return += reward
            state = next_state

            self.train_step_interval()

        # We log running mean of stats.
        self.train_return.append(episode_return)

        # We log evaluation results along with training frames = 4 * steps.
        if self.episodes % self.log_interval == 0:
            self.writer.add_scalar(
                'return/train', self.train_return.get(), 4 * self.steps)

        print(f'Episode: {self.episodes:<4}  '
              f'episode steps: {episode_steps:<4}  '
              f'return: {episode_return:<5.1f}')

    def train_step_interval(self):
        self.epsilon_train.step()

        if self.steps % self.target_update_interval == 0:
            self.update_target()

        if self.is_update():
            self.learn()

        if self.steps % self.eval_interval == 0:
            self.evaluate()
            self.save_models(os.path.join(self.model_dir, 'final'))
            self.online_net.train()

    def evaluate(self):
        self.online_net.eval()
        num_episodes = 0
        num_steps = 0
        total_return = 0.0

        while True:
            state = self.test_env.reset()
            episode_steps = 0
            episode_return = 0.0
            done = False
            while (not done) and episode_steps <= self.max_episode_steps:
                if self.is_random(eval=True):
                    action = self.explore()
                else:
                    action = self.exploit(state)

                next_state, reward, done, _ = self.test_env.step(action)
                num_steps += 1
                episode_steps += 1
                episode_return += reward
                state = next_state

            num_episodes += 1
            total_return += episode_return

            if num_steps > self.num_eval_steps:
                break

        mean_return = total_return / num_episodes

        if mean_return > self.best_eval_score:
            self.best_eval_score = mean_return
            self.save_models(os.path.join(self.model_dir, 'best'))

        # We log evaluation results along with training frames = 4 * steps.
        self.writer.add_scalar(
            'return/test', mean_return, 4 * self.steps)
        print('-' * 60)
        print(f'Num steps: {self.steps:<5}  '
              f'return: {mean_return:<5.1f}')
        print('-' * 60)

    def __del__(self):
        self.env.close()
        self.test_env.close()
        self.writer.close()
