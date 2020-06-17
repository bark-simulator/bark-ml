import unittest
import gym
import os

# BARK imports
from bark_project.modules.runtime.commons.parameters import ParameterServer

# BARK-ML imports
from bark_ml.library_wrappers.lib_tf2rl.runners.gail_runner import GAILRunner 

# TF2RL imports
from tf2rl.algos.ddpg import DDPG
from tf2rl.algos.gail import GAIL

class sample_agent():
    def __init__(self, generator, discriminator):
        self._generator = generator
        self._discriminator = discriminator


class PyGAILRunnerTests(unittest.TestCase):

    def test_initialization(self):
        """tests the __init__() method of the GAILRunner class."""

        params = ParameterServer(filename="bark_ml/tests/gail_data/params/gail_params.json")

        if len(os.listdir(params["ML"]["GAILRunner"]["tf2rl"]["expert_path_dir"])) == 0:
            print("No expert trajectories found, plaese generate demonstrations first")
            print("python tf2rl/examples/run_sac.py --env-name=Pendulum-v0 --save-test-path --test-interval=50000")
            print("After that, save expert trajectories into bark_ml/tests/gail_data/expert_data")
            exit()

        env_name = "Pendulum-v0"
        env = gym.make(env_name) 

        units = [400, 300]

        generator = DDPG(
            state_shape=env.observation_space.shape,
            action_dim=env.action_space.high.size,
            max_action=env.action_space.high[0],
            gpu=params["ML"]["GAILRunner"]["tf2rl"]["gpu"],
            actor_units=units,
            critic_units=units,
            n_warmup=100,
            batch_size=16)
        discriminator = GAIL(
            state_shape=env.observation_space.shape,
            action_dim=env.action_space.high.size,
            units=units,
            enable_sn=False,
            batch_size=8,
            gpu=params["ML"]["GAILRunner"]["tf2rl"]["gpu"]
            )

        agent = sample_agent(generator, discriminator)
        
        runner = GAILRunner(
            environment=env,
            agent=agent,
            params=params)


if __name__ == '__main__':
    unittest.main()