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
from tf2rl.experiments.utils import restore_latest_n_traj

class sample_agent():
    def __init__(self, generator, discriminator):
        self._generator = generator
        self._discriminator = discriminator


class PyGAILRunnerTests(unittest.TestCase):

    def test_initialization(self):
        """tests the __init__() method of the GAILRunner class."""

        params = ParameterServer(filename="bark_ml/tests/gail_data/params/gail_params_open-ai.json")

        if len(os.listdir(params["ML"]["GAILRunner"]["tf2rl"]["expert_path_dir"])) == 0:
            print("No expert trajectories found, plaese generate demonstrations first")
            print("python tf2rl/examples/run_sac.py --env-name=Pendulum-v0 --save-test-path --test-interval=50000")
            print("After that, save expert trajectories into bark_ml/tests/gail_data/expert_data/open-ai")
            exit()

        # creating environment:
        env_name = "Pendulum-v0"
        env = gym.make(env_name) 

        # getting the expert trajectories from the .pkl file:
        expert_trajs = restore_latest_n_traj(dirname=params["ML"]["GAILRunner"]["tf2rl"]["expert_path_dir"],
                                            max_steps=params["ML"]["GAILRunner"]["tf2rl"]["max_steps"])

        units = [400, 300]

        # creating actor and critic networks:
        generator = DDPG(
            state_shape=env.observation_space.shape,
            action_dim=env.action_space.high.size,
            max_action=env.action_space.high[0],
            gpu=params["ML"]["Settings"]["GPUUse", "", 0],
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
            gpu=params["ML"]["Settings"]["GPUUse", "", 0]
            )

        # creating sample agent:
        agent = sample_agent(generator, discriminator)
        
        # create runner:
        runner = GAILRunner(
            environment=env,
            agent=agent,
            params=params,
            expert_trajs=expert_trajs)


if __name__ == '__main__':
    unittest.main()