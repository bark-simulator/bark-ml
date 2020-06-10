import unittest
import gym
import os

# BARK imports
from bark_project.modules.runtime.commons.parameters import ParameterServer

# BARK-ML imports
from bark_ml.library_wrappers.lib_tf2rl.runners.gail_runner import GAILRunner 

class PyGAILRunnerTests(unittest.TestCase):

    def test_initialization(self):
        """tests the __init__() method of the GAILRunner class."""

        params = ParameterServer(filename="bark_ml/tests/gail_data/params/gail_params.json")

        if len(os.listdir(params["ML"]["GAILRunner"]["tf2rl"]["expert_path_dir"])) == 0:
            print("Plaese generate demonstrations first")
            print("python examples/run_sac.py --env-name=Pendulum-v0 --save-test-path --test-interval=50000")
            exit()

        env_name = "Pendulum-v0"
        env = gym.make(env_name)

        """policy = DDPG(
            state_shape=env.observation_space.shape,
            action_dim=env.action_space.high.size,
            max_action=env.action_space.high[0],
            gpu=args.gpu,
            actor_units=units,
            critic_units=units,
            n_warmup=10000,
            batch_size=100)
        irl = GAIL(
            state_shape=env.observation_space.shape,
            action_dim=env.action_space.high.size,
            units=units,
            enable_sn=args.enable_sn,
            batch_size=32,
            gpu=args.gpu
            )"""

        #runner = GAILRunner(params=params)


if __name__ == '__main__':
    unittest.main()