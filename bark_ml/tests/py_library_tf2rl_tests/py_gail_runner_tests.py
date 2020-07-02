import unittest
import gym
import os

# BARK imports
from bark_project.bark.runtime.commons.parameters import ParameterServer

# BARK-ML imports
from bark_ml.library_wrappers.lib_tf2rl.runners.gail_runner import GAILRunner 

# TF2RL imports
from tf2rl.algos.ddpg import DDPG
from tf2rl.algos.gail import GAIL
from tf2rl.experiments.utils import restore_latest_n_traj

class sample_agent():
    """dummy class just to test the runner. Has got the same values as a normal GAIL agent."""
    def __init__(self, generator, discriminator):
        """initialize tht sample agent."""
        self.generator = generator
        self.discriminator = discriminator


def dir_check(params):
    """checks some dirs whether they are present or not.
    If they are not, the function creates them.
    """
    for key in ['logdir', 'model_dir', 'expert_path_dir']:
        if not os.path.exists(params["ML"]["GAILRunner"]["tf2rl"][key]):
            os.makedirs(params["ML"]["GAILRunner"]["tf2rl"][key])


class PyGAILRunnerTests(unittest.TestCase):
    # TODO docstring
    
    def setUp(self):
        """
        setup
        """
        self.params = ParameterServer(filename=os.path.join(os.path.dirname(__file__), "gail_data/params/gail_params_open-ai.json"))

        # creating the dirs for logging if they are not present already:
        dir_check(self.params)

        if len(os.listdir(self.params["ML"]["GAILRunner"]["tf2rl"]["expert_path_dir"])) == 0:
            print("No expert trajectories found, plaese generate demonstrations first")
            print("python tf2rl/examples/run_sac.py --env-name=Pendulum-v0 --save-test-path --test-interval=50000")
            print("After that, save expert trajectories into bark_ml/tests/gail_data/expert_data/open-ai")
            exit()

        # creating environment:
        env_name = "Pendulum-v0"
        self.env = gym.make(env_name) 

        # getting the expert trajectories from the .pkl file:
        self.expert_trajs = restore_latest_n_traj(dirname=self.params["ML"]["GAILRunner"]["tf2rl"]["expert_path_dir"],
                                            max_steps=self.params["ML"]["GAILRunner"]["tf2rl"]["max_steps"])

        units = [400, 300]

        # creating actor and critic networks:
        self.generator = DDPG(
            state_shape=self.env.observation_space.shape,
            action_dim=self.env.action_space.high.size,
            max_action=self.env.action_space.high[0],
            gpu=self.params["ML"]["Settings"]["GPUUse", "", 0],
            actor_units=units,
            critic_units=units,
            n_warmup=100,
            batch_size=16)
        self.discriminator = GAIL(
            state_shape=self.env.observation_space.shape,
            action_dim=self.env.action_space.high.size,
            units=units,
            enable_sn=False,
            batch_size=8,
            gpu=self.params["ML"]["Settings"]["GPUUse", "", 0]
            )

        # creating sample agent:
        self.agent = sample_agent(self.generator, self.discriminator)
        
        # create runner:
        self.runner = GAILRunner(
            environment=self.env,
            agent=self.agent,
            params=self.params,
            expert_trajs=self.expert_trajs)

    
    def test_get_trainer(self):
        """get_trainer
        """
        trainer = self.runner.GetTrainer()
        self.assertEqual(trainer._policy.actor_units, [400, 300])

        self.assertEqual(trainer._expert_obs, self.expert_trajs['obses'])

if __name__ == '__main__':
    unittest.main()