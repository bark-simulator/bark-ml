import unittest
import gym


# BARK imports
from bark_project.modules.runtime.commons.parameters import ParameterServer

# BARK-ML imports
from bark_ml.library_wrappers.lib_tf2rl.runners.gail_runner import GAILRunner 

class PyGAILRunnerTests(unittest.TestCase):

    def test_initialization(self):
        """tests the __init__() method of the GAILRunner class."""
        env_name = "Pendulum-v0"
        env = gym.make(env_name)

        params = ParameterServer(filename="bark_ml/tests/gail_data/params/gail_params.json")
        #runner = GAILRunner(params=params)


if __name__ == '__main__':
    unittest.main()