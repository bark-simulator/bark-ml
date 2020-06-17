import os
from pathlib import Path

import gym
from absl import app
from absl import flags

# BARK imports
from bark_project.modules.runtime.commons.parameters import ParameterServer
from modules.runtime.viewer.matplotlib_viewer import MPViewer
from modules.runtime.viewer.video_renderer import VideoRenderer

# BARK-ML imports
from bark_ml.environments.blueprints import ContinuousHighwayBlueprint, \
  ContinuousMergingBlueprint, ContinuousIntersectionBlueprint
from bark_ml.environments.single_agent_runtime import SingleAgentRuntime
from bark_ml.library_wrappers.lib_tf2rl.agents.gail_agent import BehaviorGAILAgent
from bark_ml.library_wrappers.lib_tf2rl.runners.gail_runner import GAILRunner


FLAGS = flags.FLAGS
flags.DEFINE_enum("mode",
                  "visualize",
                  ["train", "visualize", "evaluate"],
                  "Mode the configuration should be executed in.")

flags.DEFINE_string("train_out",
                  help="The absolute path to where the checkpoints and summaries are saved during training.",
                  # default=os.path.join(Path.home(), ".bark-ml/gail")
                  default=os.path.join(Path.home(), "gail_data")
                  )

flags.DEFINE_string("test_env",
                  help="Example environment in accord with tf2rl to test our code.",
                  default="Pendulum-v0"
                  )

flags.DEFINE_string("gpu",
                  help="-1 for CPU, 0 for GPU",
                  default=0
                  )


class PyTrainingBARKTests(unittest.TestCase):
    def test_training(self):
        """tests the Train() method of the GAILRunner class with a bark gym environment."""

        params = ParameterServer(filename="bark_ml/tests/gail_data/params/gail_params_bark.json")

        # changing the logging directories if not the default is used. (Which would be the same as it is in the json file.)
        params["ML"]["GAILRunner"]["tf2rl"]["logdir"] = os.path.join(FLAGS.train_out, "logs", "bark")
        params["ML"]["GAILRunner"]["tf2rl"]["model_dir"] = os.path.join(FLAGS.train_out, "models", "bark")

        params["World"]["remove_agents_out_of_map"] = True
        params["ML"]["Settings"]["GPUUse"] = FLAGS.gpu

        if len(os.listdir(params["ML"]["GAILRunner"]["tf2rl"]["expert_path_dir"])) == 0:
            print("No expert trajectories found, plaese generate demonstrations first")
            exit()

        # create environment
        bp = ContinuousMergingBlueprint(params,
                                        number_of_senarios=500,
                                        random_seed=0)
        env = SingleAgentRuntime(blueprint=bp,
                                render=False)
        
        # create agent and runner:
        agent = BehaviorGAILAgent(
            environment=env,
            params=params
        )
        env.ml_behavior = agent
        runner = GAILRunner(
            environment=env,
            agent=agent,
            params=params)

        if FLAGS.mode == "train":
            runner.Train()
        elif FLAGS.mode == "visualize":
            runner.Visualize(5)


if __name__ == '__main__':
    unittest.main()