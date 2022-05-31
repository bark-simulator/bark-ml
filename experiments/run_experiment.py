from absl import app
from absl import flags

from bark_ml.experiment.experiment_runner import ExperimentRunner

FLAGS = flags.FLAGS
flags.DEFINE_enum("mode",
                  "visualize",
                  ["train", "visualize", "evaluate", "print", "save", "collisions", "validate"],
                  "Mode the configuration should be executed in.")
# NOTE: absolute paths are required
flags.DEFINE_string("exp_json",
                    "/home/xliu/bark-ml/experiments/configs/rules/single_lane_ref.json",
                    "Path to the experiment json.")
flags.DEFINE_string("save_path",
                    "/home/xliu/bark-ml/experiments/configs/new_exp.json",
                    "Path to the experiment json.")
flags.DEFINE_integer("random_seed",
                     0,
                     "Random seed to be used.")


# run experiment
def run_experiment(argv):
  exp_runner = ExperimentRunner(
    json_file=FLAGS.exp_json,
    mode=FLAGS.mode,
    random_seed=FLAGS.random_seed)

if __name__ == '__main__':
  app.run(run_experiment)