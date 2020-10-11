import pprint
from absl import app
from absl import flags

from experiments.experiment import Experiment

FLAGS = flags.FLAGS
flags.DEFINE_enum("mode",
                  "visualize",
                  ["train", "visualize", "evaluate"],
                  "Mode the configuration should be executed in.")
flags.DEFINE_string("exp_path",
                    "/Users/hart/Development/bark-ml/experiments/data/config.json",
                    "Path to the experiment json.")


class ExperimentRunner:
  def __init__(self, json_file, mode):
    self._experiment_json = json_file
    self._experiment = self.BuildExperiment(json_file)
    self.Visitor(mode)

  def Visitor(self, mode):
    if mode == "train":
      self.Train()
    if mode == "visualize":
      self.Visualize()
    if mode == "evaluate":
      self.Evaluate()
  
  def BuildExperiment(self, json_file):
    return Experiment(json_file)
  
  def GenerateHash(self):
    """hash-function to indicate whether the same json is used
       as during training"""
    ml_params = self._experiment.params.ConvertToDict()
    return hash(frozenset(ml_params.items()))

  def Train(self):
    # save json + hash in training folder
    self._experiment.params["ML"]["BehaviorTFAAgents"]["CheckpointPath"] = \
      "/Users/hart/Development/bark-ml/checkpoints_merge_spektral_att3/"
    self._experiment.params["ML"]["TFARunner"]["SummaryPath"] = \
      "/Users/hart/Development/bark-ml/checkpoints_merge_spektral_att3/"
    self._experiment.runner.SetupSummaryWriter()
    self._experiment.runner.Train()
  
  def Evaluate(self):
    # check hash
    self._experiment.runner.Run(num_episodes=5, render=False)
  
  def Visualize(self):
    # check hash
    self._experiment.runner.Run(num_episodes=5, render=True)
    
  def SerializeExperiment(self):
    return self._experiment.params.ParamsToDict()

# run experiment
def run_experiment(argv):
  # experiment json, save path, mode
  exp_runner = ExperimentRunner(
    json_file=FLAGS.exp_path,
    mode=FLAGS.mode)

if __name__ == '__main__':
  app.run(run_experiment)    