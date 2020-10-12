import pprint
import os
from pathlib import Path
from absl import app
from absl import flags

from experiments.experiment import Experiment

FLAGS = flags.FLAGS
flags.DEFINE_enum("mode",
                  "visualize",
                  ["train", "visualize", "evaluate"],
                  "Mode the configuration should be executed in.")
flags.DEFINE_string("exp_json",
                    "/Users/hart/Development/bark-ml/experiments/configs/gcn_three_layers.json",
                    "Path to the experiment json.")


class ExperimentRunner:
  def __init__(self, json_file, mode):
    self._experiment_json = json_file
    self._experiment = self.BuildExperiment(json_file)
    self._experiment_folder, self._json_name = \
      self.GetExperimentsFolder(json_file)
    self.SetCkptsAndSummaries()
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
  
  def GetExperimentsFolder(self, json_file):
    dir_name = Path(json_file).parent.parent
    if not os.path.isdir(dir_name):
      assert f"{dir_name} does not exist."
    base_name = os.path.basename(json_file)
    file_name = os.path.splitext(base_name)[0]
    return dir_name, file_name
  
  def GenerateHash(self):
    """hash-function to indicate whether the same json is used
       as during training"""
    ml_params = self._experiment.params.ConvertToDict()
    return hash(frozenset(ml_params.items()))

  def SetCkptsAndSummaries(self):
    runs_folder = \
      str(self._experiment_folder) + "/runs/" + self._json_name + "/"
    ckpt_folder = runs_folder + "ckpts/"
    summ_folder = runs_folder + "summ/"
    self._experiment.params["ML"]["BehaviorTFAAgents"]["CheckpointPath"] = \
      ckpt_folder
    self._experiment.params["ML"]["TFARunner"]["SummaryPath"] = \
      summ_folder

  def Train(self):
    # NOTE: safe params in folder
    self._experiment.runner.SetupSummaryWriter()
    self._experiment.runner.Train()
  
  def Evaluate(self):
    # NOTE: check hash
    num_episodes = \
      self._experiment.params["Experiment"]["NumEvaluationEpisodes"]
    self._experiment.runner.Run(num_episodes=num_episodes, render=False)
  
  def Visualize(self):
    # NOTE: check hash
    num_episodes = \
      self._experiment.params["Experiment"]["NumVisualizationEpisodes"]
    self._experiment.runner.Run(num_episodes=num_episodes, render=True)
    
  def SerializeExperiment(self):
    return self._experiment.params.ParamsToDict()


# run experiment
def run_experiment(argv):
  # experiment json, save path, mode
  exp_runner = ExperimentRunner(
    json_file=FLAGS.exp_json,
    mode=FLAGS.mode)

if __name__ == '__main__':
  app.run(run_experiment)    