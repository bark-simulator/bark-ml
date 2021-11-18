import pprint
import os
import hashlib
import logging
from pathlib import Path
import tensorflow as tf
import numpy as np
from bark.runtime.commons.parameters import ParameterServer
from bark_ml.experiment.experiment import Experiment


class ExperimentRunner:
  """The ExperimentRunner-Class provides an easy-to-use interface to
  train, visualize, evaluate, and manage experiments.

  Additionally, it creates an Experiment only from a json that is
  hashes before training. Thus, trained results can be matched to executions
  and evaluations.
  """

  def __init__(self,
               json_file=None,
               params=None,
               mode="visualize",
               random_seed=0):
    self._logger = logging.getLogger()
    self._experiment_json = json_file
    if params is not None:
      self._params = params
    else:
      self._params = ParameterServer(filename=json_file)
    self._experiment_folder, self._json_name = \
      self.GetExperimentsFolder(json_file)
    # set random seeds
    self._random_seed = random_seed
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
    self.SetCkptsAndSummaries()
    self._experiment = self.BuildExperiment(json_file, mode)
    self.Visitor(mode)

  def Visitor(self, mode):
    if mode == "train":
      self._experiment._params.Save(self._runs_folder+"params.json")
      self.Train()
    if mode == "visualize":
      self.Visualize()
    if mode == "evaluate":
      self.Evaluate()
    if mode == "print":
      self.PrintExperiment()
    if mode == "save":
      self.SaveExperiment(FLAGS.save_path)

  def BuildExperiment(self, json_file, mode):
    return Experiment(json_file, self._params, mode)

  @staticmethod
  def GetExperimentsFolder(json_file):
    dir_name = Path(json_file).parent
    if not os.path.isdir(dir_name):
      assert f"{dir_name} does not exist."
    base_name = os.path.basename(json_file)
    file_name = os.path.splitext(base_name)[0]
    return dir_name, file_name

  @staticmethod
  def GenerateHash(params):
    """
    Hash-function to indicate whether the same json is used
    as during training.
    """
    exp_params = params.ConvertToDict()
    return hashlib.sha1(
      repr(sorted(exp_params.items())).encode('utf-8')).hexdigest()

  def CompareHashes(self):
    experiment_hash = self.GenerateHash(self._params)
    if os.path.isfile(self._hash_file_path):
      file = open(self._hash_file_path, 'r')
      old_experiment_hash = file.readline()
      file.close()
      if experiment_hash != old_experiment_hash:
        self._logger.warning("\033[31m Trained experiment hash does not match \033[0m")

  def SetCkptsAndSummaries(self):
    self._runs_folder = \
      str(self._experiment_folder) + "/" + self._json_name + "/" + str(self._random_seed) + "/"
    ckpt_folder = self._runs_folder + "ckpts/"
    summ_folder = self._runs_folder + "summ/"
    self._logger.info(f"Run folder of the agent {self._runs_folder}.")
    self._hash_file_path = self._runs_folder + "hash.txt"
    self._params["ML"]["BehaviorTFAAgents"]["CheckpointPath"] = \
      ckpt_folder
    self._params["ML"]["TFARunner"]["SummaryPath"] = \
      summ_folder

  def Train(self):
    if not os.path.isfile(self._hash_file_path):
      os.makedirs(os.path.dirname(self._hash_file_path), exist_ok=True)
      file = open(self._hash_file_path, 'w')
      file.write(str(self.GenerateHash(self._experiment.params)))
      file.close()
    else:
      self.CompareHashes()
    self._experiment.runner.SetupSummaryWriter()
    self._experiment.runner.Train()

  def Evaluate(self):
    self.CompareHashes()
    num_episodes = \
      self._params["Experiment"]["NumEvaluationEpisodes"]
    return self._experiment.runner.Run(
      num_episodes=num_episodes, render=False, trace_colliding_ids=True)

  def Visualize(self):
    self.CompareHashes()
    num_episodes = \
      self._params["Experiment"]["NumVisualizationEpisodes"]
    self._experiment.runner.Run(num_episodes=num_episodes, render=True)

  def PrintExperiment(self):
    pprint.pprint(self._experiment.params.ConvertToDict())

  def SaveExperiment(self, file_path):
    self._params.Save(file_path)