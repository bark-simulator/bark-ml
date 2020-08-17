# Copyright (c) 2020 fortiss GmbH
#
# Authors: Silvan Wimmer
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import os
import pickle
import logging
import numpy as np
import tensorflow as tf

# Supervised learning specific imports
from bark_ml.tests.capability_gnn_actor.data_generation import DataGenerator

class SupervisedData:
  """Handler for SupervisedData

  This class coordinates the data handling. Including the generation, loading,
  transformation and storing of the supervised data. It eases the usage of
  the data.

  Args:
    observer: bark-ml Observer instance (with which the data should be generated
              or with which the data was originally loaded).
    params: bark ParameterServer instance (specifys parameters for data
            generation).
    batch_size: int, specifys batch size of tf.data.Datasets.
    train_split: float, specifys ratio of training data [should be between 0
                 and 1].
    num_scenarios: int, number of scenario runs for data generation.
  """
  def __init__(self,
               observer,
               params,
               data_path=None,
               batch_size=32,
               train_split=0.8,
               num_scenarios=100):
    """Takes care of all the data handling. In the end there are the two
    datasets of class tf.data.Dataset:

    self._train_dataset which should be used for training and
    self._test_dataset which should be used for testing/validation.
    """
    self._observer = observer
    self._params = params
    self._data_path = data_path
    self._batch_size = batch_size
    self._train_split = train_split
    self._num_scenarios = num_scenarios
    
    # Get datasets for training and test(validation)
    try:
      # Check if data_path is specified, and data is already generated
      assert self._data_path is not None
      scenarios = os.listdir(self._data_path)

      # Get number of scenario files (exclude e.g. BUILD file)
      n_scenarios = sum(1 for file_ in scenarios if file_.find(".pickle")>0)
      assert n_scenarios==self._num_scenarios
      
      logging.debug("Data is already generated")
      # Load data
      data_collection = self._load(self._data_path)

    except:
      # Data needs to be generated first
      logging.debug("Starting data_generation")
      data_collection = self._generate_data(data_path=self._data_path,
                                            num_scenarios=self._num_scenarios)
    
    # Transform to supervised dataset
    X, Y = self._transform_to_supervised_dataset(data_collection,
                                                 observer=self._observer)
    # Transform to tf_datsets
    train_dataset, test_dataset = self._transform_to_tf_datasets(X, Y,
                  train_split=self._train_split, batch_size=self._batch_size)
    self._train_dataset = train_dataset
    self._test_dataset = test_dataset

  def _load(self, data_path):
    """Load dataset from specified path.

    Args:
      data_path: path, specifies location of dataset.

    Returns:
      data_collection: list of sceanrio_data (with scenario_data being
                       a list of datapoints (datapoint is a dict with
                       keys "observation" and "label").
    """
    data_collection = list()
    # Get all individual scenario names
    scenarios = os.listdir(data_path)
    for scenario in scenarios:
      scenario_path = data_path + "/" + scenario
      with open(scenario_path, 'rb') as f:
        data = pickle.load(f)

      data_collection.append(data)

    return data_collection

  def _transform_to_tf_datasets(self, X, Y,train_split=0.8, batch_size=32):
    """Transforms supervised dataset to batched tensorflow test and train
    datasets.

    Args:
      X: numpy array of features (shape: Nxobservation_size).
      Y: numpy array of labels (shape: Nx2).
      train_split: float between 0 and 1, specifys the ratio of data
                   for training (1-train_split -> ratio of test data).
      batch_size: int, defining batch size of datasets.

    Returns:
      train_dataset: tf.data.Dataset for training.
      test_dataset: tf.data.Dataset for testing/validation.
    """
    X = tf.constant(X)
    Y = tf.constant(Y, dtype=tf.float32)
    dataset_size = X.shape[0]
    # Use tf.data.Dataset for later speedup of training
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.shuffle(dataset_size, seed=5)

    # Train/Test split
    train_size = int(train_split * dataset_size)
    train_dataset = dataset.take(train_size).batch(batch_size)
    test_dataset = dataset.skip(train_size)
    test_dataset = test_dataset.take(-1).batch(batch_size)
    return train_dataset, test_dataset

  def _generate_data(self, data_path, num_scenarios):
    """Generates a supervised dataset with the DataGenerator class.

    Args:
      data_path: path, specifys where the data will be stored (if None
                 the data will not be saved additionally but just returned).
      num_scenarios: int, specifys the number of scenario runs to generate
                     data for.
    Returns:
      data_collection: list of sceanrio_data (with scenario_data being
                       a list of datapoints (datapoint is a dict with
                       keys "observation" and "label").
    """
    logging.info("Starting data_generation")
    data_generator = DataGenerator(num_scenarios=num_scenarios,
                                    dump_dir=data_path, render=False,
                                    params=self._params)
    data_collection = data_generator.run_scenarios()
    return data_collection

  def _transform_to_supervised_dataset(self, data_collection, observer):
    """Transforms data_collection into supervised dataset.

    Transforms list of lists of dicts to two arrays (X and Y).

    Args:
      data_collection: list of sceanrio_data (with scenario_data being
                       a list of datapoints (datapoint is a dict with
                       keys "observation" and "label").
      observer: bark-ml Observer instance (with which the data were generated).
    Returns:
      X: numpy array containing the feature data (shape: Nxobservation_size).
      Y: numpy array containing the label data (shape: Nx2).
    """
    Y = list()
    X = list()
    for data in data_collection:
      for data_point in data:
        # Get raw data
        observation = data_point["observation"]
        actions = data_point["label"]

        # Transform data to arrays
        observation = observation.numpy()  
        actions = np.array([actions["steering"], actions["acceleration"]])

        # Save in training data variables
        X.append(observation)
        Y.append(actions)
    
    # Convert to numpy arrays
    X = np.array(X)
    Y = np.array(Y)
    return X, Y