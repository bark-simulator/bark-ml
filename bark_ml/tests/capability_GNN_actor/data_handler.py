# Copyright (c) 2020 Patrick Hart, Julian Bernhard,
# Klemens Esterle, Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import os
import unittest
import pickle
import logging
import numpy as np

import time
import networkx as nx
import tensorflow as tf

# Supervised learning imports
from supervised_learning.data_generation import DataGenerator

class Dataset:
  def __init__(self, data_path, observer, params, batch_size=32, train_split=0.8, num_scenarios=100):
    self.observer = observer
    self.data_path = data_path
    self.params = params

    self.batch_size = batch_size
    self.train_split = train_split
    self.num_scenarios = num_scenarios
        
  def get_datasets(self):
    try:
      scenarios = os.listdir(self.data_path)
      logging.debug("Data is already generated")
    except FileNotFoundError:
      logging.debug("Starting data_generation")
      self.generate_data(data_path=self.data_path, num_scenarios=self.num_scenarios)
    finally:
      #logging.debug("Starting to load the data")
      data_collection = self.load(self.data_path)
      X, Y = self.transform_into_supervised_dataset(data_collection, observer=self.observer)
      logging.info("len(X):"+str(len(X))+"len(x[0]):"+str(len(X[0])))
      #logging.info("Transformation to supervised dataset completed")
      self.transform_to_tensorflow_datasets(X, Y, train_split=self.train_split,
                                              batch_size=self.batch_size)

  def load(self, data_path):
    data_collection = list()
    scenarios = os.listdir(self.data_path)
    for scenario in scenarios:
      scenario_path = self.data_path + "/" + scenario
      with open(scenario_path, 'rb') as f:
        data = pickle.load(f)
      data_collection.append(data)
    return data_collection

  def transform_to_tensorflow_datasets(self, X, Y, train_split=0.8, batch_size=32):
    # Transform supervised dataset into tf.dataset
    X = tf.constant(X)
    Y = tf.constant(Y, dtype=tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset_size = X.shape[0]

    # Train/Test split
    train_size = int(train_split * dataset_size)
    full_dataset = dataset.shuffle(dataset_size, seed=5)
    self.train_dataset = full_dataset.take(train_size).batch(batch_size)
    test_dataset = full_dataset.skip(train_size)
    self.test_dataset = test_dataset.take(-1).batch(batch_size)
    return self.train_dataset, self.test_dataset

  def generate_data(self, data_path, num_scenarios):
    logging.info("Starting data_generation")
    graph_generator = DataGenerator(num_scenarios=num_scenarios, dump_dir=data_path, render=False, params=self.params)
    graph_generator.run_scenarios()

  def transform_into_supervised_dataset(self, data_collection, observer):
    # Transform raw data to supervised dataset
    Y = list()
    X = list()
    for data in data_collection:
      for data_point in data:
        # Get raw data
        observation = data_point["graph"]
        actions = data_point["actions"]
        # Transform observation tensor to array
        try:
          observation = observation.numpy()
        except AttributeError:
          graph_data = data_point["graph"]
          actions = data_point["actions"]
          graph = nx.node_link_graph(graph_data)
          # Transform graph to observation
          observation = observer._observation_from_graph(graph).numpy()
      
        actions = np.array([actions["steering"], actions["acceleration"]])
        # Save in training data variables
        X.append(observation)
        Y.append(actions)
    return X, Y