# Copyright (c) 2020 Patrick Hart, Julian Bernhard,
# Klemens Esterle, Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


import unittest
import pickle
import logging
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import time
import networkx as nx
import tensorflow as tf
import datetime

# BARK imports
from bark.runtime.commons.parameters import ParameterServer
from bark.runtime.viewer.matplotlib_viewer import MPViewer
from bark.runtime.viewer.video_renderer import VideoRenderer

# BARK-ML imports
from bark_ml.environments.blueprints import ContinuousHighwayBlueprint, \
  ContinuousMergingBlueprint, ContinuousIntersectionBlueprint
from bark_ml.environments.single_agent_runtime import SingleAgentRuntime
from bark_ml.library_wrappers.lib_tf_agents.agents import BehaviorSACAgent, BehaviorPPOAgent, BehaviorGraphSACAgent
from bark_ml.library_wrappers.lib_tf_agents.runners import SACRunner, PPORunner
from bark_ml.observers.graph_observer import GraphObserver
from supervised_learning.data_generation import DataGenerator

class PyGNNActorTests(unittest.TestCase):
    def setUp(self):
        ######################
        #    Parameter       #
        self.log_dir = "/home/silvan/working_bark/supervised_learning/logs/"
        self.epochs = 3
        self.batch_size = 32
        self.train_split = 0.8
        #self.test_split = 0.2 # results from  (1 - train_split)
        self.data_path = "/home/silvan/working_bark/supervised_learning/data/"
        ######################

        """Setting up the test case"""
        params = ParameterServer()
        self.observer = GraphObserver(params)

        try:
            scenarios = os.listdir(self.data_path)
            logging.info("Data is already generated - just load the data")

        except:
            logging.info("Starting data_generation")
            graph_generator = DataGenerator(num_scenarios=100, dump_dir=self.data_path, render=False)
            graph_generator.run_scenarios()

        finally:
            # Load raw data
            data_collection = list()
            scenarios = os.listdir(self.data_path)
            for scenario in scenarios:
                scenario_path = self.data_path + "/" + scenario
                with open(scenario_path, 'rb') as f:
                    data = pickle.load(f)
                data_collection.append(data)
            logging.info("Raw data loading completed")

        # Transform raw data to supervised dataset
        Y = list()
        X = list()
        for data in data_collection:
            for data_point in data:
                # Transform raw data to nx.Graph
                graph_data = data_point["graph"]
                actions = data_point["actions"]
                graph = nx.node_link_graph(graph_data)
                # Transform graph to observation
                observation = self.observer._observation_from_graph(graph).numpy()
                actions = np.array([actions["steering"], actions["acceleration"]])
                # Save in training data variables
                X.append(observation)
                Y.append(actions)
        logging.info("Transformation to supervised dataset completed")

        # Transform supervised dataset into tf.dataset
        self.X = tf.constant(X)
        self.Y = tf.constant(Y, dtype=tf.float32)
        dataset = tf.data.Dataset.from_tensor_slices((self.X, self.Y))
        dataset_size = self.X.shape[0]
        logging.info("Transformation to tf.dataset completed")

        # Train/Test split
        train_size = int(self.train_split * dataset_size)
        full_dataset = dataset.shuffle(dataset_size, seed=5)

        self.train_dataset = full_dataset.take(train_size).batch(self.batch_size)
        test_dataset = full_dataset.skip(train_size)
        self.test_dataset = test_dataset.take(-1).batch(self.batch_size)
        logging.info("Train/Test split completed")

        # Get actor net
        params = ParameterServer(filename="examples/example_params/tfa_params.json")
        params["ML"]["BehaviorTFAAgents"]["NumCheckpointsToKeep"] = None
        params["ML"]["SACRunner"]["EvaluateEveryNSteps"] = 50
        params["ML"]["BehaviorSACAgent"]["BatchSize"] = 32
        params["World"]["remove_agents_out_of_map"] = False
        
        bp = ContinuousHighwayBlueprint(params, number_of_senarios=2500, random_seed=0)
        env = SingleAgentRuntime(blueprint=bp, observer=self.observer, render=False)
        sac_agent = BehaviorGraphSACAgent(environment=env, params=params)
        actor_net = sac_agent._agent._actor_network
        self.actor_net = actor_net
        logging.info("Loading of actor net completed")
    
    def test_actor_network(self):
        # Evaluates actor net formalia
        actor_net = self.actor_net
        self.assertIsNotNone(actor_net)
    
    def test_training(self):
        self.model = self.actor_net
        self.loss_object = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam()
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.summary_writer = tf.summary.create_file_writer(self.log_dir+'/'+ current_time)

        for epoch in range(self.epochs):
            # Reset the metrics at the start of the next epoch
            self.train_loss.reset_states()
            self.test_loss.reset_states()
            logging.info('Starting training in epoch '+str(epoch))

            for inputs, labels in self.train_dataset:
                train_step(inputs, labels, model=self.model, loss_object=self.loss_object,
                                        optimizer=self.optimizer, train_loss=self.train_loss)
                
            for test_inputs, test_labels in self.test_dataset:
                test_step(test_inputs, test_labels, model=self.model, loss_object=self.loss_object,
                                      test_loss=self.test_loss)
            # Histogram data
            logging.info("starting to create histogram data")
            true_results, preds, abs_losses = create_histogram_data(self.train_dataset, self.model)

            with self.summary_writer.as_default():
                tf.summary.scalar('loss/train_loss', self.train_loss.result(), step=epoch)
                tf.summary.scalar('loss/test_loss', self.test_loss.result(), step=epoch)
                tf.summary.histogram("steering/labels", tf.constant(true_results[:,0]), step=epoch)
                tf.summary.histogram("steering/predictions", tf.constant(preds[:,0]), step=epoch)
                tf.summary.histogram("acceleration/labels", tf.constant(true_results[:,1]), step=epoch)
                tf.summary.histogram("acceleration/predictions", tf.constant(preds[:,1]), step=epoch)
                tf.summary.histogram("absolute_losses/steering", tf.constant(abs_losses[:,0]), step=epoch)
                tf.summary.histogram("absolute_losses/acceleration", tf.constant(abs_losses[:,1]), step=epoch)


            template = 'Epoch {}, Loss: {}, Test Loss: {}'
            print(template.format(epoch + 1,
                                    self.train_loss.result(),
                                    self.test_loss.result()))
       
        # Show exemplary predictions of model
        abs_losses = list()
        predictions = list()
        true_results = list()
        for input_data, y_true in self.train_dataset:
            result = self.model(input_data, training=False)
            y_pred = result[0].mean()

            predictions.extend(y_pred.numpy())
            true_results.extend(y_true.numpy())
            absolute_loss = y_true.numpy() - y_pred.numpy()
            abs_losses.extend(absolute_loss)
        
        # Visualize predictions
        abs_losses = np.array(abs_losses)
        predictions = np.array(predictions)
        true_results = np.array(true_results)
        colors = ['b','r'] 
        fig, axes = plt.subplots(nrows=4)
        # Hist of steering
        y1 = true_results[:,0]
        y2 = predictions[:,0]
        axes[0].hist([y1,y2], bins=100, color=colors)
        axes[0].legend(["Label", "Prediction"])
        #axes[0].set_xlim(-10,10) 
        axes[0].set_ylabel("Counts")
        axes[0].set_xlabel("Steering")
        axes[0].legend(loc='upper right')

        # Hist of acc
        y3 = true_results[:,1]
        y4 = predictions[:,1]
        axes[1].hist([y3,y4], bins=100, color=colors) 
        axes[1].set_ylabel("Counts")
        axes[1].set_xlabel("Acceleration")
        axes[1].legend(["Label", "Prediction"])

        # Hist of abs steer losses
        y5 = abs_losses[:,0]
        axes[2].hist(y5, bins=200) 
        axes[2].set_ylabel("Counts")
        axes[2].set_xlabel("Absolute losses steering (True-Pred)")

        # Hist of abs steer losses
        y6 = abs_losses[:,1]
        axes[3].hist(y6, bins=200) 
        axes[3].set_ylabel("Counts")
        axes[3].set_xlabel("Absolute losses acceleration (True-Pred)")

        plt.tight_layout() 
        plt.show()

def train_step(inputs, labels, model, loss_object, optimizer, train_loss):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(inputs, training=True)
        predictions = predictions[0].mean()
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    #train_accuracy(labels, predictions)
    
def test_step(inputs, labels, model, loss_object, test_loss):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(inputs, training=False)
    predictions = predictions[0].mean()
    t_loss = loss_object(labels, predictions)
    test_loss(t_loss)
    #test_accuracy(labels, predictions)

def create_histogram_data(dataset, model):
    abs_losses = list()
    preds = list()
    true_results = list()

    # Create raw data
    for input_data, y_true in dataset:
        result = model(input_data, training=False)
        y_pred = result[0].mean()

        preds.extend(y_pred.numpy())
        true_results.extend(y_true.numpy())
        absolute_loss = tf.math.subtract(y_true, y_pred)
        absolute_loss = y_true.numpy() - y_pred.numpy()
        abs_losses.extend(absolute_loss)

    abs_losses = np.array(abs_losses)
    preds = np.array(preds)
    true_results = np.array(true_results)
    return true_results, preds, abs_losses    

def create_tf_hist(values, bins):
    values = np.array(values)
    counts, bin_edges = np.histogram(values, bins=bins)

    # Fill fields of histogram proto
    hist = tf.HistogramProto()
    hist.min = float(np.min(values))
    hist.max = float(np.max(values))
    hist.num = int(np.prod(values.shape))
    hist.sum = float(np.sum(values))
    hist.sum_squares = float(np.sum(values**2))

    bin_edges = bin_edges[1:]
    # Add bin edges and counts
    for edge in bin_edges:
        hist.bucket_limit.append(edge)
    for c in counts:
        hist.bucket.append(c)
    return hist

if __name__ == '__main__':
    unittest.main()
