# Copyright (c) 2020 Patrick Hart, Julian Bernhard,
# Klemens Esterle, Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


import logging
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import time
import networkx as nx
import tensorflow as tf
import datetime

# Supervised learning imports
from supervised_learning.actor_nets import ConstantActorNet, RandomActorNet, \
    get_GNN_SAC_actor_net, get_SAC_actor_net

class Learner:
    def __init__(self, model, train_dataset, test_dataset, log_dir):
        """Function to set up the Learner"""
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.log_dir = log_dir

    def train(self, epochs=10, only_test=False, mode="Distribution"):
        """Function to train the model
            
            optional parameters:
                epochs      :   int     #number of epochs 
                only_test   :   bool    #if false: no real training, just tests
                mode        :   str     #"Distribution" or sth else"""

        self.loss_object = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam()
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.summary_writer = tf.summary.create_file_writer(self.log_dir+'/'+ current_time)

        for epoch in range(epochs):
            # Reset the metrics at the start of the next epoch
            self.train_loss.reset_states()
            self.test_loss.reset_states()
            logging.info('Starting training in epoch '+str(epoch))
            # Start real training with gradients or only evaluate
            if only_test==False:
                for inputs, labels in self.train_dataset:
                    self.train_step(inputs, labels, model=self.model, loss_object=self.loss_object,
                                    optimizer=self.optimizer, train_loss=self.train_loss, mode=mode)
            elif only_test==True:
                for test_inputs, test_labels in self.train_dataset:
                    self.test_step(test_inputs, test_labels, model=self.model, loss_object=self.loss_object,
                                   test_loss=self.train_loss, mode=mode)
            # Evaluate on test_dataset
            for test_inputs, test_labels in self.test_dataset:
                self.test_step(test_inputs, test_labels, model=self.model, loss_object=self.loss_object,
                               test_loss=self.test_loss, mode=mode)
            # Histogram data
            logging.info("starting to create histogram data")
            true_results, preds, abs_losses = self.create_histogram_data(self.train_dataset, self.model,
                                                                         mode=mode)

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

    def visualize_predictions(self, dataset, title=None, mode="Distribution"):
        # Show exemplary predictions of model
        abs_losses = list()
        predictions = list()
        true_results = list()
        for input_data, y_true in dataset:
            y_pred = self.model(input_data, training=False, step_type=None, network_state=())
            if mode=="Distribution":
                y_pred = y_pred[0].mean()

            predictions.extend(y_pred.numpy())
            true_results.extend(y_true.numpy())
            absolute_loss = y_true.numpy() - y_pred.numpy()
            abs_losses.extend(absolute_loss)
        
        # Visualize predictions
        abs_losses = np.array(abs_losses)
        predictions = np.array(predictions)
        true_results = np.array(true_results)
        fig = self.create_figure(abs_losses, predictions, true_results, title)
        plt.show()

    @staticmethod
    def create_figure(abs_losses, predictions, true_results, title=None):
        colors = ['b','r'] 
        fig, axes = plt.subplots(nrows=4)
        fig.suptitle(title, fontsize=16)
        # Hist of steering
        y1 = true_results[:,0]
        y2 = predictions[:,0]
        __ = axes[0].hist([y1,y2], bins=100, color=colors)
        axes[0].legend(["Label", "Prediction"])
        #axes[0].set_xlim(-10,10) 
        axes[0].set_ylabel("Counts")
        axes[0].set_xlabel("Steering")
        axes[0].legend(loc='upper right')

        # Hist of acc
        y3 = true_results[:,1]
        y4 = predictions[:,1]
        __ = axes[1].hist([y3, y4], bins=100, color=colors) 
        axes[1].set_ylabel("Counts")
        axes[1].set_xlabel("Acceleration")
        axes[1].legend(["Label", "Prediction"])

        # Hist of abs steer losses
        y5 = abs_losses[:,0]
        __ = axes[2].hist(y5, bins=200) 
        axes[2].set_ylabel("Counts")
        axes[2].set_xlabel("Absolute losses steering (True-Pred)")

        # Hist of abs steer losses
        y6 = abs_losses[:,1]
        __ = axes[3].hist(y6, bins=200) 
        axes[3].set_ylabel("Counts")
        axes[3].set_xlabel("Absolute losses acceleration (True-Pred)")

        fig.tight_layout() 
        return fig

    @staticmethod
    def train_step(inputs, labels, model, loss_object, optimizer, train_loss, mode="Distribution"):
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True, step_type=None, network_state=())
            if mode=="Distribution":
                predictions = predictions[0].mean()
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)

    @staticmethod 
    def test_step(inputs, labels, model, loss_object, test_loss, mode="Distribution"):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(inputs, training=False, step_type=None, network_state=())
        if mode=="Distribution":
            predictions = predictions[0].mean()
        t_loss = loss_object(labels, predictions)
        test_loss(t_loss)
        #test_accuracy(labels, predictions)

    @staticmethod
    def create_histogram_data(dataset, model, mode="Distribution"):
        abs_losses = list()
        preds = list()
        true_results = list()

        # Create raw data
        for input_data, y_true in dataset:
            y_pred = model(input_data, training=False, step_type=None, network_state=())
            if mode=="Distribution":
                y_pred = y_pred[0].mean()

            preds.extend(y_pred.numpy())
            true_results.extend(y_true.numpy())
            absolute_loss = tf.math.subtract(y_true, y_pred)
            absolute_loss = y_true.numpy() - y_pred.numpy()
            abs_losses.extend(absolute_loss)

        abs_losses = np.array(abs_losses)
        preds = np.array(preds)
        true_results = np.array(true_results)
        return true_results, preds, abs_losses