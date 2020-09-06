# Copyright (c) 2020 fortiss GmbH
#
# Authors: Silvan Wimmer
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import logging
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime

class Learner:
  """Learner class handling the supervised learning of the model.

  Args:
    model: trainable tf_agents model
    train_dataset: tf.data.Dataset for training
    test_dataset: tf.data.Dataset for testing/validation
    log_dir: path, specifys directory of storing tf.summarys
  """
  def __init__(self, model, train_dataset, test_dataset, log_dir=None):
    """Initialization of the Learner"""
    self._model = model
    self._train_dataset = train_dataset
    self._test_dataset = test_dataset
    self._log_dir = log_dir

  def train(self, epochs=10, only_test=False, mode="Distribution"):
    """Trains the model supervised for N epochs.
        
    Args:
      epochs: int, number of epochs.
      only_test: bool, if false: no real training, just tests.
      mode: str, "Distribution" or sth else (Distribution is necessary
            for all models outputing a distribution).
    
    Returns:
      losses: dict with keys "train" and "test" corresponding each to a list
              of losses (one loss per epoch)
    """
    self._loss_object = tf.keras.losses.MeanSquaredError()
    self._optimizer = tf.keras.optimizers.Adam()
    self._train_loss = tf.keras.metrics.Mean(name='train_loss')
    self._test_loss = tf.keras.metrics.Mean(name='test_loss')

    # Create new directory for current summary data with time information
    if self._log_dir is not None:
      current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
      self._summary_writer = tf.summary.create_file_writer(self._log_dir+'/'+ \
                                                           current_time)

    history_train_loss = list()
    history_test_loss = list()
    
    for epoch in range(epochs):
      self._train_loss.reset_states()
      self._test_loss.reset_states()

      if not only_test:
        # Start real training cycle
        for inputs, labels in self._train_dataset:
          self._train_step(inputs, labels, model=self._model,
                           loss_object=self._loss_object,
                           optimizer=self._optimizer,
                           train_loss=self._train_loss, mode=mode)

      elif only_test:
        # Only evaluate actor on training set (for constant and random actor)
        for test_inputs, test_labels in self._train_dataset:
          self._test_step(test_inputs, test_labels, model=self._model,
                          loss_object=self._loss_object,
                          test_loss=self._train_loss, mode=mode)

      # Evaluate model on test_dataset
      for test_inputs, test_labels in self._test_dataset:
        self._test_step(test_inputs, test_labels, model=self._model,
                        loss_object=self._loss_object,
                        test_loss=self._test_loss, mode=mode)

      if self._log_dir is not None:
        # Create histogram data for better understanding of predictions of model
        true_results, preds, abs_losses = self._create_histogram_data(\
              self._train_dataset, self._model, mode=mode)

        # Write summary data to file
        with self._summary_writer.as_default():
          tf.summary.scalar('loss/train_loss', self._train_loss.result(),
                            step=epoch)
          tf.summary.scalar('loss/test_loss', self._test_loss.result(),
                            step=epoch)
          tf.summary.histogram("steering/labels", tf.constant(true_results[:,0]),
                              step=epoch)
          tf.summary.histogram("steering/predictions", tf.constant(preds[:,0]),
                              step=epoch)
          tf.summary.histogram("acceleration/labels",
                              tf.constant(true_results[:,1]), step=epoch)
          tf.summary.histogram("acceleration/predictions", tf.constant(preds[:,1]),
                              step=epoch)
          tf.summary.histogram("absolute_losses/steering",
                              tf.constant(abs_losses[:,0]), step=epoch)
          tf.summary.histogram("absolute_losses/acceleration",
                              tf.constant(abs_losses[:,1]), step=epoch)
      
      # Save losses in history_lists
      history_train_loss.append(self._train_loss.result().numpy())
      history_test_loss.append(self._test_loss.result().numpy())
    losses = dict()
    losses["train"] = history_train_loss
    losses["test"] = history_test_loss
    return losses

  def visualize_predictions(self, dataset, title=None, mode="Distribution"):
    """Show output of model as an figure with 4 subplots.

    Args:
      dataset: tf.data.Dataset to evaluate.
      title: str
      mode: str ("Distribution" or sth else, see definition of base class)
    """
    abs_losses = list()
    predictions = list()
    true_results = list()
    for input_data, y_true in dataset:
      # Get predictions of model
      y_pred = self._model(input_data, training=False, step_type=None,
                           network_state=())
      if mode=="Distribution":
        y_pred = y_pred[0].mean()

      predictions.extend(y_pred.numpy())
      true_results.extend(y_true.numpy())
      # Calculate losses of predictions
      absolute_loss = y_true.numpy() - y_pred.numpy()
      abs_losses.extend(absolute_loss)
    
    # Visualize predictions with true labels and losses
    abs_losses = np.array(abs_losses)
    predictions = np.array(predictions)
    true_results = np.array(true_results)
    fig = self._create_figure(abs_losses, predictions, true_results, title)
    plt.show()

  @staticmethod
  def _create_figure(abs_losses, predictions, true_results, title=None):
    """Creates figure with losses, predictions and labels.

    Creates 4 histograms. The first one shows the predictions and the labels of
    the steering. The second one shows the predictions and the labels for the
    acceleration. The third one shows the absolute losses of the steering
    predictions. The fourth one shows the losses of the acceleration
    predictions.

    Args:
      abs_losses: np.array of absolute losses.
      predictions: np.array of predictions.
      true_results: np.array of labels.
      title: str

    Returns:
      fig: matplotlib Figure
    """
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
  def _train_step(inputs,
                  labels,
                  model,
                  loss_object,
                  optimizer,
                  train_loss,
                  mode="Distribution"):
    """Supervised training step.

    Args:
      inputs: batch of features
      labels: batch of labels
      model: trainable model (tf model or tf_agent)
      loss_object: tf.keras.losses
      optimizer: tf.keras.optimizer
      train_loss: tf.keras.metrics
      mode: str, see Class definition
    """
    with tf.GradientTape() as tape:
      predictions = model(inputs, training=True, step_type=None,
                          network_state=())
      if mode=="Distribution":
        predictions = predictions[0].mean()
      loss = loss_object(labels, predictions)
    
    # Calculate gradients for trainable variables and apply them
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)

  @staticmethod 
  def _test_step(inputs,
                  labels,
                  model,
                  loss_object,
                  test_loss,
                  mode="Distribution"):
    """Supervised test step.

    Args:
      inputs: batch of features
      labels: batch of labels
      model: trainable model (tf model or tf_agent)
      loss_object: tf.keras.losses
      optimizer: tf.keras.optimizer
      test_loss: tf.keras.metrics for testing
      mode: str, see Class definition
    """
    predictions = model(inputs, training=False, step_type=None,
                        network_state=())
    if mode=="Distribution":
      predictions = predictions[0].mean()

    t_loss = loss_object(labels, predictions)
    test_loss(t_loss)

  @staticmethod
  def _create_histogram_data(dataset, model, mode="Distribution"):
    """Creates the histogram data for tensorboard.

    Args:
      dataset: tf.data.Dataset, for which to create a histogram
      model: model, with which to creat histogram data
      mode: str, see Class Definition

    Returns:
      true_results: np.array with labels
      preds: np.array with predictions
      abs_losses: np.array with absolute losses
    """
    abs_losses = list()
    preds = list()
    true_results = list()

    # Create raw data
    for input_data, y_true in dataset:
      y_pred = model(input_data, training=False, step_type=None,
                     network_state=())
      if mode=="Distribution":
        y_pred = y_pred[0].mean()

      preds.extend(y_pred.numpy())
      true_results.extend(y_true.numpy())
      # Calc absolute loss
      absolute_loss = tf.math.subtract(y_true, y_pred)
      absolute_loss = y_true.numpy() - y_pred.numpy()
      abs_losses.extend(absolute_loss)

    abs_losses = np.array(abs_losses)
    preds = np.array(preds)
    true_results = np.array(true_results)
    return true_results, preds, abs_losses