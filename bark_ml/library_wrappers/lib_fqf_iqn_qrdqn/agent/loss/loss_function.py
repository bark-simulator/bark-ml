from torch import sigmoid, nn
import torch

def apply_sigmoid_to_dict(dict_values):
  """
  Returns a new dictionary.
  """
  sigmoid_values = dict()
  for key in dict_values.keys():
    sigmoid_values[key] = sigmoid(dict_values[key])
  return sigmoid_values


class Loss:
  """
  A helper class for calculating loss.
  """
  def __init__(self, criterion, weights=None):
    self._criterion = criterion
    self._weights = weights

  def __call__(self, current_values, desired_values, logits):
    """
    Calculates loss.

    Arguments current_values and desired_values are expected to be
    dictionaries.
    """
    if logits:
      # Transform raw output to values between 0 and 1
      current_values = apply_sigmoid_to_dict(current_values)

    return self._calculate_weighted_loss(current_values, desired_values,
                                         logits)

  def _select_criterion(self, _):
    return self._criterion

  def _calculate_weighted_loss(self, current_values, desired_values, logits):
    losses = []
    weights_sum = 0
    criterion = self._select_criterion(logits)

    for key in current_values.keys():
      weight = self._weights[key] if self._weights is not None else 1
      weights_sum += weight

      loss = criterion(current_values[key], desired_values[key])
      losses.append(weight * loss)

    return sum(losses) / weights_sum


class LossMSE(Loss):
  def __init__(self, weights=None):
    criterion = nn.MSELoss()
    super(LossMSE, self).__init__(criterion, weights)


class LossBCE(Loss):
  def __init__(self, weights=None):
    criterion = nn.BCELoss()
    super(LossBCE, self).__init__(criterion, weights)
    self._criterion_logits = nn.BCEWithLogitsLoss()

  def _select_criterion(self, logits):
    if logits:
      return self._criterion_logits
    return self._criterion

  def __call__(self, current_values, desired_values, logits):
    return self._calculate_weighted_loss(current_values, desired_values,
                                         logits)

class LossPolicyCrossEntropy(Loss):
  def __init__(self):
    pass

  def __call__(self, current_values, desired_values, logits):
    logsoftmax = nn.LogSoftmax(dim=1)
    target = desired_values["Policy"]
    pred = current_values["Policy"]
    return torch.mean(torch.sum(-target*logsoftmax(pred), 1))