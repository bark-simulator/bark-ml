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

  def __call__(self, current_values, desired_values, logits, return_intermediate_losses=False):
    """
    Calculates loss.

    Arguments current_values and desired_values are expected to be
    dictionaries.

    If return_intermediate_losses==True, return not only the weighted loss,
    but also a dictionary with a loss for each of the value functions, e.g.:
      {"Return": 0.03, "Envelope": 0.02, "Collision": 0.01}
    """
    if logits:
      # Transform raw output to values between 0 and 1
      current_values = apply_sigmoid_to_dict(current_values)

    return self._calculate_weighted_loss(current_values, desired_values, logits,
                                         return_intermediate_losses=return_intermediate_losses)

  def _select_criterion(self, _logits, _value_func):
    return self._criterion

  def _calculate_weighted_loss(self, current_values, desired_values, logits,
                               return_intermediate_losses=False):
    losses = {}
    weights_sum = 0

    for value_func in current_values.keys():
      criterion = self._select_criterion(logits, value_func)
      weight = self._weights[value_func] if self._weights is not None else 1
      weights_sum += weight

      loss = criterion(current_values[value_func], desired_values[value_func])
      losses[value_func] = weight * loss

    weighted_loss = sum(losses.values()) / weights_sum

    if return_intermediate_losses:
      return weighted_loss, losses

    return weighted_loss


class LossMSE(Loss):
  def __init__(self, weights=None):
    criterion = nn.MSELoss()
    super(LossMSE, self).__init__(criterion, weights)


class LossBCE(Loss):
  def __init__(self, weights=None):
    criterion = nn.BCELoss()
    super(LossBCE, self).__init__(criterion, weights)
    self._criterion_logits = nn.BCEWithLogitsLoss()

  def _select_criterion(self, logits, _):
    if logits:
      return self._criterion_logits
    return self._criterion

  def __call__(self, current_values, desired_values, logits, return_intermediate_losses=False):
    return self._calculate_weighted_loss(current_values, desired_values, logits,
                                         return_intermediate_losses=return_intermediate_losses)


class LossHuber(Loss):

  class HuberCriterion:
    """
    torch1.6 does not provide the Huber loss (only available in version 1.9)
    """
    def __init__(self, delta, normalize=False):
      self.delta = delta
      if normalize:
        # Make the loss equal to 1 if the absolute difference between target
        # and prediction is 1
        max_loss = self._unnormalized_loss(torch.Tensor([1.0]),
                                           torch.Tensor([0.0])).item()
        self.normalizing_factor = 1 / max_loss
      else:
        self.normalizing_factor = 1

    def _unnormalized_loss(self, current_values, desired_values):
      error = current_values - desired_values
      loss = torch.where(error < self.delta,
                         error**2,
                         self.delta * (torch.abs(error) - self.delta / 2))
      return loss.sum() / current_values.data.nelement()

    def __call__(self, current_values, desired_values):
      return self.normalizing_factor * self._unnormalized_loss(
          current_values, desired_values)

  def __init__(self, weights=None, delta=None, normalize=False):
    if delta is None:
      criterion = self.HuberCriterion(delta=0.5, normalize=normalize)
      super(LossHuber, self).__init__(criterion, weights)
    else:
      self._criterions = {
          value_func: self.HuberCriterion(delta=d, normalize=normalize)
          for value_func, d in delta.items()
      }
      super(LossHuber, self).__init__(None, weights)

  def _select_criterion(self, _, value_func):
    if self._criterion is not None:
      return self._criterion
    return self._criterions[value_func]


class LossTukey(Loss):

  class TukeyCriterion:
    def __init__(self, c, normalize=False):
      self.c = c
      if normalize:
        # Make the loss equal to 1 if the absolute difference between target
        # and prediction is 1
        max_loss = self._unnormalized_loss(torch.Tensor([1.0]),
                                           torch.Tensor([0.0])).item()
        self.normalizing_factor = 1 / max_loss
      else:
        self.normalizing_factor = 1

    def _unnormalized_loss(self, current_values, desired_values):
      error = current_values - desired_values
      const = torch.ones_like(error) * self.c**2 / 6
      loss = torch.where(error < self.c,
                         self.c**2 / 6 * (1 - (1 - (error / self.c)**2)**3),
                         const)
      return loss.sum() / current_values.data.nelement()

    def __call__(self, current_values, desired_values):
      return self.normalizing_factor * self._unnormalized_loss(
          current_values, desired_values)

  def __init__(self, weights=None, c=0.5, normalize=False):
    criterion = self.TukeyCriterion(c, normalize=normalize)
    super().__init__(criterion, weights=weights)


class LossEpsInsensitiveHuber(Loss):

  class EpsInsensitiveHuberCriterion:
    def __init__(self, eps, delta, normalize=False):
      self.eps = eps
      self.delta = delta
      if normalize:
        max_loss = self._unnormalized_loss(torch.Tensor([1.0]),
                                           torch.Tensor([0.0])).item()
        self.normalizing_factor = 1 / max_loss
      else:
        self.normalizing_factor = 1

    def _unnormalized_loss(self, current_values, desired_values):
      abs_error = torch.abs(current_values - desired_values)
      loss = torch.where(
          abs_error > self.delta,
          2 * (self.delta - self.eps) * abs_error - self.delta**2 + self.eps**2,
          torch.Tensor([0.0]))
      loss = torch.where((abs_error <= self.delta) & (abs_error > self.eps),
                         (abs_error - self.eps)**2,
                         loss)
      return loss.sum() / current_values.data.nelement()

    def __call__(self, current_values, desired_values):
      return self.normalizing_factor * self._unnormalized_loss(
          current_values, desired_values)

  def __init__(self, weights=None, eps=0.01, delta=0.5, normalize=False):
    criterion = self.EpsInsensitiveHuberCriterion(eps, delta, normalize)
    super().__init__(criterion, weights=weights)


class LossRelative(Loss):
  def __init__(self, weights=None, eps=1e-6):
    criterion = self._loss
    self.eps = eps
    super().__init__(criterion, weights=weights)

  def _loss(self, current_values, desired_values):
    error = current_values - desired_values
    loss = torch.abs(error / (desired_values + self.eps))
    return loss.sum() / current_values.data.nelement()


class LossPolicyCrossEntropy(Loss):
  def __init__(self):
    pass

  def __call__(self, current_values, desired_values, logits, return_intermediate_losses=False):
    logsoftmax = nn.LogSoftmax(dim=1)
    target = desired_values["Policy"]
    pred = current_values["Policy"]
    loss = torch.mean(torch.sum(-target*logsoftmax(pred), 1))
    if return_intermediate_losses:  # Added to match the interface of other losses
      return loss, None
    return loss
